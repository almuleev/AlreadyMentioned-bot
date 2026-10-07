"""Long-polling entry point."""

import asyncio
import logging
from contextlib import suppress

from aiogram import Dispatcher

from already_mentioned.bot.factory import create_bot
from already_mentioned.config import Settings, load_settings
from already_mentioned.database.connection import connect_database
from already_mentioned.database.embedding_contract import require_frida
from already_mentioned.database.process_lock import database_process_lock
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers.feedback import router as feedback_router
from already_mentioned.handlers.management import router as management_router
from already_mentioned.handlers.messages import router as messages_router
from already_mentioned.handlers.solve import router as solve_router
from already_mentioned.handlers.start import router as start_router
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.confirmations import ForgetConfirmations
from already_mentioned.services.embeddings import FridaEmbeddingService
from already_mentioned.services.feedback_cleanup import FeedbackCleanup
from already_mentioned.services.reply_cache import ReplyCache
from already_mentioned.services.reranking import OnnxReranker
from already_mentioned.services.search import SearchService


async def run() -> None:
    settings = load_settings()
    with database_process_lock(settings.database_path):
        await _run(settings)


async def _run(settings: Settings) -> None:
    bot = create_bot(settings)
    dispatcher = Dispatcher()
    dispatcher.include_router(start_router)
    dispatcher.include_router(solve_router)
    dispatcher.include_router(management_router)
    dispatcher.include_router(messages_router)
    dispatcher.include_router(feedback_router)
    database = None
    cleanup_task = None
    try:
        database = await connect_database(settings.database_path)
        await initialize_database(database)
        await require_frida(database, adopt_empty=True)
        chats = ChatRepository(database)
        solutions = SolutionRepository(database)
        embeddings = FridaEmbeddingService()
        # Fail before polling if the pinned cache cannot be loaded.
        await embeddings.embed_query("Проверка загрузки")
        reranker = None
        if settings.search_mode == "hybrid":
            reranker = OnnxReranker(local_only=True, batch_size=1)
            await asyncio.to_thread(
                reranker.predict,
                "Проверка загрузки",
                ["Вопрос: Проверка\nОтвет: Проверка"],
            )
        search = SearchService(
            embeddings, chats, solutions, mode=settings.search_mode, reranker=reranker
        )
        logging.info("Search ready: %s; MiniLM batch_size=1", search.mode_label)
        feedback = FeedbackRepository(database)
        feedback_cleanup = FeedbackCleanup(bot, feedback)
        cleanup_task = asyncio.create_task(feedback_cleanup.run())
        await dispatcher.start_polling(
            bot,
            close_bot_session=False,
            reply_cache=ReplyCache(),
            confirmations=ForgetConfirmations(),
            chats=chats,
            solutions=solutions,
            feedback=feedback,
            feedback_cleanup=feedback_cleanup,
            embeddings=embeddings,
            search=search,
        )
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
        if database is not None:
            await database.close()
        await bot.session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.info("Bot stopped")


if __name__ == "__main__":
    main()
