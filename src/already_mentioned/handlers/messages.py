"""Reply with one saved answer for a likely new question."""

import logging
from html import escape

from aiogram import Bot, F, Router
from aiogram.enums import ChatType, ParseMode
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from already_mentioned.handlers.solve import (
    autosave_admin_answer,
    remember_admin_answer,
)
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.embeddings import EmbeddingService
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reply_cache import ReplyCache
from already_mentioned.services.search import SearchService

router = Router()
MAX_INLINE_ANSWER_LENGTH = 3000


def format_answer_preview(answer: str) -> str:
    """Show saved text as a safe Telegram HTML quote within message limits."""
    shortened = len(answer) > MAX_INLINE_ANSWER_LENGTH
    preview = answer[:MAX_INLINE_ANSWER_LENGTH].rstrip() if shortened else answer
    result = f"Похожий ответ из этого чата:\n<blockquote>{escape(preview)}</blockquote>"
    if shortened:
        result += "\nОтвет сокращён. Полный текст — в оригинале."
    return result


@router.message(F.text)
async def handle_text(
    message: Message,
    bot: Bot,
    reply_cache: ReplyCache,
    search: SearchService,
    chats: ChatRepository,
    solutions: SolutionRepository,
    embeddings: EmbeddingService,
) -> None:
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return
    if message.from_user is None or message.from_user.is_bot or not message.text:
        return
    if message.text.startswith("/"):
        return
    if message.reply_to_message is not None:
        if await remember_admin_answer(message, bot, reply_cache):
            await autosave_admin_answer(
                message, reply_cache, chats, solutions, embeddings
            )
            return
    if not is_question_candidate(message.text):
        return
    try:
        match = await search.find_best(message.chat.id, message.text)
    except Exception:
        logging.exception("Search failed for chat %s", message.chat.id)
        return
    if match is None:
        return
    callback_prefix = f"vote:{match.solution.id}:{message.message_id}"
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👍 Помогло", callback_data=f"{callback_prefix}:helpful"
                ),
                InlineKeyboardButton(
                    text="👎 Не подходит",
                    callback_data=f"{callback_prefix}:not_helpful",
                ),
            ],
            [InlineKeyboardButton(text="📎 Оригинал", url=match.solution.answer_link)],
        ]
    )
    await message.reply(
        format_answer_preview(match.solution.answer_text),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
