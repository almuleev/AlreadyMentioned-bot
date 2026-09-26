"""Administrator confirmation of a question–answer reply chain."""

import logging
from time import monotonic

from aiogram import Bot, Router
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import Message

from already_mentioned.bot.permissions import is_chat_admin
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.embeddings import EmbeddingService, normalize_vector
from already_mentioned.services.reply_cache import PendingAnswer, ReplyCache

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
SOLVE_USAGE = (
    "Как сохранить решение: администратор отвечает на исходный вопрос текстовым "
    "сообщением, затем отвечает на свой ответ командой /solve. "
    "Сделайте это, пока бот запущен и помнит цепочку ответов."
)


@router.message(Command("solve"))
async def solve(
    message: Message,
    bot: Bot,
    reply_cache: ReplyCache,
    chats: ChatRepository,
    solutions: SolutionRepository,
    embeddings: EmbeddingService,
) -> None:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("/solve работает только в группе или супергруппе.")
        return
    if message.from_user is None or not await is_chat_admin(
        bot, message.chat.id, message.from_user.id
    ):
        await message.answer("Сохранять решения может только администратор чата.")
        return

    answer = message.reply_to_message
    if (
        answer is None
        or not answer.text
        or answer.from_user is None
        or answer.from_user.id != message.from_user.id
    ):
        await message.answer(SOLVE_USAGE)
        return

    pending = reply_cache.get(message.chat.id, answer.message_id)
    if pending is None or pending.answer_author_id != message.from_user.id:
        await message.answer(SOLVE_USAGE)
        return

    answer_link = answer.get_url()
    if not pending.question_link or not answer_link:
        await message.answer(
            "В этой группе Telegram не даёт прямую ссылку на сообщение. "
            "Для сохранения решений нужна супергруппа."
        )
        return

    await chats.ensure_chat(message.chat.id, message.chat.title or "")
    existing = await solutions.get_by_pair(
        message.chat.id, pending.question_message_id, answer.message_id
    )
    if existing is not None:
        await message.answer("Эта пара «вопрос → ответ» уже сохранена.")
        return
    try:
        embedding = normalize_vector(
            await embeddings.embed_passage(pending.question_text)
        ).tobytes()
    except Exception:
        logging.exception("Could not embed confirmed question")
        await message.answer(
            "Не удалось подготовить локальную модель поиска. "
            "Решение не сохранено; попробуйте /solve позже."
        )
        return
    solution_id = await solutions.add_solution(
        chat_id=message.chat.id,
        question_message_id=pending.question_message_id,
        answer_message_id=answer.message_id,
        question_text=pending.question_text,
        answer_text=answer.text,
        question_embedding=embedding,
        question_link=pending.question_link,
        answer_link=answer_link,
    )
    if solution_id is None:
        await message.answer("Эта пара «вопрос → ответ» уже сохранена.")
    else:
        await message.answer("Решение сохранено.")


async def remember_admin_answer(
    message: Message, bot: Bot, reply_cache: ReplyCache
) -> bool:
    """Retain only admin text replies that can become confirmed solutions."""
    if message.chat.type not in GROUP_TYPES or message.from_user is None:
        return False
    if message.text is None or message.text.startswith("/"):
        return False
    question = message.reply_to_message
    if question is None or not question.text or question.text.startswith("/"):
        return False
    if not await is_chat_admin(bot, message.chat.id, message.from_user.id):
        return False
    reply_cache.remember(
        PendingAnswer(
            chat_id=message.chat.id,
            answer_message_id=message.message_id,
            answer_author_id=message.from_user.id,
            question_message_id=question.message_id,
            question_text=question.text,
            question_link=question.get_url(),
            recorded_at=monotonic(),
        )
    )
    return True
