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
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reply_cache import PendingAnswer, ReplyCache

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
SOLVE_USAGE = (
    "Ответ администратора на вопрос сохраняется автоматически, если вопрос "
    "прошёл фильтр. Для ручного сохранения ответьте на свой ответ командой "
    "/solve, пока бот помнит цепочку."
)


async def autosave_admin_answer(
    message: Message,
    reply_cache: ReplyCache,
    chats: ChatRepository,
    solutions: SolutionRepository,
    embeddings: EmbeddingService,
) -> None:
    """Silently store an eligible administrator reply; never post to the chat."""
    pending = reply_cache.get(message.chat.id, message.message_id)
    if pending is None or not is_question_candidate(pending.question_text):
        return
    answer_link = message.get_url()
    if not pending.question_link or not answer_link:
        return
    if (
        await solutions.get_by_pair(
            message.chat.id, pending.question_message_id, message.message_id
        )
        is not None
    ):
        return
    try:
        embedding = normalize_vector(
            await embeddings.embed_passage(pending.question_text)
        ).tobytes()
        if reply_cache.get(message.chat.id, message.message_id) is None:
            return
        await chats.ensure_chat(message.chat.id, message.chat.title or "")
        if reply_cache.get(message.chat.id, message.message_id) is None:
            return
        await solutions.add_solution(
            chat_id=message.chat.id,
            question_message_id=pending.question_message_id,
            answer_message_id=message.message_id,
            question_text=pending.question_text,
            answer_text=message.text,
            question_embedding=embedding,
            question_link=pending.question_link,
            answer_link=answer_link,
        )
        if reply_cache.get(message.chat.id, message.message_id) is None:
            await solutions.delete_by_answer(message.chat.id, message.message_id)
    except Exception:
        logging.exception(
            "Could not automatically save answer %s in chat %s",
            message.message_id,
            message.chat.id,
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


@router.message(Command("undo"))
async def undo(
    message: Message,
    bot: Bot,
    reply_cache: ReplyCache,
    solutions: SolutionRepository,
) -> None:
    """Undo a saved answer by replying to it, without affecting other chats."""
    if message.chat.type not in GROUP_TYPES:
        await message.answer("/undo работает только в группе или супергруппе.")
        return
    if message.from_user is None or not await is_chat_admin(
        bot, message.chat.id, message.from_user.id
    ):
        await message.answer("Удалять решения может только администратор чата.")
        return
    answer = message.reply_to_message
    if answer is None:
        await message.answer("Ответьте командой /undo на сохранённый ответ.")
        return
    pending = reply_cache.get(message.chat.id, answer.message_id)
    deleted = await solutions.delete_by_answer(message.chat.id, answer.message_id)
    reply_cache.discard(message.chat.id, answer.message_id)
    await message.answer(
        "Решение удалено."
        if deleted
        else (
            "Сохранение отменено."
            if pending is not None and is_question_candidate(pending.question_text)
            else "Для этого ответа нет сохранённого решения."
        )
    )


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
    if question.from_user is not None and getattr(question.from_user, "is_bot", False):
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
