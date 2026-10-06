"""Accept feedback only from the question author while its buttons are active."""

from time import time

from aiogram import F, Router
from aiogram.types import CallbackQuery

from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.feedback_cleanup import (
    FEEDBACK_LIFETIME_SECONDS,
    FeedbackCleanup,
)

router = Router()


@router.callback_query(F.data.startswith("vote:"))
async def record_feedback(
    callback: CallbackQuery,
    solutions: SolutionRepository,
    feedback: FeedbackRepository,
    feedback_cleanup: FeedbackCleanup,
) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 5 or parts[4] not in {"helpful", "not_helpful"}:
        await callback.answer("Некорректная оценка.")
        return
    try:
        solution_id, query_message_id = int(parts[1]), int(parts[2])
        author_id = int(parts[3])
    except ValueError:
        await callback.answer("Некорректная оценка.")
        return
    if callback.message is None or callback.message.date.timestamp() == 0:
        await callback.answer("Сообщение больше недоступно.")
        return
    saved_author_id = await feedback.keyboard_author(
        callback.message.chat.id, callback.message.message_id
    )
    if saved_author_id is None:
        await callback.answer("Оценка для этого сообщения больше недоступна.")
        return
    if callback.from_user.id != saved_author_id or author_id != saved_author_id:
        await callback.answer("Оценить ответ может только автор вопроса.")
        return
    if time() >= callback.message.date.timestamp() + FEEDBACK_LIFETIME_SECONDS:
        await callback.answer("Время для оценки истекло.")
        await feedback_cleanup.remove(
            callback.message.chat.id, callback.message.message_id
        )
        return
    solution = await solutions.get_for_chat(callback.message.chat.id, solution_id)
    if solution is None:
        await callback.answer("Решение больше недоступно.")
        return
    created = await feedback.add_feedback(
        solution_id=solution_id,
        query_message_id=query_message_id,
        user_id=callback.from_user.id,
        vote=parts[4],
    )
    await callback.answer(
        "Спасибо за оценку!" if created else "Вы уже оценили это предложение."
    )
    await feedback_cleanup.remove(callback.message.chat.id, callback.message.message_id)
