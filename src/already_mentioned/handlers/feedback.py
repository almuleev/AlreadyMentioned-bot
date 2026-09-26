"""Store one feedback vote per user and shown suggestion."""

from aiogram import F, Router
from aiogram.types import CallbackQuery

from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.repositories.solutions import SolutionRepository

router = Router()


@router.callback_query(F.data.startswith("vote:"))
async def record_feedback(
    callback: CallbackQuery,
    solutions: SolutionRepository,
    feedback: FeedbackRepository,
) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 4 or parts[3] not in {"helpful", "not_helpful"}:
        await callback.answer("Некорректная оценка.")
        return
    try:
        solution_id, query_message_id = int(parts[1]), int(parts[2])
    except ValueError:
        await callback.answer("Некорректная оценка.")
        return
    if callback.message is None:
        await callback.answer("Сообщение больше недоступно.")
        return
    solution = await solutions.get_for_chat(callback.message.chat.id, solution_id)
    if solution is None:
        await callback.answer("Решение больше недоступно.")
        return
    created = await feedback.add_feedback(
        solution_id=solution_id,
        query_message_id=query_message_id,
        user_id=callback.from_user.id,
        vote=parts[3],
    )
    await callback.answer(
        "Спасибо за оценку!" if created else "Вы уже оценили это предложение."
    )
