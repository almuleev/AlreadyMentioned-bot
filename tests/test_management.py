from collections.abc import AsyncIterator
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.enums import ChatMemberStatus, ChatType

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers.feedback import record_feedback
from already_mentioned.handlers.management import (
    confirm_forget,
    forget_command,
    status_command,
    threshold_command,
)
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.confirmations import ForgetConfirmations
from already_mentioned.services.reply_cache import PendingAnswer, ReplyCache


@pytest_asyncio.fixture
async def repositories(
    tmp_path: Path,
) -> AsyncIterator[tuple[ChatRepository, SolutionRepository, FeedbackRepository]]:
    connection = await connect_database(tmp_path / "bot.db")
    await initialize_database(connection)
    try:
        yield (
            ChatRepository(connection),
            SolutionRepository(connection),
            FeedbackRepository(connection),
        )
    finally:
        await connection.close()


def make_message(
    text: str, *, chat_id: int = -1001, user_id: int = 10
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=chat_id, title="Тест", type=ChatType.SUPERGROUP),
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
    )


def make_callback(
    data: str, *, chat_id: int = -1001, user_id: int = 10
) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        message=SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type=ChatType.SUPERGROUP)
        ),
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
    )


def admin_bot() -> SimpleNamespace:
    return SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.ADMINISTRATOR)
        )
    )


async def add_solution(solutions: SolutionRepository, chat_id: int) -> int:
    result = await solutions.add_solution(
        chat_id=chat_id,
        question_message_id=1,
        answer_message_id=2,
        question_text="Как войти?",
        answer_text="Откройте страницу входа.",
        question_embedding=b"vector",
        question_link="https://t.me/c/1/1",
        answer_link="https://t.me/c/1/2",
    )
    assert result is not None
    return result


@pytest.mark.asyncio
async def test_status_and_admin_threshold(
    repositories: tuple[ChatRepository, SolutionRepository, FeedbackRepository],
) -> None:
    chats, solutions, _ = repositories
    await chats.ensure_chat(-1001, "Тест")
    await add_solution(solutions, -1001)

    status = make_message("/status")
    await status_command(status, chats, solutions)
    assert "решений: 1" in status.answer.await_args.args[0]
    assert "0.88" in status.answer.await_args.args[0]

    member_bot = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.MEMBER)
        )
    )
    command = make_message("/threshold 0,92")
    await threshold_command(command, member_bot, chats)
    assert (await chats.get_chat(-1001)).similarity_threshold == 0.88
    await threshold_command(command, admin_bot(), chats)
    assert (await chats.get_chat(-1001)).similarity_threshold == 0.92


@pytest.mark.asyncio
async def test_forget_needs_same_admin_confirmation_and_is_chat_scoped(
    repositories: tuple[ChatRepository, SolutionRepository, FeedbackRepository],
) -> None:
    chats, solutions, feedback = repositories
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    first_id = await add_solution(solutions, -1001)
    second_id = await add_solution(solutions, -1002)
    await feedback.add_feedback(
        solution_id=first_id, query_message_id=3, user_id=20, vote="helpful"
    )
    await feedback.add_feedback(
        solution_id=second_id, query_message_id=3, user_id=20, vote="not_helpful"
    )
    cache = ReplyCache()
    cache.remember(
        PendingAnswer(-1001, 2, 10, 1, "Вопрос?", "https://t.me/c/1/1", monotonic())
    )
    assert cache.get(-1001, 2) is not None
    confirmations = ForgetConfirmations()
    request = make_message("/forget")
    bot = admin_bot()
    await forget_command(request, bot, confirmations)
    keyboard = request.answer.await_args.kwargs["reply_markup"]
    confirm_data = keyboard.inline_keyboard[0][0].callback_data
    cancel_data = keyboard.inline_keyboard[0][1].callback_data
    assert await solutions.count_for_chat(-1001) == 1

    wrong_user = make_callback(confirm_data, user_id=11)
    await confirm_forget(wrong_user, bot, confirmations, chats, cache)
    assert await solutions.count_for_chat(-1001) == 1

    cancel = make_callback(cancel_data)
    await confirm_forget(cancel, bot, confirmations, chats, cache)
    assert await solutions.count_for_chat(-1001) == 1

    await forget_command(request, bot, confirmations)
    confirm_data = (
        request.answer.await_args.kwargs["reply_markup"]
        .inline_keyboard[0][0]
        .callback_data
    )
    confirm = make_callback(confirm_data)
    await confirm_forget(confirm, bot, confirmations, chats, cache)
    assert await solutions.count_for_chat(-1001) == 0
    assert await feedback.list_for_chat(-1001) == []
    assert cache.get(-1001, 2) is None
    assert await solutions.count_for_chat(-1002) == 1
    assert len(await feedback.list_for_chat(-1002)) == 1
    await confirm_forget(confirm, bot, confirmations, chats, cache)
    assert "истекло" in confirm.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_feedback_once_per_user_and_no_cross_chat_vote(
    repositories: tuple[ChatRepository, SolutionRepository, FeedbackRepository],
) -> None:
    chats, solutions, feedback = repositories
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    solution_id = await add_solution(solutions, -1001)
    callback = make_callback(f"vote:{solution_id}:50:helpful")
    await record_feedback(callback, solutions, feedback)
    await record_feedback(callback, solutions, feedback)
    assert len(await feedback.list_for_chat(-1001)) == 1
    assert "уже оценили" in callback.answer.await_args.args[0]

    cross_chat = make_callback(f"vote:{solution_id}:50:helpful", chat_id=-1002)
    await record_feedback(cross_chat, solutions, feedback)
    assert await feedback.list_for_chat(-1002) == []
