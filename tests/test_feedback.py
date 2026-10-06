from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageReplyMarkup

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers.feedback import record_feedback
from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.services.feedback_cleanup import FeedbackCleanup


def callback(*, user_id=10, age=0, data="vote:7:50:10:helpful"):
    return SimpleNamespace(
        data=data,
        message=SimpleNamespace(
            chat=SimpleNamespace(id=-1001),
            message_id=100,
            date=datetime.now(UTC) - timedelta(seconds=age),
        ),
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outsider_data", ["vote:7:50:10:helpful", "vote:7:50:11:helpful"]
)
async def test_only_author_can_vote_and_success_removes_buttons(outsider_data):
    solutions = SimpleNamespace(get_for_chat=AsyncMock(return_value=object()))
    feedback = SimpleNamespace(
        add_feedback=AsyncMock(return_value=True),
        keyboard_author=AsyncMock(return_value=10),
    )
    cleanup = SimpleNamespace(remove=AsyncMock())
    outsider = callback(user_id=11, data=outsider_data)
    await record_feedback(outsider, solutions, feedback, cleanup)
    assert "только автор" in outsider.answer.await_args.args[0]
    feedback.add_feedback.assert_not_awaited()
    cleanup.remove.assert_not_awaited()

    author = callback()
    await record_feedback(author, solutions, feedback, cleanup)
    feedback.add_feedback.assert_awaited_once_with(
        solution_id=7, query_message_id=50, user_id=10, vote="helpful"
    )
    cleanup.remove.assert_awaited_once_with(-1001, 100)
    assert "Спасибо" in author.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_expired_vote_is_rejected_even_before_timer_runs():
    solutions = SimpleNamespace(get_for_chat=AsyncMock())
    feedback = SimpleNamespace(
        add_feedback=AsyncMock(), keyboard_author=AsyncMock(return_value=10)
    )
    cleanup = SimpleNamespace(remove=AsyncMock())
    expired = callback(age=3601)
    await record_feedback(expired, solutions, feedback, cleanup)
    feedback.add_feedback.assert_not_awaited()
    assert "истекло" in expired.answer.await_args.args[0]
    cleanup.remove.assert_awaited_once_with(-1001, 100)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ["vote:7:50:helpful", "vote:x:50:10:helpful"])
async def test_invalid_or_legacy_buttons_do_not_allow_unrestricted_votes(data):
    feedback = SimpleNamespace(add_feedback=AsyncMock())
    await record_feedback(callback(data=data), None, feedback, None)
    feedback.add_feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_survives_restart_and_leaves_unexpired_buttons(tmp_path):
    path = tmp_path / "bot.db"
    connection = await connect_database(path)
    await initialize_database(connection)
    repository = FeedbackRepository(connection)
    bot = SimpleNamespace(edit_message_reply_markup=AsyncMock())
    cleanup = FeedbackCleanup(bot, repository)
    sent = callback().message
    await cleanup.schedule(sent, 10)
    await repository.schedule_keyboard(-1002, 200, 0)
    await connection.close()

    connection = await connect_database(path)
    try:
        await initialize_database(connection)
        repository = FeedbackRepository(connection)
        cleanup = FeedbackCleanup(bot, repository)
        assert await repository.keyboard_author(-1001, 100) == 10
        await cleanup.cleanup_due()
        bot.edit_message_reply_markup.assert_awaited_once_with(
            chat_id=-1002, message_id=200, reply_markup=None
        )
        assert await repository.due_keyboards(float("inf")) == [(-1001, 100)]
        await cleanup.remove(-1001, 100)
        assert await repository.due_keyboards(float("inf")) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_cleanup_retries_transient_failure_and_discards_deleted_message(
    tmp_path, caplog
):
    connection = await connect_database(tmp_path / "bot.db")
    try:
        await initialize_database(connection)
        repository = FeedbackRepository(connection)
        bot = SimpleNamespace(
            edit_message_reply_markup=AsyncMock(side_effect=RuntimeError("SECRET"))
        )
        cleanup = FeedbackCleanup(bot, repository)
        await cleanup.remove(-1001, 100)
        assert await repository.due_keyboards(float("inf")) == [(-1001, 100)]
        assert "SECRET" not in caplog.text
        bot.edit_message_reply_markup.side_effect = None
        await cleanup.cleanup_due()
        assert await repository.due_keyboards(float("inf")) == []

        bot.edit_message_reply_markup.side_effect = TelegramBadRequest(
            method=EditMessageReplyMarkup(chat_id=-1001, message_id=100),
            message="message to edit not found",
        )
        await cleanup.remove(-1001, 100)
        assert await repository.due_keyboards(float("inf")) == []
    finally:
        await connection.close()
