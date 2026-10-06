from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from aiogram.enums import ChatMemberStatus, ChatType

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers.feedback import record_feedback
from already_mentioned.handlers.management import (
    confirm_forget,
    edit_solution_command,
    forget_command,
    help_command,
    solutions_command,
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
        reply_to_message=None,
        answer=AsyncMock(),
    )


def make_callback(
    data: str, *, chat_id: int = -1001, user_id: int = 10
) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        message=SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type=ChatType.SUPERGROUP),
            message_id=100,
            date=datetime.now(UTC),
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
    assert "0.458" in status.answer.await_args.args[0]

    member_bot = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.MEMBER)
        )
    )
    command = make_message("/threshold 0,92")
    await threshold_command(command, member_bot, chats)
    assert (await chats.get_chat(-1001)).similarity_threshold == 0.458
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
    cleanup = SimpleNamespace(remove=AsyncMock())
    await feedback.schedule_keyboard(-1001, 100, float("inf"), author_id=10)
    callback = make_callback(f"vote:{solution_id}:50:10:helpful")
    await record_feedback(callback, solutions, feedback, cleanup)
    await record_feedback(callback, solutions, feedback, cleanup)
    assert len(await feedback.list_for_chat(-1001)) == 1
    assert "уже оценили" in callback.answer.await_args.args[0]

    cross_chat = make_callback(f"vote:{solution_id}:50:10:helpful", chat_id=-1002)
    await record_feedback(cross_chat, solutions, feedback, cleanup)
    assert await feedback.list_for_chat(-1002) == []


@pytest.mark.asyncio
async def test_solution_pages_are_chat_scoped_and_count_votes(
    repositories: tuple[ChatRepository, SolutionRepository, FeedbackRepository],
) -> None:
    chats, solutions, feedback = repositories
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    empty = make_message("/solutions")
    await solutions_command(empty, admin_bot(), solutions)
    assert "пока нет" in empty.answer.await_args.args[0]

    ids = []
    for number in range(6):
        solution_id = await solutions.add_solution(
            chat_id=-1001,
            question_message_id=number * 2 + 1,
            answer_message_id=number * 2 + 2,
            question_text=f"Вопрос <{number}>?",
            answer_text="Ответ & " + "д" * 200,
            question_embedding=b"vector",
            question_link=f"https://t.me/c/1/{number * 2 + 1}",
            answer_link=f"https://t.me/c/1/{number * 2 + 2}",
        )
        ids.append(solution_id)
    other_id = await add_solution(solutions, -1002)
    for vote, user in (("helpful", 20), ("helpful", 21), ("not_helpful", 22)):
        await feedback.add_feedback(
            solution_id=ids[-1], query_message_id=50, user_id=user, vote=vote
        )
    await feedback.add_feedback(
        solution_id=other_id, query_message_id=50, user_id=23, vote="helpful"
    )

    first = make_message("/solutions")
    await solutions_command(first, admin_bot(), solutions)
    output = first.answer.await_args.args[0]
    assert "страница 1/2" in output and "👍 2 / 👎 1" in output
    assert f"#{ids[-1]}" in output and f"#{ids[0]}" not in output
    assert f"#{other_id}" not in output
    assert "Вопрос &lt;5&gt;?" in output and "Ответ &amp;" in output
    assert "https://t.me/c/1/12" in output and "https://t.me/c/1/11" in output
    assert "Далее: /solutions 2" in output
    second = make_message("/solutions 2")
    await solutions_command(second, admin_bot(), solutions)
    assert f"#{ids[0]}" in second.answer.await_args.args[0]
    assert f"#{ids[1]}" not in second.answer.await_args.args[0]
    assert "Назад: /solutions 1" in second.answer.await_args.args[0]

    missing = make_message("/solutions 3")
    await solutions_command(missing, admin_bot(), solutions)
    assert "Такой страницы нет" in missing.answer.await_args.args[0]
    member = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.MEMBER)
        )
    )
    denied = make_message("/solutions")
    await solutions_command(denied, member, solutions)
    assert "только администратору" in denied.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_edit_question_and_answer_reembed_their_own_vectors(
    repositories: tuple[ChatRepository, SolutionRepository, FeedbackRepository],
) -> None:
    chats, solutions, _ = repositories
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    solution_id = await add_solution(solutions, -1001)
    other_id = await add_solution(solutions, -1002)

    class FakeEmbeddings:
        def __init__(self) -> None:
            self.passages = []

        async def embed_passage(self, text):
            self.passages.append(text)
            return np.array([3.0, 4.0], dtype=np.float32)

    embeddings = FakeEmbeddings()
    reply = SimpleNamespace(message_id=2)
    answer = make_message("/editanswer 1 Новый ответ")
    answer.text = f"/editanswer {solution_id} Новый ответ"
    answer.reply_to_message = reply
    await edit_solution_command(answer, admin_bot(), solutions, embeddings)
    saved = await solutions.get_for_chat(-1001, solution_id)
    assert saved.answer_text == "Новый ответ"
    assert saved.question_embedding == b"vector"
    assert saved.answer_embedding == np.array([0.6, 0.8], dtype=np.float32).tobytes()
    assert embeddings.passages == ["Новый ответ"]

    question = make_message(f"/editquestion {solution_id} Где вход?")
    question.reply_to_message = reply
    await edit_solution_command(question, admin_bot(), solutions, embeddings)
    saved = await solutions.get_for_chat(-1001, solution_id)
    assert saved.question_text == "Где вход?"
    assert saved.question_embedding == np.array([0.6, 0.8], dtype=np.float32).tobytes()
    assert embeddings.passages == ["Новый ответ", "Где вход?"]
    assert (await solutions.get_for_chat(-1002, other_id)).answer_text != "Новый ответ"

    wrong_chat = make_message(f"/editanswer {other_id} Чужой ответ")
    wrong_chat.reply_to_message = reply
    await edit_solution_command(wrong_chat, admin_bot(), solutions, embeddings)
    assert "не найдено" in wrong_chat.answer.await_args.args[0]
    wrong_reply = make_message(f"/editanswer {solution_id} Неверный reply")
    wrong_reply.reply_to_message = SimpleNamespace(message_id=99)
    await edit_solution_command(wrong_reply, admin_bot(), solutions, embeddings)
    assert "не найдено" in wrong_reply.answer.await_args.args[0]
    no_reply = make_message(f"/editanswer {solution_id} Нет reply")
    await edit_solution_command(no_reply, admin_bot(), solutions, embeddings)
    assert "нужно отправить reply" in no_reply.answer.await_args.args[0]

    member = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.MEMBER)
        )
    )
    denied = make_message(f"/editanswer {solution_id} Не менять")
    denied.reply_to_message = reply
    await edit_solution_command(denied, member, solutions, embeddings)
    assert "только администратору" in denied.answer.await_args.args[0]
    saved = await solutions.get_for_chat(-1001, solution_id)
    assert saved.answer_text == "Новый ответ"

    help_message = make_message("/help")
    await help_command(help_message)
    assert "/solutions" in help_message.answer.await_args.args[0]
    assert "/editquestion" in help_message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_long_question_is_shown_in_chunks(
    repositories: tuple[ChatRepository, SolutionRepository, FeedbackRepository],
) -> None:
    chats, solutions, _ = repositories
    await chats.ensure_chat(-1001, "Тест")
    await solutions.add_solution(
        chat_id=-1001,
        question_message_id=1,
        answer_message_id=2,
        question_text="<" * 3000,
        answer_text="Ответ",
        question_embedding=b"vector",
        question_link="https://t.me/c/1/1",
        answer_link="https://t.me/c/1/2",
    )
    command = make_message("/solutions")
    await solutions_command(command, admin_bot(), solutions)
    messages = [call.args[0] for call in command.answer.await_args_list]
    assert sum(item.count("&lt;") for item in messages) == 3000
    assert all(len(item) <= 4000 for item in messages)
    assert any("👍 0 / 👎 0" in item for item in messages)
