from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.types import Chat, Message

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers.solve import remember_admin_answer, solve
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.reply_cache import PendingAnswer, ReplyCache


class FakeEmbeddings:
    async def embed_passage(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


@pytest_asyncio.fixture
async def repositories(
    tmp_path: Path,
) -> AsyncIterator[tuple[ChatRepository, SolutionRepository]]:
    connection = await connect_database(tmp_path / "bot.db")
    await initialize_database(connection)
    try:
        yield ChatRepository(connection), SolutionRepository(connection)
    finally:
        await connection.close()


def make_message(
    message_id: int,
    *,
    text: str,
    user_id: int,
    reply_to_message: SimpleNamespace | None = None,
    chat_id: int = -1001234567890,
    link: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        message_id=message_id,
        chat=SimpleNamespace(id=chat_id, type=ChatType.SUPERGROUP, title="Тест"),
        from_user=SimpleNamespace(id=user_id),
        text=text,
        reply_to_message=reply_to_message,
        get_url=lambda: link,
        answer=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_admin_reply_chain_saved_once(
    repositories: tuple[ChatRepository, SolutionRepository],
) -> None:
    chats, solutions = repositories
    cache = ReplyCache()
    bot = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.ADMINISTRATOR)
        )
    )
    question = make_message(
        10, text="Как войти?", user_id=100, link="https://t.me/c/1234567890/10"
    )
    answer = make_message(
        11,
        text="Откройте страницу входа.",
        user_id=200,
        reply_to_message=question,
        link="https://t.me/c/1234567890/11",
    )
    command = make_message(12, text="/solve", user_id=200, reply_to_message=answer)

    await remember_admin_answer(answer, bot, cache)
    await solve(command, bot, cache, chats, solutions, FakeEmbeddings())
    saved = await solutions.list_for_chat(question.chat.id)
    assert len(saved) == 1
    assert saved[0].question_text == "Как войти?"
    assert saved[0].answer_text == "Откройте страницу входа."
    assert saved[0].question_link == "https://t.me/c/1234567890/10"
    assert saved[0].answer_link == "https://t.me/c/1234567890/11"
    assert (
        saved[0].question_embedding == np.array([1.0, 0.0], dtype=np.float32).tobytes()
    )

    await solve(command, bot, cache, chats, solutions, FakeEmbeddings())
    assert await solutions.count_for_chat(question.chat.id) == 1
    assert "уже сохранена" in command.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_solve_rejects_non_admin_and_broken_chain(
    repositories: tuple[ChatRepository, SolutionRepository],
) -> None:
    chats, solutions = repositories
    cache = ReplyCache()
    non_admin_bot = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.MEMBER)
        )
    )
    command = make_message(12, text="/solve", user_id=100)
    await solve(command, non_admin_bot, cache, chats, solutions, FakeEmbeddings())
    assert "только администратор" in command.answer.await_args.args[0]

    admin_bot = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.CREATOR)
        )
    )
    await solve(command, admin_bot, cache, chats, solutions, FakeEmbeddings())
    assert "Как сохранить решение" in command.answer.await_args.args[0]
    assert await solutions.count_for_chat(command.chat.id) == 0


@pytest.mark.asyncio
async def test_solve_requires_own_answer_and_a_link(
    repositories: tuple[ChatRepository, SolutionRepository],
) -> None:
    chats, solutions = repositories
    cache = ReplyCache()
    bot = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.ADMINISTRATOR)
        )
    )
    question = make_message(10, text="Где вход?", user_id=100)
    answer = make_message(
        11, text="Вот ссылка.", user_id=200, reply_to_message=question
    )
    await remember_admin_answer(answer, bot, cache)

    other_admin_command = make_message(
        12, text="/solve", user_id=300, reply_to_message=answer
    )
    await solve(other_admin_command, bot, cache, chats, solutions, FakeEmbeddings())
    assert "Как сохранить решение" in other_admin_command.answer.await_args.args[0]

    own_command = make_message(13, text="/solve", user_id=200, reply_to_message=answer)
    await solve(own_command, bot, cache, chats, solutions, FakeEmbeddings())
    assert "нужна супергруппа" in own_command.answer.await_args.args[0]
    assert await solutions.count_for_chat(question.chat.id) == 0


def test_cache_expires_and_is_scoped_to_chat() -> None:
    current_time = [0.0]
    cache = ReplyCache(ttl_seconds=10, clock=lambda: current_time[0])
    pending = PendingAnswer(
        chat_id=-1001,
        answer_message_id=2,
        answer_author_id=3,
        question_message_id=1,
        question_text="Вопрос?",
        question_link="https://t.me/c/1/1",
        recorded_at=0.0,
    )
    cache.remember(pending)
    assert cache.get(-1002, 2) is None
    assert cache.get(-1001, 2) == pending
    current_time[0] = 10.0
    assert cache.get(-1001, 2) is None


@pytest.mark.parametrize(
    ("chat_id", "username", "expected"),
    [
        (-1001234567890, "public_chat", "https://t.me/public_chat/42"),
        (-1001234567890, None, "https://t.me/c/1234567890/42"),
    ],
)
def test_aiogram_message_links(
    chat_id: int, username: str | None, expected: str
) -> None:
    chat = Chat(id=chat_id, type=ChatType.SUPERGROUP, title="Тест", username=username)
    message = Message(message_id=42, date=datetime.now(UTC), chat=chat, text="Ответ")
    assert message.get_url() == expected


def test_basic_group_has_no_direct_message_link() -> None:
    chat = Chat(id=-1234, type=ChatType.GROUP, title="Обычная группа")
    message = Message(message_id=42, date=datetime.now(UTC), chat=chat, text="Ответ")
    assert message.get_url() is None
