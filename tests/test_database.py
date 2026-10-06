from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from already_mentioned.config import DEFAULT_SIMILARITY_THRESHOLD
from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.repositories.solutions import SolutionRepository


@pytest_asyncio.fixture
async def database(tmp_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    connection = await connect_database(tmp_path / "nested" / "bot.db")
    await initialize_database(connection)
    try:
        yield connection
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_chat_settings_and_parent_directory(
    database: aiosqlite.Connection, tmp_path: Path
) -> None:
    assert (tmp_path / "nested" / "bot.db").is_file()
    chats = ChatRepository(database)
    await chats.ensure_chat(-1001, "Первый чат")
    assert (await chats.get_chat(-1001)).similarity_threshold == pytest.approx(
        DEFAULT_SIMILARITY_THRESHOLD
    )

    assert await chats.set_threshold(-1001, 0.91)
    await chats.ensure_chat(-1001, "Новое название")
    updated = await chats.get_chat(-1001)
    assert updated.title == "Новое название"
    assert updated.similarity_threshold == pytest.approx(0.91)
    assert not await chats.set_threshold(-9999, 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        await chats.set_threshold(-1001, 1.1)


@pytest.mark.asyncio
async def test_chat_isolation_duplicates_and_forget(
    database: aiosqlite.Connection,
) -> None:
    chats = ChatRepository(database)
    solutions = SolutionRepository(database)
    feedback = FeedbackRepository(database)
    await chats.ensure_chat(-1001, "Чат А")
    await chats.ensure_chat(-1002, "Чат Б")

    first_id = await solutions.add_solution(
        chat_id=-1001,
        question_message_id=10,
        answer_message_id=11,
        question_text="Как войти?",
        answer_text="Откройте страницу входа.",
        question_embedding=b"first-vector",
        question_link="https://t.me/c/1/10",
        answer_link="https://t.me/c/1/11",
    )
    second_id = await solutions.add_solution(
        chat_id=-1002,
        question_message_id=10,
        answer_message_id=11,
        question_text="Как войти?",
        answer_text="Используйте другой адрес.",
        question_embedding=b"second-vector",
        question_link="https://t.me/c/2/10",
        answer_link="https://t.me/c/2/11",
    )
    assert first_id is not None and second_id is not None
    assert first_id != second_id
    assert (
        await solutions.add_solution(
            chat_id=-1001,
            question_message_id=10,
            answer_message_id=11,
            question_text="Как войти?",
            answer_text="Откройте страницу входа.",
            question_embedding=b"first-vector",
            question_link="https://t.me/c/1/10",
            answer_link="https://t.me/c/1/11",
        )
        is None
    )

    assert await feedback.add_feedback(
        solution_id=first_id, query_message_id=20, user_id=123, vote="helpful"
    )
    assert not await feedback.add_feedback(
        solution_id=first_id, query_message_id=20, user_id=123, vote="not_helpful"
    )
    assert await feedback.add_feedback(
        solution_id=second_id, query_message_id=20, user_id=123, vote="not_helpful"
    )
    assert [item.id for item in await solutions.list_for_chat(-1001)] == [first_id]
    assert [item.id for item in await solutions.list_for_chat(-1002)] == [second_id]
    assert [item.vote for item in await feedback.list_for_chat(-1001)] == ["helpful"]
    assert [item.vote for item in await feedback.list_for_chat(-1002)] == [
        "not_helpful"
    ]

    assert await chats.forget_chat_data(-1001) == 1
    assert await solutions.count_for_chat(-1001) == 0
    assert await feedback.list_for_chat(-1001) == []
    assert await solutions.count_for_chat(-1002) == 1
    assert [item.vote for item in await feedback.list_for_chat(-1002)] == [
        "not_helpful"
    ]
    assert await chats.get_chat(-1001) is not None


@pytest.mark.asyncio
async def test_solution_requires_existing_chat(database: aiosqlite.Connection) -> None:
    solutions = SolutionRepository(database)
    with pytest.raises(aiosqlite.IntegrityError):
        await solutions.add_solution(
            chat_id=-404,
            question_message_id=1,
            answer_message_id=2,
            question_text="Вопрос",
            answer_text="Ответ",
            question_embedding=b"vector",
            question_link="https://t.me/c/4/1",
            answer_link="https://t.me/c/4/2",
        )
