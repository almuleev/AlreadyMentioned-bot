"""Version adoption and future migration mechanics use only temporary SQLite files."""

import sqlite3

import aiosqlite
import pytest

from already_mentioned.database import schema
from already_mentioned.database.connection import connect_database
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.repositories.solutions import SolutionRepository

LEGACY_SCHEMA = """
CREATE TABLE chats (
    telegram_chat_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    similarity_threshold REAL NOT NULL DEFAULT 0.88
        CHECK (similarity_threshold >= 0 AND similarity_threshold <= 1),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(telegram_chat_id) ON DELETE CASCADE,
    question_message_id INTEGER NOT NULL,
    answer_message_id INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    question_embedding BLOB NOT NULL,
    question_link TEXT NOT NULL,
    answer_link TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chat_id, question_message_id, answer_message_id)
);
CREATE INDEX idx_solutions_chat_id ON solutions(chat_id);
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    solution_id INTEGER NOT NULL REFERENCES solutions(id) ON DELETE CASCADE,
    query_message_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    vote TEXT NOT NULL CHECK (vote IN ('helpful', 'not_helpful')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (solution_id, query_message_id, user_id)
);
CREATE INDEX idx_feedback_solution_id ON feedback(solution_id);
"""


@pytest.mark.asyncio
async def test_version_one_migrates_answer_vectors_without_losing_data(
    tmp_path,
) -> None:
    path = tmp_path / "version-one.db"
    with sqlite3.connect(path) as old:
        old.executescript(LEGACY_SCHEMA)
        old.execute("PRAGMA user_version = 1")
        old.execute(
            "INSERT INTO chats (telegram_chat_id, title) VALUES (-1001, 'Тест')"
        )
        old.execute(
            """INSERT INTO solutions (
               chat_id, question_message_id, answer_message_id,
               question_text, answer_text, question_embedding,
               question_link, answer_link)
               VALUES (-1001, 1, 2, 'Вопрос', 'Ответ', ?, 'q', 'a')""",
            (b"existing-vector",),
        )
    connection = await connect_database(path)
    try:
        await schema.initialize_database(connection)
        await schema.initialize_database(connection)
        solution = (await SolutionRepository(connection).list_for_chat(-1001))[0]
        assert solution.question_embedding == b"existing-vector"
        assert solution.answer_text == "Ответ"
        assert solution.answer_embedding is None
        async with connection.execute("PRAGMA user_version") as cursor:
            assert (await cursor.fetchone())[0] == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_new_database_is_versioned_and_repeated_start_is_safe(tmp_path) -> None:
    path = tmp_path / "new.db"
    connection = await connect_database(path)
    try:
        await schema.initialize_database(connection)
        await schema.initialize_database(connection)
        async with connection.execute("PRAGMA user_version") as cursor:
            assert (await cursor.fetchone())[0] == 3
        await ChatRepository(connection).ensure_chat(-1001, "Тест")
        assert (await ChatRepository(connection).get_chat(-1001)).title == "Тест"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_adopts_copy_of_old_schema_without_changing_chat_data(tmp_path) -> None:
    path = tmp_path / "old-copy.db"
    with sqlite3.connect(path) as old:
        old.executescript(LEGACY_SCHEMA)
        old.executemany(
            "INSERT INTO chats (telegram_chat_id, title, similarity_threshold) "
            "VALUES (?, ?, ?)",
            [(-1001, "Первый", 0.73), (-1002, "Второй", 0.94)],
        )
        old.executemany(
            """INSERT INTO solutions (
                chat_id, question_message_id, answer_message_id,
                question_text, answer_text, question_embedding,
                question_link, answer_link
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (-1001, 1, 2, "Вопрос 1", "Ответ 1", b"one", "q1", "a1"),
                (-1002, 3, 4, "Вопрос 2", "Ответ 2", b"two", "q2", "a2"),
            ],
        )
        old.execute(
            "INSERT INTO feedback (solution_id, query_message_id, user_id, vote) "
            "VALUES (1, 5, 7, 'helpful')"
        )
    connection = await connect_database(path)
    try:
        await schema.initialize_database(connection)
        await schema.initialize_database(connection)
        chats = ChatRepository(connection)
        solutions = SolutionRepository(connection)
        feedback = FeedbackRepository(connection)
        assert (await chats.get_chat(-1001)).similarity_threshold == 0.73
        assert (await chats.get_chat(-1002)).similarity_threshold == 0.94
        assert [
            item.question_embedding for item in await solutions.list_for_chat(-1001)
        ] == [b"one"]
        assert [
            item.question_embedding for item in await solutions.list_for_chat(-1002)
        ] == [b"two"]
        assert (await solutions.list_for_chat(-1001))[0].answer_embedding is None
        assert [item.vote for item in await feedback.list_for_chat(-1001)] == [
            "helpful"
        ]
        assert await feedback.list_for_chat(-1002) == []
        async with connection.execute("PRAGMA user_version") as cursor:
            assert (await cursor.fetchone())[0] == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rejects_incomplete_or_newer_schema_without_partial_changes(
    tmp_path,
) -> None:
    partial = await connect_database(tmp_path / "partial.db")
    try:
        await partial.execute("CREATE TABLE chats (telegram_chat_id INTEGER)")
        await partial.commit()
        with pytest.raises(RuntimeError, match="incomplete"):
            await schema.initialize_database(partial)
        async with partial.execute("PRAGMA user_version") as cursor:
            assert (await cursor.fetchone())[0] == 0
    finally:
        await partial.close()

    newer = await connect_database(tmp_path / "newer.db")
    try:
        await newer.execute("PRAGMA user_version = 99")
        with pytest.raises(RuntimeError, match="newer"):
            await schema.initialize_database(newer)
        async with newer.execute("PRAGMA user_version") as cursor:
            assert (await cursor.fetchone())[0] == 99
    finally:
        await newer.close()


@pytest.mark.asyncio
async def test_migration_statements_and_version_roll_back_together(
    tmp_path, monkeypatch
) -> None:
    connection = await connect_database(tmp_path / "migration.db")
    try:
        await schema.initialize_database(connection)
        monkeypatch.setattr(schema, "SCHEMA_VERSION", 4)
        monkeypatch.setattr(
            schema,
            "MIGRATIONS",
            {4: ("CREATE TABLE temporary_example (id INTEGER)", "INVALID SQL")},
        )
        with pytest.raises(aiosqlite.OperationalError):
            await schema.initialize_database(connection)
        async with connection.execute("PRAGMA user_version") as cursor:
            assert (await cursor.fetchone())[0] == 3
        async with connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'temporary_example'"
        ) as cursor:
            assert await cursor.fetchone() is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_future_migration_applies_once_in_a_transaction(
    tmp_path, monkeypatch
) -> None:
    connection = await connect_database(tmp_path / "future.db")
    try:
        await schema.initialize_database(connection)
        monkeypatch.setattr(schema, "SCHEMA_VERSION", 4)
        monkeypatch.setattr(
            schema,
            "MIGRATIONS",
            {4: ("CREATE INDEX test_chat_title ON chats(title)",)},
        )
        await schema.initialize_database(connection)
        await schema.initialize_database(connection)
        async with connection.execute("PRAGMA user_version") as cursor:
            assert (await cursor.fetchone())[0] == 4
        async with connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'test_chat_title'"
        ) as cursor:
            assert (await cursor.fetchone())[0] == "test_chat_title"
    finally:
        await connection.close()
