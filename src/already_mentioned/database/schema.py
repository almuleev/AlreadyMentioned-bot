"""Versioned SQLite schema; version 1 is the original three-table layout."""

import json

import aiosqlite

from already_mentioned.services.embeddings import E5_CONTRACT

SCHEMA_VERSION = 5

# Add consecutive target versions here when an existing table must change.
# Each tuple runs inside the same transaction as its user_version update.
MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: ("ALTER TABLE solutions ADD COLUMN answer_embedding BLOB",),
    3: (
        "CREATE TABLE embedding_state (id INTEGER PRIMARY KEY CHECK (id = 1), "
        "contract TEXT NOT NULL)",
        "INSERT INTO embedding_state (id, contract) VALUES (1, '"
        + json.dumps(E5_CONTRACT, sort_keys=True)
        + "')",
    ),
    4: (
        "CREATE TABLE feedback_keyboards (chat_id INTEGER NOT NULL, "
        "message_id INTEGER NOT NULL, expires_at REAL NOT NULL, author_id INTEGER, "
        "PRIMARY KEY (chat_id, message_id))",
        "CREATE INDEX idx_feedback_keyboards_expiry ON feedback_keyboards(expires_at)",
    ),
    5: (
        "ALTER TABLE chats ADD COLUMN hybrid_threshold REAL "
        "CHECK (hybrid_threshold >= 0 AND hybrid_threshold <= 1)",
    ),
}

BASE_SCHEMA = (
    """CREATE TABLE chats (
        telegram_chat_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        similarity_threshold REAL NOT NULL DEFAULT 0.88
            CHECK (similarity_threshold >= 0 AND similarity_threshold <= 1),
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE solutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL
            REFERENCES chats(telegram_chat_id) ON DELETE CASCADE,
        question_message_id INTEGER NOT NULL,
        answer_message_id INTEGER NOT NULL,
        question_text TEXT NOT NULL,
        answer_text TEXT NOT NULL,
        question_embedding BLOB NOT NULL,
        question_link TEXT NOT NULL,
        answer_link TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (chat_id, question_message_id, answer_message_id)
    )""",
    "CREATE INDEX idx_solutions_chat_id ON solutions(chat_id)",
    """CREATE TABLE feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        solution_id INTEGER NOT NULL
            REFERENCES solutions(id) ON DELETE CASCADE,
        query_message_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        vote TEXT NOT NULL CHECK (vote IN ('helpful', 'not_helpful')),
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (solution_id, query_message_id, user_id)
    )""",
    "CREATE INDEX idx_feedback_solution_id ON feedback(solution_id)",
)

REQUIRED_COLUMNS = {
    "chats": {"telegram_chat_id", "title", "similarity_threshold", "created_at"},
    "solutions": {
        "id",
        "chat_id",
        "question_message_id",
        "answer_message_id",
        "question_text",
        "answer_text",
        "question_embedding",
        "question_link",
        "answer_link",
        "created_at",
    },
    "feedback": {
        "id",
        "solution_id",
        "query_message_id",
        "user_id",
        "vote",
        "created_at",
    },
}


async def _check_baseline(connection: aiosqlite.Connection) -> None:
    for table, required in REQUIRED_COLUMNS.items():
        async with connection.execute(f"PRAGMA table_info({table})") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if not required <= columns:
            raise RuntimeError(f"SQLite schema is missing required columns in {table}")


async def initialize_database(connection: aiosqlite.Connection) -> None:
    """Create or adopt the baseline, then apply each explicit migration once."""
    await connection.execute("BEGIN IMMEDIATE")
    try:
        async with connection.execute("PRAGMA user_version") as cursor:
            version = (await cursor.fetchone())[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError("SQLite schema is newer than this bot version")
        async with connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        ) as cursor:
            tables = {row[0] for row in await cursor.fetchall()}
        if version == 0:
            if not tables:
                for statement in BASE_SCHEMA:
                    await connection.execute(statement)
            elif not REQUIRED_COLUMNS.keys() <= tables:
                raise RuntimeError("Unversioned SQLite schema is incomplete or unknown")
            await _check_baseline(connection)
            version = 1
            await connection.execute("PRAGMA user_version = 1")
        else:
            await _check_baseline(connection)
            if version >= 2:
                async with connection.execute("PRAGMA table_info(solutions)") as cursor:
                    columns = {row[1] for row in await cursor.fetchall()}
                if "answer_embedding" not in columns:
                    raise RuntimeError("SQLite schema is missing answer_embedding")
            if version >= 3:
                async with connection.execute(
                    "SELECT contract FROM embedding_state WHERE id = 1"
                ) as cursor:
                    if await cursor.fetchone() is None:
                        raise RuntimeError(
                            "SQLite schema is missing embedding contract"
                        )
            if version >= 4:
                async with connection.execute(
                    "SELECT chat_id, message_id, expires_at, author_id "
                    "FROM feedback_keyboards LIMIT 0"
                ):
                    pass
            if version >= 5:
                async with connection.execute(
                    "SELECT hybrid_threshold FROM chats LIMIT 0"
                ):
                    pass
        for target in range(version + 1, SCHEMA_VERSION + 1):
            statements = MIGRATIONS.get(target)
            if not statements:
                raise RuntimeError(f"Missing SQLite migration to version {target}")
            for statement in statements:
                await connection.execute(statement)
            await connection.execute(f"PRAGMA user_version = {target}")
        await connection.commit()
    except BaseException:
        await connection.rollback()
        raise
