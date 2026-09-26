"""Initial SQLite schema for confirmed solutions and their feedback."""

import aiosqlite


async def initialize_database(connection: aiosqlite.Connection) -> None:
    """Create the MVP tables without importing or retaining chat history."""
    await connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS chats (
            telegram_chat_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            similarity_threshold REAL NOT NULL DEFAULT 0.88
                CHECK (similarity_threshold >= 0 AND similarity_threshold <= 1),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS solutions (
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
        );

        CREATE INDEX IF NOT EXISTS idx_solutions_chat_id
            ON solutions(chat_id);

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            solution_id INTEGER NOT NULL
                REFERENCES solutions(id) ON DELETE CASCADE,
            query_message_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            vote TEXT NOT NULL CHECK (vote IN ('helpful', 'not_helpful')),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (solution_id, query_message_id, user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_solution_id
            ON feedback(solution_id);
        """
    )
