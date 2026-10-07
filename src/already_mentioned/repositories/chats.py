"""Chat settings and scoped data deletion."""

import aiosqlite

from already_mentioned.config import DEFAULT_SIMILARITY_THRESHOLD
from already_mentioned.models.entities import Chat


class ChatRepository:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

    async def ensure_chat(self, telegram_chat_id: int, title: str) -> None:
        """Create the chat or refresh its title without changing its threshold."""
        await self.connection.execute(
            """
            INSERT INTO chats (telegram_chat_id, title, similarity_threshold)
            VALUES (?, ?, ?)
            ON CONFLICT(telegram_chat_id) DO UPDATE SET title = excluded.title
            """,
            (telegram_chat_id, title, DEFAULT_SIMILARITY_THRESHOLD),
        )
        await self.connection.commit()

    async def get_chat(self, telegram_chat_id: int) -> Chat | None:
        async with self.connection.execute(
            """
            SELECT telegram_chat_id, title, similarity_threshold, hybrid_threshold
            FROM chats WHERE telegram_chat_id = ?
            """,
            (telegram_chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return Chat(*row) if row is not None else None

    async def set_threshold(
        self, telegram_chat_id: int, threshold: float, *, hybrid: bool = False
    ) -> bool:
        if not 0 <= threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        column = "hybrid_threshold" if hybrid else "similarity_threshold"
        cursor = await self.connection.execute(
            f"UPDATE chats SET {column} = ? WHERE telegram_chat_id = ?",
            (threshold, telegram_chat_id),
        )
        await self.connection.commit()
        return cursor.rowcount > 0

    async def forget_chat_data(self, telegram_chat_id: int) -> int:
        """Delete this chat's solutions and cascaded feedback; keep chat settings."""
        cursor = await self.connection.execute(
            "DELETE FROM solutions WHERE chat_id = ?", (telegram_chat_id,)
        )
        await self.connection.commit()
        return cursor.rowcount
