"""One vote per user and suggested solution for each query message."""

from typing import Literal

import aiosqlite

from already_mentioned.models.entities import Feedback

Vote = Literal["helpful", "not_helpful"]


class FeedbackRepository:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

    async def schedule_keyboard(
        self,
        chat_id: int,
        message_id: int,
        expires_at: float,
        author_id: int | None = None,
    ) -> None:
        await self.connection.execute(
            "INSERT INTO feedback_keyboards "
            "(chat_id, message_id, expires_at, author_id) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(chat_id, message_id) "
            "DO UPDATE SET expires_at = MIN(expires_at, excluded.expires_at), "
            "author_id = COALESCE(feedback_keyboards.author_id, excluded.author_id)",
            (chat_id, message_id, expires_at, author_id),
        )
        await self.connection.commit()

    async def keyboard_author(self, chat_id: int, message_id: int) -> int | None:
        async with self.connection.execute(
            "SELECT author_id FROM feedback_keyboards WHERE chat_id = ? "
            "AND message_id = ?",
            (chat_id, message_id),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    async def due_keyboards(self, now: float) -> list[tuple[int, int]]:
        async with self.connection.execute(
            "SELECT chat_id, message_id FROM feedback_keyboards "
            "WHERE expires_at <= ? ORDER BY expires_at LIMIT 100",
            (now,),
        ) as cursor:
            return [(row[0], row[1]) for row in await cursor.fetchall()]

    async def delete_keyboard(self, chat_id: int, message_id: int) -> None:
        await self.connection.execute(
            "DELETE FROM feedback_keyboards WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        )
        await self.connection.commit()

    async def add_feedback(
        self,
        *,
        solution_id: int,
        query_message_id: int,
        user_id: int,
        vote: Vote,
    ) -> bool:
        """Return False when this user already voted on this suggestion."""
        cursor = await self.connection.execute(
            """
            INSERT INTO feedback (solution_id, query_message_id, user_id, vote)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(solution_id, query_message_id, user_id) DO NOTHING
            """,
            (solution_id, query_message_id, user_id, vote),
        )
        await self.connection.commit()
        return cursor.rowcount > 0

    async def list_for_chat(self, chat_id: int) -> list[Feedback]:
        async with self.connection.execute(
            """
            SELECT feedback.id, solution_id, query_message_id, user_id, vote
            FROM feedback
            JOIN solutions ON solutions.id = feedback.solution_id
            WHERE solutions.chat_id = ?
            ORDER BY feedback.id
            """,
            (chat_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [Feedback(*row) for row in rows]
