"""One vote per user and suggested solution for each query message."""

from typing import Literal

import aiosqlite

from already_mentioned.models.entities import Feedback

Vote = Literal["helpful", "not_helpful"]


class FeedbackRepository:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

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
