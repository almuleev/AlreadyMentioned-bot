"""Confirmed question–answer pair storage scoped by Telegram chat."""

import aiosqlite

from already_mentioned.models.entities import Solution


class SolutionRepository:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

    async def add_solution(
        self,
        *,
        chat_id: int,
        question_message_id: int,
        answer_message_id: int,
        question_text: str,
        answer_text: str,
        question_embedding: bytes,
        question_link: str,
        answer_link: str,
    ) -> int | None:
        """Return a new ID, or None if this exact pair is already stored."""
        cursor = await self.connection.execute(
            """
            INSERT INTO solutions (
                chat_id, question_message_id, answer_message_id,
                question_text, answer_text, question_embedding,
                question_link, answer_link
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, question_message_id, answer_message_id) DO NOTHING
            """,
            (
                chat_id,
                question_message_id,
                answer_message_id,
                question_text,
                answer_text,
                question_embedding,
                question_link,
                answer_link,
            ),
        )
        await self.connection.commit()
        return cursor.lastrowid if cursor.rowcount else None

    async def list_for_chat(self, chat_id: int) -> list[Solution]:
        async with self.connection.execute(
            """
            SELECT id, chat_id, question_message_id, answer_message_id,
                   question_text, answer_text, question_embedding,
                   question_link, answer_link
            FROM solutions WHERE chat_id = ? ORDER BY id
            """,
            (chat_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [Solution(*row) for row in rows]

    async def count_for_chat(self, chat_id: int) -> int:
        async with self.connection.execute(
            "SELECT COUNT(*) FROM solutions WHERE chat_id = ?", (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0]

    async def get_for_chat(self, chat_id: int, solution_id: int) -> Solution | None:
        async with self.connection.execute(
            """
            SELECT id, chat_id, question_message_id, answer_message_id,
                   question_text, answer_text, question_embedding,
                   question_link, answer_link
            FROM solutions WHERE chat_id = ? AND id = ?
            """,
            (chat_id, solution_id),
        ) as cursor:
            row = await cursor.fetchone()
        return Solution(*row) if row is not None else None

    async def get_by_pair(
        self, chat_id: int, question_message_id: int, answer_message_id: int
    ) -> Solution | None:
        async with self.connection.execute(
            """
            SELECT id, chat_id, question_message_id, answer_message_id,
                   question_text, answer_text, question_embedding,
                   question_link, answer_link
            FROM solutions
            WHERE chat_id = ? AND question_message_id = ? AND answer_message_id = ?
            """,
            (chat_id, question_message_id, answer_message_id),
        ) as cursor:
            row = await cursor.fetchone()
        return Solution(*row) if row is not None else None

    async def set_embedding(self, solution_id: int, embedding: bytes) -> None:
        await self.connection.execute(
            "UPDATE solutions SET question_embedding = ? WHERE id = ?",
            (embedding, solution_id),
        )
        await self.connection.commit()
