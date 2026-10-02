"""Confirmed question–answer pair storage scoped by Telegram chat."""

import aiosqlite

from already_mentioned.models.entities import Solution, SolutionSummary


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
        answer_embedding: bytes | None = None,
    ) -> int | None:
        """Return a new ID, or None if this exact pair is already stored."""
        cursor = await self.connection.execute(
            """
            INSERT INTO solutions (
                chat_id, question_message_id, answer_message_id,
                question_text, answer_text, question_embedding,
                question_link, answer_link, answer_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                answer_embedding,
            ),
        )
        await self.connection.commit()
        return cursor.lastrowid if cursor.rowcount else None

    async def list_for_chat(self, chat_id: int) -> list[Solution]:
        async with self.connection.execute(
            """
            SELECT id, chat_id, question_message_id, answer_message_id,
                   question_text, answer_text, question_embedding,
                   question_link, answer_link, answer_embedding
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

    async def page_for_chat(
        self, chat_id: int, *, limit: int, offset: int
    ) -> list[SolutionSummary]:
        """Return a bounded page with vote totals, newest solution first."""
        async with self.connection.execute(
            """
            SELECT s.id, s.chat_id, s.question_message_id, s.answer_message_id,
                   s.question_text, s.answer_text, s.question_embedding,
                   s.question_link, s.answer_link, s.answer_embedding,
                   COUNT(CASE WHEN f.vote = 'helpful' THEN 1 END),
                   COUNT(CASE WHEN f.vote = 'not_helpful' THEN 1 END)
            FROM solutions AS s
            LEFT JOIN feedback AS f ON f.solution_id = s.id
            WHERE s.chat_id = ?
            GROUP BY s.id
            ORDER BY s.id DESC
            LIMIT ? OFFSET ?
            """,
            (chat_id, limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            SolutionSummary(Solution(*row[:9], row[9]), row[10], row[11])
            for row in rows
        ]

    async def get_for_chat(self, chat_id: int, solution_id: int) -> Solution | None:
        async with self.connection.execute(
            """
            SELECT id, chat_id, question_message_id, answer_message_id,
                   question_text, answer_text, question_embedding,
                   question_link, answer_link, answer_embedding
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
                   question_link, answer_link, answer_embedding
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

    async def set_answer_embedding(self, solution_id: int, embedding: bytes) -> None:
        await self.connection.execute(
            "UPDATE solutions SET answer_embedding = ? "
            "WHERE id = ? AND answer_embedding IS NULL",
            (embedding, solution_id),
        )
        await self.connection.commit()

    async def update_question(
        self,
        chat_id: int,
        solution_id: int,
        answer_message_id: int,
        text: str,
        embedding: bytes,
    ) -> bool:
        cursor = await self.connection.execute(
            """UPDATE solutions SET question_text = ?, question_embedding = ?
               WHERE chat_id = ? AND id = ? AND answer_message_id = ?""",
            (text, embedding, chat_id, solution_id, answer_message_id),
        )
        await self.connection.commit()
        return cursor.rowcount > 0

    async def update_answer(
        self,
        chat_id: int,
        solution_id: int,
        answer_message_id: int,
        text: str,
        embedding: bytes,
    ) -> bool:
        cursor = await self.connection.execute(
            """UPDATE solutions SET answer_text = ?, answer_embedding = ?
               WHERE chat_id = ? AND id = ? AND answer_message_id = ?""",
            (text, embedding, chat_id, solution_id, answer_message_id),
        )
        await self.connection.commit()
        return cursor.rowcount > 0

    async def delete_by_answer(self, chat_id: int, answer_message_id: int) -> bool:
        """Remove a saved answer in this chat and its cascaded feedback."""
        cursor = await self.connection.execute(
            "DELETE FROM solutions WHERE chat_id = ? AND answer_message_id = ?",
            (chat_id, answer_message_id),
        )
        await self.connection.commit()
        return cursor.rowcount > 0
