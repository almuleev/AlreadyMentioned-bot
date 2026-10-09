"""Persistent, scoped progress of explicitly confirmed FAQ publication."""

import asyncio
import json
import secrets
import sqlite3
from dataclasses import dataclass

from already_mentioned.services.embeddings import FRIDA_CONTRACT


@dataclass(frozen=True, slots=True)
class FaqJob:
    id: str
    chat_id: int
    user_id: int
    thread_id: int | None
    status: str
    contract: str


@dataclass(frozen=True, slots=True)
class FaqItem:
    job_id: str
    row_number: int
    question: str
    answer: str
    status: str
    question_message_id: int | None
    answer_message_id: int | None
    question_vector: bytes | None
    answer_vector: bytes | None
    error_type: str | None
    retry_at: float | None


class FaqRepository:
    def __init__(self, connection):
        self.connection = connection
        self.creation_lock = asyncio.Lock()

    async def create(self, chat_id, user_id, thread_id, pairs):
        async with self.creation_lock:
            return await self._create(chat_id, user_id, thread_id, pairs)

    async def _create(self, chat_id, user_id, thread_id, pairs):
        job_id = secrets.token_urlsafe(8)
        try:
            await self.connection.execute(
                "INSERT INTO faq_jobs(id, chat_id, user_id, thread_id, "
                "status, contract) "
                "VALUES (?, ?, ?, ?, 'paused', ?)",
                (
                    job_id,
                    chat_id,
                    user_id,
                    thread_id,
                    json.dumps(FRIDA_CONTRACT, sort_keys=True),
                ),
            )
            await self.connection.executemany(
                "INSERT INTO faq_items(job_id, row_number, question, answer) "
                "VALUES (?, ?, ?, ?)",
                [(job_id, p.row, p.question, p.answer) for p in pairs],
            )
            await self.connection.commit()
        except sqlite3.IntegrityError:
            await self.connection.rollback()
            return None
        except BaseException:
            await self.connection.rollback()
            raise
        return job_id

    async def get(self, job_id, chat_id, user_id):
        async with self.connection.execute(
            "SELECT id, chat_id, user_id, thread_id, status, contract FROM faq_jobs "
            "WHERE id = ? AND chat_id = ? AND user_id = ?",
            (job_id, chat_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()
        return FaqJob(*row) if row else None

    async def active_for_chat(self, chat_id):
        async with self.connection.execute(
            "SELECT id, chat_id, user_id, thread_id, status, contract FROM faq_jobs "
            "WHERE chat_id = ? AND status IN ('running', 'paused')",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return FaqJob(*row) if row else None

    async def items(self, job_id):
        async with self.connection.execute(
            "SELECT job_id, row_number, question, answer, status, question_message_id, "
            "answer_message_id, question_vector, answer_vector, error_type, retry_at "
            "FROM faq_items WHERE job_id = ? ORDER BY row_number",
            (job_id,),
        ) as cursor:
            return [FaqItem(*row) for row in await cursor.fetchall()]

    async def state(self, job_id, status):
        cursor = await self.connection.execute(
            "UPDATE faq_jobs SET status = ? WHERE id = ? "
            "AND status IN ('running', 'paused')",
            (status, job_id),
        )
        await self.connection.commit()
        return cursor.rowcount > 0

    async def update(self, job_id, row_number, **fields):
        if not fields or not fields.keys() <= {
            "status",
            "question_message_id",
            "answer_message_id",
            "question_vector",
            "answer_vector",
            "error_type",
            "retry_at",
        }:
            raise ValueError("Invalid FAQ update")
        cursor = await self.connection.execute(
            "UPDATE faq_items SET "
            + ",".join(f"{key} = ?" for key in fields)
            + " WHERE job_id = ? AND row_number = ?",
            (*fields.values(), job_id, row_number),
        )
        await self.connection.commit()
        return cursor.rowcount > 0

    async def finish_item(self, job, item):
        """Insert the solution and progress marker together, only for a running job."""
        link = f"https://t.me/c/{str(job.chat_id).removeprefix('-100')}/"
        cursor = await self.connection.execute(
            "INSERT INTO solutions(chat_id, question_message_id, answer_message_id, "
            "question_text, answer_text, question_embedding, answer_embedding, "
            "question_link, answer_link) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ? "
            "WHERE EXISTS(SELECT 1 FROM faq_jobs WHERE id = ? AND status = 'running') "
            "ON CONFLICT(chat_id, question_message_id, answer_message_id) DO NOTHING",
            (
                job.chat_id,
                item.question_message_id,
                item.answer_message_id,
                item.question,
                item.answer,
                item.question_vector,
                item.answer_vector,
                link + str(item.question_message_id),
                link + str(item.answer_message_id),
                job.id,
            ),
        )
        if cursor.rowcount:
            await self.connection.execute(
                "UPDATE faq_items SET status = 'done', question_vector = NULL, "
                "answer_vector = NULL, error_type = NULL "
                "WHERE job_id = ? AND row_number = ?",
                (job.id, item.row_number),
            )
        await self.connection.commit()
        return cursor.rowcount > 0
