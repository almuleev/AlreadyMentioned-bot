"""Serial, resumable FAQ publication after an explicit administrator confirmation."""

import asyncio
import json
from time import time

from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.types import LinkPreviewOptions, ReplyParameters

from already_mentioned.bot.permissions import is_chat_admin
from already_mentioned.database.embedding_contract import read_contract, validate_blob
from already_mentioned.error_logging import log_operation_error
from already_mentioned.services.embeddings import FRIDA_CONTRACT, normalize_vector
from already_mentioned.services.faq_data import pair_key


def publication_text(item, field):
    return (
        f"FAQ · {'Вопрос' if field == 'question' else 'Ответ'}\n{getattr(item, field)}"
    )


class FaqPublisher:
    def __init__(self, bot, repository, solutions, embeddings, *, interval=4):
        self.bot, self.repository = bot, repository
        self.solutions, self.embeddings = solutions, embeddings
        self.interval = interval
        self.active = set()

    async def running(self, job):
        latest = await self.repository.get(job.id, job.chat_id, job.user_id)
        return latest is not None and latest.status == "running"

    async def run(self, job_id, chat_id, user_id, progress):
        if chat_id in self.active:
            return
        self.active.add(chat_id)

        async def notify():
            try:
                await progress()
            except Exception as error:
                log_operation_error("faq_progress", chat_id, 0, error)

        item = None
        try:
            job = await self.repository.get(job_id, chat_id, user_id)
            if job is None or job.status not in {"paused", "running"}:
                return
            if (
                json.loads(job.contract) != FRIDA_CONTRACT
                or await read_contract(self.repository.connection) != FRIDA_CONTRACT
            ):
                raise ValueError("FAQ embedding contract mismatch")
            if not await is_chat_admin(self.bot, chat_id, user_id):
                return
            await self.repository.state(job.id, "running")
            await notify()
            for item in await self.repository.items(job.id):
                if not await self.running(job):
                    break
                if item.status in {"done", "skipped"}:
                    continue
                if item.retry_at is not None and item.retry_at > time():
                    await self.repository.state(job.id, "paused")
                    break
                if item.status.startswith("sending_"):
                    await self.repository.state(job.id, "paused")
                    break  # Unknown send outcome: never automatically send it again.
                known = {
                    pair_key(s.question_text, s.answer_text)
                    for s in await self.solutions.list_for_chat(chat_id)
                }
                if pair_key(item.question, item.answer) in known:
                    await self.repository.update(
                        job.id,
                        item.row_number,
                        status="skipped",
                        question_vector=None,
                        answer_vector=None,
                    )
                    continue
                if item.question_vector is None or item.answer_vector is None:
                    vectors = [
                        normalize_vector(
                            await self.embeddings.embed_passage(text)
                        ).tobytes()
                        for text in (item.question, item.answer)
                    ]
                    for vector in vectors:
                        validate_blob(vector)
                    if not await self.running(job):
                        break
                    await self.repository.update(
                        job.id,
                        item.row_number,
                        question_vector=vectors[0],
                        answer_vector=vectors[1],
                    )
                for field in ("question", "answer"):
                    item = next(
                        i
                        for i in await self.repository.items(job.id)
                        if i.row_number == item.row_number
                    )
                    validate_blob(item.question_vector)
                    validate_blob(item.answer_vector)
                    if getattr(item, field + "_message_id") is not None:
                        continue
                    await asyncio.sleep(self.interval)
                    if not await self.running(job):
                        break
                    if not await is_chat_admin(self.bot, chat_id, user_id):
                        await self.repository.state(job.id, "paused")
                        break
                    previous = "pending" if field == "question" else "question"
                    await self.repository.update(
                        job.id,
                        item.row_number,
                        status="sending_" + field,
                        error_type=None,
                    )
                    try:
                        sent = await self.bot.send_message(
                            chat_id,
                            publication_text(item, field),
                            message_thread_id=job.thread_id,
                            parse_mode=None,
                            link_preview_options=LinkPreviewOptions(is_disabled=True),
                            disable_notification=True,
                            reply_parameters=ReplyParameters(
                                message_id=item.question_message_id
                            )
                            if field == "answer"
                            else None,
                        )
                    except (
                        TelegramBadRequest,
                        TelegramForbiddenError,
                        TelegramRetryAfter,
                    ) as error:
                        await self.repository.update(
                            job.id,
                            item.row_number,
                            status=previous,
                            error_type=type(error).__name__,
                        )
                        raise
                    await self.repository.update(
                        job.id,
                        item.row_number,
                        **{
                            field + "_message_id": sent.message_id,
                            "status": field,
                            "error_type": None,
                        },
                    )
                if not await self.running(job):
                    break
                item = next(
                    i
                    for i in await self.repository.items(job.id)
                    if i.row_number == item.row_number
                )
                if item.answer_message_id is not None:
                    validate_blob(item.question_vector)
                    validate_blob(item.answer_vector)
                    await self.repository.finish_item(job, item)
                if item.row_number % 5 == 0:
                    await notify()
            if await self.running(job):
                pending = [
                    i
                    for i in await self.repository.items(job.id)
                    if i.status not in {"done", "skipped"}
                ]
                await self.repository.state(job.id, "paused" if pending else "done")
        except Exception as error:
            log_operation_error(
                "faq_publish", chat_id, item.row_number if item else 0, error
            )
            if item:
                await self.repository.update(
                    job_id,
                    item.row_number,
                    error_type=type(error).__name__,
                    retry_at=time() + error.retry_after
                    if isinstance(error, TelegramRetryAfter)
                    else None,
                )
            await self.repository.state(job_id, "paused")
        finally:
            self.active.discard(chat_id)
            await notify()
