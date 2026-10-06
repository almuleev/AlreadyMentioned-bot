"""Remove feedback keyboards after a vote or one hour, including after restart."""

import asyncio
from time import time

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from already_mentioned.error_logging import log_operation_error
from already_mentioned.repositories.feedback import FeedbackRepository

FEEDBACK_LIFETIME_SECONDS = 3600
CLEANUP_INTERVAL_SECONDS = 15


class FeedbackCleanup:
    def __init__(self, bot: Bot, feedback: FeedbackRepository) -> None:
        self.bot = bot
        self.feedback = feedback

    async def schedule(self, message: Message, author_id: int) -> None:
        await self.feedback.schedule_keyboard(
            message.chat.id,
            message.message_id,
            message.date.timestamp() + FEEDBACK_LIFETIME_SECONDS,
            author_id,
        )

    async def remove(self, chat_id: int, message_id: int) -> None:
        # Persist immediate cleanup first so a failed request is retried.
        await self.feedback.schedule_keyboard(chat_id, message_id, time())
        await self._remove(chat_id, message_id)

    async def _remove(self, chat_id: int, message_id: int) -> None:
        try:
            await self.bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=message_id, reply_markup=None
            )
        except TelegramBadRequest as error:
            # Deleted/inaccessible messages and already removed keyboards need no retry.
            log_operation_error("feedback_cleanup", chat_id, message_id, error)
        except Exception as error:
            log_operation_error("feedback_cleanup", chat_id, message_id, error)
            return
        await self.feedback.delete_keyboard(chat_id, message_id)

    async def cleanup_due(self) -> None:
        for chat_id, message_id in await self.feedback.due_keyboards(time()):
            await self._remove(chat_id, message_id)

    async def run(self) -> None:
        while True:
            try:
                await self.cleanup_due()
            except Exception as error:
                log_operation_error("feedback_cleanup_queue", 0, 0, error)
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
