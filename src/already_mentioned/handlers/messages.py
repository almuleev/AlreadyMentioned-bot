"""Offer one confirmed answer for a likely new question."""

import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from already_mentioned.handlers.solve import remember_admin_answer
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reply_cache import ReplyCache
from already_mentioned.services.search import SearchService

router = Router()


@router.message(F.text)
async def handle_text(
    message: Message, bot: Bot, reply_cache: ReplyCache, search: SearchService
) -> None:
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return
    if message.from_user is None or message.from_user.is_bot or not message.text:
        return
    if message.text.startswith("/"):
        return
    if message.reply_to_message is not None:
        if await remember_admin_answer(message, bot, reply_cache):
            return
    if not is_question_candidate(message.text):
        return
    try:
        match = await search.find_best(message.chat.id, message.text)
    except Exception:
        logging.exception("Search failed for chat %s", message.chat.id)
        return
    if match is None:
        return
    callback_prefix = f"vote:{match.solution.id}:{message.message_id}"
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👍 Помогло", callback_data=f"{callback_prefix}:helpful"
                ),
                InlineKeyboardButton(
                    text="👎 Не подходит",
                    callback_data=f"{callback_prefix}:not_helpful",
                ),
            ]
        ]
    )
    await message.answer(
        f"Похоже, это уже обсуждали. Возможно, поможет: {match.solution.answer_link}",
        reply_markup=keyboard,
    )
