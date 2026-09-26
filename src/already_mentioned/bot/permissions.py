"""Telegram chat permission checks."""

from aiogram import Bot
from aiogram.enums import ChatMemberStatus

ADMIN_STATUSES = {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR}


async def is_chat_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    member = await bot.get_chat_member(chat_id, user_id)
    return member.status in ADMIN_STATUSES
