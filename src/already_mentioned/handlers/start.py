"""Brief setup guidance for the bot."""

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router()


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "AlreadyMentioned ищет подтверждённые ответы в этом чате. "
        "Администратор сохраняет решение через /solve. "
        "Напишите /help, чтобы узнать правила и команды."
    )
