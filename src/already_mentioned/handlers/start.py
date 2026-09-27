"""Brief setup guidance for the bot."""

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router()


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "AlreadyMentioned ищет сохранённые ответы в этом чате. "
        "Ответ администратора на вопрос сохраняется автоматически. "
        "Напишите /help, чтобы узнать правила и команды."
    )
