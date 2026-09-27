"""Status, threshold, help, and confirmed chat-data removal."""

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from already_mentioned.bot.permissions import is_chat_admin
from already_mentioned.handlers.solve import SOLVE_USAGE
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.confirmations import ForgetConfirmations
from already_mentioned.services.reply_cache import ReplyCache

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}


async def _require_group_admin(message: Message, bot: Bot) -> bool:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("Команда доступна только в группе или супергруппе.")
        return False
    if message.from_user is None or not await is_chat_admin(
        bot, message.chat.id, message.from_user.id
    ):
        await message.answer("Команда доступна только администратору чата.")
        return False
    return True


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(
        "AlreadyMentioned предлагает ссылки на сохранённые ответы этого чата.\n"
        f"{SOLVE_USAGE}\n"
        "/undo — ответить на сохранённый ответ и удалить его (администратор).\n"
        "/status — число решений и порог совпадения.\n"
        "/threshold 0.88 — изменить порог (администратор).\n"
        "/forget — удалить решения и оценки после подтверждения (администратор)."
    )


@router.message(Command("status"))
async def status_command(
    message: Message, chats: ChatRepository, solutions: SolutionRepository
) -> None:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("/status работает только в группе или супергруппе.")
        return
    await chats.ensure_chat(message.chat.id, message.chat.title or "")
    chat = await chats.get_chat(message.chat.id)
    count = await solutions.count_for_chat(message.chat.id)
    await message.answer(
        f"Сохранённых решений: {count}. "
        f"Порог совпадения: {chat.similarity_threshold:.2f}."
    )


@router.message(Command("threshold"))
async def threshold_command(message: Message, bot: Bot, chats: ChatRepository) -> None:
    if not await _require_group_admin(message, bot):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Укажите порог от 0 до 1: /threshold 0.88")
        return
    try:
        value = float(parts[1].replace(",", "."))
        if not 0 <= value <= 1:
            raise ValueError
    except ValueError:
        await message.answer("Порог должен быть числом от 0 до 1, например 0.88.")
        return
    await chats.ensure_chat(message.chat.id, message.chat.title or "")
    await chats.set_threshold(message.chat.id, value)
    await message.answer(f"Порог совпадения этого чата: {value:.2f}.")


@router.message(Command("forget"))
async def forget_command(
    message: Message, bot: Bot, confirmations: ForgetConfirmations
) -> None:
    if not await _require_group_admin(message, bot):
        return
    token = confirmations.create(message.chat.id, message.from_user.id)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Удалить данные", callback_data=f"forget:confirm:{token}"
                ),
                InlineKeyboardButton(
                    text="Отмена", callback_data=f"forget:cancel:{token}"
                ),
            ]
        ]
    )
    await message.answer(
        "Удалить все подтверждённые решения и оценки этого чата? "
        "Действие нельзя отменить.",
        reply_markup=keyboard,
    )


@router.callback_query(F.data.startswith("forget:"))
async def confirm_forget(
    callback: CallbackQuery,
    bot: Bot,
    confirmations: ForgetConfirmations,
    chats: ChatRepository,
    reply_cache: ReplyCache,
) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 3 or parts[1] not in {"confirm", "cancel"}:
        await callback.answer("Некорректное подтверждение.")
        return
    if callback.message is None or callback.message.chat.type not in GROUP_TYPES:
        await callback.answer("Сообщение больше недоступно.")
        return
    chat_id = callback.message.chat.id
    user_id = callback.from_user.id
    if not await is_chat_admin(bot, chat_id, user_id):
        await callback.answer("Только администратор может подтвердить удаление.")
        return
    if not confirmations.consume(parts[2], chat_id, user_id):
        await callback.answer("Подтверждение истекло или было использовано.")
        return
    if parts[1] == "cancel":
        await callback.answer("Удаление отменено.")
    else:
        deleted = await chats.forget_chat_data(chat_id)
        reply_cache.clear_chat(chat_id)
        await callback.answer(f"Удалено решений: {deleted}.")
    if isinstance(callback.message, Message):
        await callback.message.edit_reply_markup(reply_markup=None)
