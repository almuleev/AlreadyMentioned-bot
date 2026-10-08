"""Safe HTML and shared editing helpers for administrator menus."""

from html import escape

from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions


def keyboard(*rows, prefix="cfg"):
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=text, callback_data=f"{prefix}:{action}")
                for text, action in row
            ]
            for row in rows
        ]
    )


def clip(text, length):
    return escape(text[:length] + ("…" if len(text) > length else ""))


async def edit_menu(bot, chat_id, menu_id, text, markup):
    try:
        await bot.edit_message_text(
            text,
            chat_id=chat_id,
            message_id=menu_id,
            reply_markup=markup,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as error:
        if "message is not modified" not in error.message.lower():
            raise
