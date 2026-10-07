"""Chat-scoped status, solution management, and confirmed data removal."""

from html import escape

from aiogram import Bot, F, Router
from aiogram.enums import ChatType, ParseMode
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from already_mentioned.bot.permissions import is_chat_admin
from already_mentioned.config import DEFAULT_SIMILARITY_THRESHOLD
from already_mentioned.error_logging import log_operation_error
from already_mentioned.handlers.solve import SOLVE_USAGE
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.confirmations import ForgetConfirmations
from already_mentioned.services.embeddings import EmbeddingService, normalize_vector
from already_mentioned.services.reply_cache import ReplyCache
from already_mentioned.services.search import SearchService

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
PAGE_SIZE = 5
ANSWER_PREVIEW = 160


def _preview(text: str, limit: int) -> str:
    return escape(text[:limit].rstrip() + ("…" if len(text) > limit else ""))


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
async def help_command(message: Message, search: SearchService | None = None) -> None:
    await message.answer(
        "AlreadyMentioned предлагает ссылки на сохранённые ответы этого чата.\n"
        f"{SOLVE_USAGE}\n"
        "/undo — ответить на сохранённый ответ и удалить его (администратор).\n"
        "/status — число решений и порог совпадения.\n"
        "/solutions [страница] — список решений этого чата (администратор).\n"
        "/editquestion ID новый текст — reply на исходный ответ, исправить вопрос.\n"
        "/editanswer ID новый текст — reply на исходный ответ, исправить ответ.\n"
        "ID и ссылку на исходный ответ возьмите из /solutions.\n"
        "/threshold ЧИСЛО — изменить порог активного поиска "
        "(администратор).\n"
        "/forget — удалить решения и оценки после подтверждения (администратор)."
    )


@router.message(Command("solutions"))
async def solutions_command(
    message: Message, bot: Bot, solutions: SolutionRepository
) -> None:
    if not await _require_group_admin(message, bot):
        return
    parts = (message.text or "").split()
    if len(parts) > 2 or (len(parts) == 2 and not parts[1].isdecimal()):
        await message.answer("Укажите номер страницы: /solutions 1")
        return
    page = int(parts[1]) if len(parts) == 2 else 1
    if page < 1:
        await message.answer("Номер страницы должен быть положительным: /solutions 1")
        return
    total = await solutions.count_for_chat(message.chat.id)
    if total == 0:
        await message.answer("В этом чате пока нет сохранённых решений.")
        return
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    if page > pages:
        await message.answer(f"Такой страницы нет. Доступны страницы 1–{pages}.")
        return
    entries = await solutions.page_for_chat(
        message.chat.id, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE
    )
    header = f"Решения этого чата — страница {page}/{pages} (всего {total}):"
    entries_html = []
    for entry in entries:
        solution = entry.solution
        question = escape(solution.question_text)
        entries_html.append(
            f"\n<b>#{solution.id}</b> {question}\n"
            f"Ответ: {_preview(solution.answer_text, ANSWER_PREVIEW)}\n"
            f'<a href="{escape(solution.question_link, quote=True)}">Вопрос</a> · '
            f'<a href="{escape(solution.answer_link, quote=True)}">Ответ</a> · '
            f"👍 {entry.helpful} / 👎 {entry.not_helpful}"
        )
    navigation = []
    if page < pages:
        navigation.append(f"Далее: /solutions {page + 1}")
    if page > 1:
        navigation.append(f"Назад: /solutions {page - 1}")
    body = "\n".join([header, *entries_html, *navigation])
    if len(body) <= 4000:
        await message.answer(body, parse_mode=ParseMode.HTML)
        return
    await message.answer(header)
    for entry, rendered in zip(entries, entries_html, strict=True):
        if len(rendered) <= 4000:
            await message.answer(rendered, parse_mode=ParseMode.HTML)
            continue
        solution = entry.solution
        for start in range(0, len(solution.question_text), 600):
            chunk = escape(solution.question_text[start : start + 600])
            await message.answer(
                f"<b>#{solution.id} вопрос</b>\n{chunk}", parse_mode=ParseMode.HTML
            )
        await message.answer(
            f"Ответ: {_preview(solution.answer_text, ANSWER_PREVIEW)}\n"
            f'<a href="{escape(solution.question_link, quote=True)}">Вопрос</a> · '
            f'<a href="{escape(solution.answer_link, quote=True)}">Ответ</a> · '
            f"👍 {entry.helpful} / 👎 {entry.not_helpful}",
            parse_mode=ParseMode.HTML,
        )
    if navigation:
        await message.answer("\n".join(navigation))


@router.message(Command("editquestion", "editanswer"))
async def edit_solution_command(
    message: Message,
    bot: Bot,
    solutions: SolutionRepository,
    embeddings: EmbeddingService,
) -> None:
    if not await _require_group_admin(message, bot):
        return
    parts = (message.text or "").split(maxsplit=2)
    command = parts[0].split("@", maxsplit=1)[0].lstrip("/") if parts else ""
    if len(parts) != 3 or not parts[1].isdecimal() or not parts[2].strip():
        await message.answer(
            "Ответьте на исходный сохранённый ответ командой "
            f"/{command} ID новый текст. ID указан в /solutions."
        )
        return
    original = message.reply_to_message
    if original is None:
        await message.answer(
            "Команду нужно отправить reply на исходный сохранённый ответ."
        )
        return
    solution_id = int(parts[1])
    solution = await solutions.get_for_chat(message.chat.id, solution_id)
    if solution is None or solution.answer_message_id != original.message_id:
        await message.answer(
            "Решение с этим ID и исходным ответом в этом чате не найдено."
        )
        return
    new_text = parts[2].strip()
    if command == "editquestion":
        try:
            embedding = normalize_vector(
                await embeddings.embed_passage(new_text)
            ).tobytes()
        except Exception as error:
            log_operation_error(
                "edit_embedding", message.chat.id, message.message_id, error
            )
            await message.answer("Не удалось пересчитать вопрос. Решение не изменено.")
            return
        updated = await solutions.update_question(
            message.chat.id, solution_id, original.message_id, new_text, embedding
        )
    else:
        try:
            embedding = normalize_vector(
                await embeddings.embed_passage(new_text)
            ).tobytes()
        except Exception as error:
            log_operation_error(
                "edit_embedding", message.chat.id, message.message_id, error
            )
            await message.answer("Не удалось пересчитать ответ. Решение не изменено.")
            return
        updated = await solutions.update_answer(
            message.chat.id, solution_id, original.message_id, new_text, embedding
        )
    await message.answer(
        "Решение исправлено." if updated else "Решение больше недоступно."
    )


@router.message(Command("status"))
async def status_command(
    message: Message, chats: ChatRepository, solutions: SolutionRepository,
    search: SearchService | None = None,
) -> None:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("/status работает только в группе или супергруппе.")
        return
    await chats.ensure_chat(message.chat.id, message.chat.title or "")
    chat = await chats.get_chat(message.chat.id)
    count = await solutions.count_for_chat(message.chat.id)
    threshold = search.threshold_for(chat) if search else chat.similarity_threshold
    mode = f"Поиск: {search.mode_label}. " if search else ""
    await message.answer(
        f"Сохранённых решений: {count}. "
        f"{mode}Порог совпадения: {threshold:.3f}."
    )


@router.message(Command("threshold"))
async def threshold_command(
    message: Message, bot: Bot, chats: ChatRepository,
    search: SearchService | None = None,
) -> None:
    if not await _require_group_admin(message, bot):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer(
            f"Укажите порог от 0 до 1: /threshold {DEFAULT_SIMILARITY_THRESHOLD:.3f}"
        )
        return
    try:
        value = float(parts[1].replace(",", "."))
        if not 0 <= value <= 1:
            raise ValueError
    except ValueError:
        await message.answer(
            "Порог должен быть числом от 0 до 1, "
            f"например {DEFAULT_SIMILARITY_THRESHOLD:.3f}."
        )
        return
    await chats.ensure_chat(message.chat.id, message.chat.title or "")
    await chats.set_threshold(message.chat.id, value,
                              hybrid=search is not None and search.mode == "hybrid")
    await message.answer(f"Порог совпадения этого чата: {value:.3f}.")


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
