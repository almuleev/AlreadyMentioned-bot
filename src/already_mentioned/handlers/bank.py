"""Chat-scoped administrator bank browsing and guarded document changes."""

from html import escape

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ChatType, ParseMode
from aiogram.types import (
    CallbackQuery,
    ForceReply,
    InaccessibleMessage,
    InlineKeyboardButton,
    LinkPreviewOptions,
    Message,
)

from already_mentioned.bot.permissions import is_chat_admin
from already_mentioned.error_logging import log_operation_error
from already_mentioned.handlers.menu import clip, edit_menu, keyboard
from already_mentioned.models.entities import SolutionRevision
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.bank_sessions import BankSessions
from already_mentioned.services.embeddings import EmbeddingService, normalize_vector
from already_mentioned.services.reply_cache import ReplyCache

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
PROMPTS = {
    "search": "Поиск в банке ответов",
    "question": "Исправление сохранённого вопроса",
    "answer": "Исправление сохранённого ответа",
}


class BankExpired(ValueError):
    pass


def controls(*rows):
    result = keyboard(*rows, prefix="bank")
    result.inline_keyboard.append(
        [InlineKeyboardButton(text="К настройкам", callback_data="cfg:home")]
    )
    return result


async def bank_page(solutions, chat_id, page, *, view="-", query=None):
    total = await solutions.count_for_chat(chat_id, query=query)
    pages = max(1, (total + 2) // 3)
    if not 1 <= page <= pages:
        return "Страница устарела. Откройте список заново.", controls(
            [("Открыть список", "list:-:1")]
        )
    rows = await solutions.page_for_chat(
        chat_id, limit=3, offset=(page - 1) * 3, query=query
    )
    lines = [f"<b>Сохранённые ответы</b> — {page}/{pages}, всего {total}."]
    if query is not None:
        lines.append(
            f"Поиск: {clip(query, 200)}\n"
            "Фрагмент вопроса или ответа, без учёта регистра."
        )
    if not rows:
        lines.append(
            "Ничего не найдено. Попробуйте другой фрагмент текста."
            if query is not None
            else "Ответов пока нет. Администратор отвечает через reply на реальный "
            "вопрос участника; подходящая пара сохранится тихо."
        )
    buttons = []
    for row in rows:
        s = row.solution
        lines.append(
            f"<b>#{s.id}</b> {clip(s.question_text, 200)}\n"
            f"{clip(s.answer_text, 260)}\n"
            f'<a href="{escape(s.answer_link, quote=True)}">Оригинал</a> · '
            f"👍 {row.helpful} / 👎 {row.not_helpful}"
        )
        buttons.append([(f"Открыть #{s.id}", f"card:{s.id}:{view}:{page}:0")])
    nav = []
    if page > 1:
        nav.append(("Назад", f"list:{view}:{page - 1}"))
    if page < pages:
        nav.append(("Далее", f"list:{view}:{page + 1}"))
    if nav:
        buttons.append(nav)
    buttons.append([("Найти в банке", "search")])
    if query is not None:
        buttons.append([("Сбросить поиск", "list:-:1")])
    lines.append(
        "Откройте карточку, чтобы прочитать целиком, исправить или удалить пару."
    )
    return "\n\n".join(lines), controls(*buttons)


def query_for(sessions, view, chat_id, user_id, menu_id):
    if view == "-":
        return None
    state = sessions.view(view, chat_id, user_id, menu_id)
    if state is None:
        raise BankExpired(
            "Поиск устарел или открыт другим администратором. Откройте банк заново."
        )
    return state.query


async def card(solutions, chat_id, solution_id, view, page, part=0, *, notice=""):
    row = await solutions.summary_for_chat(chat_id, solution_id)
    if row is None:
        raise BankExpired(
            "Пара уже удалена или недоступна в этой группе. Откройте банк заново."
        )
    s = row.solution
    body = f"Вопрос:\n{s.question_text}\n\nОтвет:\n{s.answer_text}"
    # 1400 code points leave room for up to 2800 UTF-16 units plus controls.
    chunks = [body[i : i + 1400] for i in range(0, len(body), 1400)]
    part = min(part, len(chunks) - 1)
    text = (
        (escape(notice) + "\n\n" if notice else "")
        + f"<b>Пара #{s.id}</b> — часть {part + 1}/{len(chunks)}\n\n"
        + f"<blockquote>{escape(chunks[part])}</blockquote>\n\n"
        + f'<a href="{escape(s.question_link, quote=True)}">Исходный вопрос</a> · '
        + f'<a href="{escape(s.answer_link, quote=True)}">Исходный ответ</a>\n'
        + f"👍 {row.helpful} / 👎 {row.not_helpful}\n\n"
        + "Правки меняют банк для будущих подсказок. "
        "Исходные сообщения Telegram остаются прежними."
    )
    rows, nav = [], []
    if part > 0:
        nav.append(("Предыдущая часть", f"card:{s.id}:{view}:{page}:{part - 1}"))
    if part < len(chunks) - 1:
        nav.append(("Следующая часть", f"card:{s.id}:{view}:{page}:{part + 1}"))
    if nav:
        rows.append(nav)
    rows += [
        [("Исправить вопрос", f"question:{s.id}:{view}:{page}")],
        [("Исправить ответ", f"answer:{s.id}:{view}:{page}")],
        [("Удалить пару", f"delete:{s.id}:{view}:{page}")],
        [("К результатам" if view != "-" else "К списку", f"list:{view}:{page}")],
    ]
    return text, controls(*rows)


async def prompt(bot, message, user_id, kind, *, solution=None):
    description = (
        "Напишите фрагмент вопроса или ответа (до 200 символов)."
        if kind == "search"
        else "Напишите новый полный текст. Он заменит сохранённый "
        + ("вопрос." if kind == "question" else "ответ.")
    )
    current = ""
    if solution:
        text = solution.question_text if kind == "question" else solution.answer_text
        current = f"\nПара #{solution.id}. Сейчас: {clip(text, 250)}\n"
    return await bot.send_message(
        message.chat.id,
        f'{PROMPTS[kind]}\n<a href="tg://user?id={user_id}">Администратор</a>, '
        + description
        + current
        + "\nОтветьте на это сообщение в течение 5 минут.",
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
        message_thread_id=message.message_thread_id,
        reply_markup=ForceReply(
            selective=True, input_field_placeholder="Ответьте здесь"
        ),
    )


@router.callback_query(F.data.startswith("bank:"))
async def navigate(
    callback: CallbackQuery,
    bot: Bot,
    solutions: SolutionRepository,
    bank_sessions: BankSessions,
    reply_cache: ReplyCache,
):
    message = callback.message
    if (
        message is None
        or isinstance(message, InaccessibleMessage)
        or message.chat.type not in GROUP_TYPES
    ):
        await callback.answer("Откройте /settings в нужной группе.", show_alert=True)
        return
    chat_id, user_id, menu_id = (
        message.chat.id,
        callback.from_user.id,
        message.message_id,
    )
    changed = False
    try:
        if not await is_chat_admin(bot, chat_id, user_id):
            await callback.answer(
                "Банк доступен администратору группы.", show_alert=True
            )
            return
        parts = (callback.data or "").split(":")[1:]
        action = parts[0]
        if action in {"confirm", "cancel"} and len(parts) == 2:
            state = bank_sessions.take_deletion(parts[1], chat_id, user_id, menu_id)
            if state is None:
                raise BankExpired(
                    "Подтверждение устарело или открыто другим администратором."
                )
            if action == "confirm":
                if not await solutions.delete_revision(state.revision):
                    raise BankExpired(
                        "Пара уже изменена или удалена. Откройте её заново."
                    )
                changed = True
                reply_cache.discard(chat_id, state.revision.answer_message_id)
                text, markup = await bank_page(solutions, chat_id, 1)
                text = "Пара и её голоса удалены.\n\n" + text
            else:
                # A search can expire earlier than its deletion preview.
                view = (
                    state.view
                    if state.view == "-"
                    or bank_sessions.view(state.view, chat_id, user_id, menu_id)
                    else "-"
                )
                text, markup = await card(
                    solutions,
                    chat_id,
                    state.revision.id,
                    view,
                    state.page,
                    notice="Удаление отменено.",
                )
        elif action == "search" and len(parts) == 1:
            if bank_sessions.pending(chat_id, user_id, menu_id, "search"):
                await callback.answer("Ответьте на уже отправленный запрос.")
                return
            bank_sessions.discard_menu(chat_id, menu_id)
            sent = await prompt(bot, message, user_id, "search")
            bank_sessions.add_input(
                sent.message_id, chat_id, user_id, menu_id, "search"
            )
            text, markup = (
                "Ответьте на запрос поиска. Банк не изменится.",
                controls([("Отмена", "list:-:1")]),
            )
        else:
            if action == "list" and len(parts) == 3:
                view, page = parts[1], int(parts[2])
                query = query_for(bank_sessions, view, chat_id, user_id, menu_id)
                bank_sessions.discard_menu(chat_id, menu_id)
                text, markup = await bank_page(
                    solutions, chat_id, page, view=view, query=query
                )
            elif action in {"card", "question", "answer", "delete"} and len(parts) == (
                5 if action == "card" else 4
            ):
                solution_id, view, page = int(parts[1]), parts[2], int(parts[3])
                part = int(parts[4]) if action == "card" else 0
                if solution_id < 1 or page < 1 or part < 0:
                    raise ValueError
                query_for(bank_sessions, view, chat_id, user_id, menu_id)
                if action in {"question", "answer"} and bank_sessions.pending(
                    chat_id, user_id, menu_id, action, solution_id
                ):
                    await callback.answer("Ответьте на уже отправленный запрос.")
                    return
                bank_sessions.discard_menu(chat_id, menu_id)
                solution = await solutions.get_for_chat(chat_id, solution_id)
                if solution is None:
                    raise BankExpired("Пара уже удалена или недоступна в этой группе.")
                if action == "card":
                    text, markup = await card(
                        solutions, chat_id, solution_id, view, page, part
                    )
                elif action == "delete":
                    token = bank_sessions.deletion(
                        chat_id,
                        user_id,
                        menu_id,
                        SolutionRevision.of(solution),
                        view,
                        page,
                    )
                    text = (
                        f"<b>Удалить пару #{solution_id} и её голоса?</b>\n\n"
                        f"Вопрос: {clip(solution.question_text, 350)}\n\n"
                        f"Ответ: {clip(solution.answer_text, 650)}\n\n"
                        "Удаляется только эта пара из банка этой группы. "
                        "Исходные сообщения Telegram останутся. "
                        "Отменить удаление нельзя."
                    )
                    markup = controls(
                        [("Удалить эту пару и голоса", f"confirm:{token}")],
                        [("Прочитать целиком", f"card:{solution_id}:{view}:{page}:0")],
                        [("Отмена", f"cancel:{token}")],
                    )
                else:
                    sent = await prompt(
                        bot, message, user_id, action, solution=solution
                    )
                    bank_sessions.add_input(
                        sent.message_id,
                        chat_id,
                        user_id,
                        menu_id,
                        action,
                        revision=SolutionRevision.of(solution),
                        view=view,
                        page=page,
                    )
                    text = f"Пара #{solution_id}: ответьте на запрос нового текста."
                    markup = controls(
                        [("Отмена", f"card:{solution_id}:{view}:{page}:0")]
                    )
            else:
                raise ValueError
        await edit_menu(bot, chat_id, menu_id, text, markup)
        await callback.answer()
    except BankExpired as error:
        await callback.answer(str(error), show_alert=True)
    except Exception as error:
        log_operation_error("bank_callback", chat_id, menu_id, error)
        await callback.answer(
            "Удаление выполнено. Откройте /settings заново."
            if changed
            else "Не удалось открыть действие. Откройте банк через /settings.",
            show_alert=True,
        )


@router.message(F.text, F.reply_to_message, ~F.text.startswith("/"))
async def receive_input(
    message: Message,
    bot: Bot,
    solutions: SolutionRepository,
    embeddings: EmbeddingService,
    bank_sessions: BankSessions,
):
    reply = message.reply_to_message
    if (
        message.chat.type not in GROUP_TYPES
        or message.from_user is None
        or message.from_user.is_bot
    ):
        raise SkipHandler
    state = bank_sessions.input(message.chat.id, reply.message_id)
    if state is None:
        if (
            reply.from_user
            and reply.from_user.id == bot.id
            and any((reply.text or "").startswith(p) for p in PROMPTS.values())
        ):
            await message.reply(
                "Запрос устарел или отменён. Откройте банк через /settings."
            )
            return
        raise SkipHandler
    changed = False
    claimed = False
    try:
        if state.user_id != message.from_user.id or not await is_chat_admin(
            bot, message.chat.id, message.from_user.id
        ):
            await message.reply("Ответить может только открывший запрос администратор.")
            return
        text = message.text.strip()
        limit = 200 if state.kind == "search" else 4096
        if not text or len(text) > limit:
            await message.reply(
                f"Нужен непустой текст до {limit} символов. Ответьте ещё раз на запрос."
            )
            return
        state = bank_sessions.claim_input(
            reply.message_id, state.chat_id, state.user_id, state.menu_id
        )
        if state is None:
            await message.reply("Запрос уже обработан или отменён.")
            return
        claimed = True
        if state.kind == "search":
            token = bank_sessions.query(
                state.chat_id, state.user_id, state.menu_id, text
            )
            rendered, markup = await bank_page(
                solutions, state.chat_id, 1, view=token, query=text
            )
        else:
            # Recheck the current row before expensive encoding and atomically on write.
            current = await solutions.get_for_chat(state.chat_id, state.revision.id)
            if current is None or SolutionRevision.of(current) != state.revision:
                await message.reply(
                    "Пара уже изменена или удалена. Откройте её заново."
                )
                return
            vector = normalize_vector(await embeddings.embed_passage(text)).tobytes()
            if not await is_chat_admin(bot, state.chat_id, state.user_id):
                await message.reply(
                    "Права администратора изменились. Пара не изменена."
                )
                return
            if not bank_sessions.input_is_active(reply.message_id, state):
                await message.reply("Правка отменена или устарела. Пара не изменена.")
                return
            if not await solutions.replace_text(
                state.revision, field=state.kind, text=text, embedding=vector
            ):
                await message.reply(
                    "Пара уже изменена или удалена. Откройте её заново."
                )
                return
            changed = True
            view = (
                state.view
                if state.view == "-"
                or bank_sessions.view(
                    state.view, state.chat_id, state.user_id, state.menu_id
                )
                else "-"
            )
            rendered, markup = await card(
                solutions,
                state.chat_id,
                state.revision.id,
                view,
                state.page,
                notice="Изменение сохранено.",
            )
        if changed or bank_sessions.input_is_active(reply.message_id, state):
            await edit_menu(bot, state.chat_id, state.menu_id, rendered, markup)
    except Exception as error:
        log_operation_error("bank_input", message.chat.id, message.message_id, error)
        await message.reply(
            "Изменение сохранено. Откройте карточку заново через /settings."
            if changed
            else "Действие не выполнено. Откройте банк через /settings заново."
        )
    finally:
        if claimed:
            bank_sessions.finish_input(message.chat.id, reply.message_id)
