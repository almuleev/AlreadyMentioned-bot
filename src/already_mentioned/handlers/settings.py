"""Administrator settings, reversible pause and read-only search previews."""

import math
from html import escape

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ChatType, ParseMode
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    ForceReply,
    InaccessibleMessage,
    Message,
)

from already_mentioned.bot.permissions import is_chat_admin
from already_mentioned.config import (
    DEFAULT_HYBRID_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from already_mentioned.error_logging import log_operation_error
from already_mentioned.handlers.bank import bank_page
from already_mentioned.handlers.menu import clip, edit_menu, keyboard
from already_mentioned.handlers.start import HELP, connection
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.bank_sessions import BankSessions
from already_mentioned.services.faq_sessions import FaqSessions
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.search import SearchService
from already_mentioned.services.settings_sessions import SettingsSessions

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
PROMPTS = {"question": "Проверка вопроса", "threshold": "Новый порог подсказок"}


def back():
    return keyboard([("Назад к настройкам", "home")])


def home_keyboard():
    return keyboard(
        [("Подсказки", "suggestions"), ("Частота ответов", "frequency")],
        [("Сохранённые ответы", "bank:1")],
        [("Проверить вопрос", "question")],
        [("Состояние", "status"), ("Помощь", "help")],
    )


async def home(chats, solutions, chat_id):
    chat = await chats.get_chat(chat_id)
    count = await solutions.count_for_chat(chat_id)
    state = "включены" if chat.suggestions_enabled else "на паузе"
    return (
        f"<b>Настройки этой группы</b>\n\nПодсказки: {state}.\n"
        f"Сохранённых ответов: {count}.\n"
        "Автосохранение ответов администратора работает.\n\n"
        "Ниже порог — больше подсказок. Перед изменением покажу примеры."
    ), home_keyboard()


async def inspect(search, chat_id, question):
    candidates = await search.find_candidates(
        chat_id, question, limit=5, backfill=False
    )
    return await search.rerank_candidates(question, candidates)


async def threshold_preview(
    bot,
    chats,
    solutions,
    search,
    settings_sessions,
    chat_id,
    user_id,
    menu_id,
    proposed,
):
    chat = await chats.get_chat(chat_id)
    current = search.threshold_for(chat)
    rows = await solutions.page_for_chat(chat_id, limit=3, offset=0)
    lines = [
        f"<b>Предварительный просмотр</b>\nПорог: {current:.3f} → {proposed:.3f}.",
        "Ниже — больше подсказок, выше — меньше. Оценка не является "
        "вероятностью правильности ответа.",
    ]
    if not chat.suggestions_enabled:
        lines.append("Подсказки на паузе. Ниже — сравнение для включённых подсказок.")
    if not rows:
        lines.append(
            "Банк пока пуст: примеры недоступны. Сначала сохраните реальный "
            "ответ администратора или подтвердите порог без примеров."
        )
    else:
        lines.append(
            "Примеры взяты из сохранённых вопросов, включая совпадение "
            "с самими собой. Это иллюстрация порога, не проверка качества. "
            "Свой вопрос можно проверить через меню."
        )
        for item in rows:
            question = item.solution.question_text
            ranked = await inspect(search, chat_id, question)
            passed = is_question_candidate(question)
            before = passed and bool(ranked) and ranked[0].similarity >= current
            after = passed and bool(ranked) and ranked[0].similarity >= proposed
            lines.append(
                f"• {clip(question, 150)}\n"
                f"Сейчас: {'ответ' if before else 'молчание'}; "
                f"после: {'ответ' if after else 'молчание'}."
            )
            if ranked:
                lines.append(
                    f"Ближайший ответ ({ranked[0].similarity:.3f}): "
                    + clip(ranked[0].solution.answer_text, 150)
                )
    previous = (
        chat.hybrid_threshold if search.mode == "hybrid" else chat.similarity_threshold
    )
    token = settings_sessions.preview(
        chat_id, user_id, menu_id, search.mode, previous, proposed
    )
    await edit_menu(
        bot,
        chat_id,
        menu_id,
        "\n\n".join(lines),
        keyboard([("Применить порог", f"apply:{token}")], [("Отмена", "frequency")]),
    )


async def question_result(bot, chats, search, chat_id, menu_id, question):
    chat = await chats.get_chat(chat_id)
    threshold = search.threshold_for(chat)
    passed = is_question_candidate(question)
    ranked = await inspect(search, chat_id, question)
    if not chat.suggestions_enabled:
        reason = "Бот промолчит: подсказки на паузе."
    elif not passed:
        reason = "Бот промолчит: сообщение не прошло фильтр вопросов."
    elif not ranked:
        reason = "Бот промолчит: в группе пока нет доступных сохранённых ответов."
    elif ranked[0].similarity < threshold:
        reason = "Бот промолчит: ближайший ответ не достиг порога."
    else:
        reason = "Бот предложил бы этот ответ."
    lines = [
        "<b>Проверка вопроса</b>",
        clip(question, 240),
        reason,
        f"Фильтр вопросов: {'пройден' if passed else 'не пройден'}. "
        f"Порог: {threshold:.3f}.",
    ]
    if ranked:
        best = ranked[0]
        lines += [
            f"Ближайший ответ, оценка {best.similarity:.3f} "
            "(не вероятность правильности):",
            f"Сохранённый вопрос: {clip(best.solution.question_text, 240)}",
            f"<blockquote>{clip(best.solution.answer_text, 700)}</blockquote>",
            f'<a href="{escape(best.solution.answer_link, quote=True)}">'
            "Оригинал ответа</a>",
        ]
        if len(ranked) > 1:
            lines.append(
                "Другие кандидаты:\n"
                + "\n".join(
                    f"• #{m.solution.id}: {clip(m.solution.question_text, 110)} "
                    f"({m.similarity:.3f})"
                    for m in ranked[1:3]
                )
            )
    lines.append("Проверка не сохраняет вопрос и не меняет банк или настройки.")
    await edit_menu(
        bot,
        chat_id,
        menu_id,
        "\n\n".join(lines),
        keyboard([("Другой вопрос", "question")], [("Назад к настройкам", "home")]),
    )


@router.message(Command("settings"))
async def settings_command(
    message: Message, bot: Bot, chats: ChatRepository, solutions: SolutionRepository
):
    if message.chat.type not in GROUP_TYPES:
        await message.answer(
            "Откройте /settings в нужной группе. Настройки доступны её администраторам."
        )
        return
    try:
        if message.from_user is None or not await is_chat_admin(
            bot, message.chat.id, message.from_user.id
        ):
            await message.answer("Настройки доступны администратору этой группы.")
            return
        await chats.ensure_chat(message.chat.id, message.chat.title or "")
        text, markup = await home(chats, solutions, message.chat.id)
        await message.answer(text, reply_markup=markup, parse_mode=ParseMode.HTML)
    except Exception as error:
        log_operation_error("settings_open", message.chat.id, message.message_id, error)
        await message.answer("Не удалось открыть настройки. Повторите /settings.")


@router.callback_query(F.data.startswith("cfg:"))
async def navigate(
    callback: CallbackQuery,
    bot: Bot,
    chats: ChatRepository,
    solutions: SolutionRepository,
    search: SearchService,
    settings_sessions: SettingsSessions,
    bank_sessions: BankSessions | None = None,
    faq_sessions: FaqSessions | None = None,
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
                "Настройки доступны администратору группы.", show_alert=True
            )
            return
        if bank_sessions is not None:
            bank_sessions.discard_menu(chat_id, menu_id)
        if faq_sessions is not None:
            faq_sessions.discard_menu(chat_id, menu_id)
        action = (callback.data or "").removeprefix("cfg:")
        if not await chats.get_chat(chat_id):
            await chats.ensure_chat(chat_id, message.chat.title or "")
        if action.startswith("apply:"):
            request = settings_sessions.take_preview(
                action[6:], chat_id, user_id, menu_id
            )
            if (
                request is None
                or request.mode != search.mode
                or not await chats.confirm_threshold(
                    chat_id,
                    request.previous,
                    request.proposed,
                    hybrid=search.mode == "hybrid",
                )
            ):
                await callback.answer(
                    "Просмотр устарел или открыт другим администратором. "
                    "Выберите частоту заново.",
                    show_alert=True,
                )
                return
            changed = True
            text, markup = await home(chats, solutions, chat_id)
            text = f"Порог {request.proposed:.3f} сохранён.\n\n" + text
        else:
            settings_sessions.discard_previews(chat_id, menu_id)
            chat = await chats.get_chat(chat_id)
            if action == "home":
                text, markup = await home(chats, solutions, chat_id)
            elif action == "suggestions" or action in {"set:0", "set:1"}:
                if action.startswith("set:"):
                    await chats.set_suggestions_enabled(chat_id, action == "set:1")
                    changed = True
                    chat = await chats.get_chat(chat_id)
                text = (
                    "<b>Подсказки</b>\n\n"
                    + ("Включены." if chat.suggestions_enabled else "На паузе.")
                    + " Автосохранение ответов администратора продолжает работать."
                )
                markup = keyboard(
                    [
                        (
                            "Приостановить" if chat.suggestions_enabled else "Включить",
                            "set:0" if chat.suggestions_enabled else "set:1",
                        )
                    ],
                    [("Частота ответов", "frequency")],
                    [("К настройкам", "home")],
                )
            elif action == "frequency":
                text = (
                    "<b>Частота ответов</b>\n\n"
                    f"Текущий порог: {search.threshold_for(chat):.3f}.\n"
                    "Ниже — больше подсказок и больше риск ошибки; "
                    "выше — меньше подсказок. "
                    "Сначала покажу примеры, затем попрошу подтвердить изменение."
                )
                markup = keyboard(
                    [
                        ("Больше подсказок", "preview:more"),
                        ("Меньше подсказок", "preview:less"),
                    ],
                    [
                        ("Порог выпуска", "preview:default"),
                        ("Ввести число", "threshold"),
                    ],
                    [("К настройкам", "home")],
                )
            elif action in {"preview:more", "preview:less", "preview:default"}:
                proposed = (
                    (
                        DEFAULT_HYBRID_THRESHOLD
                        if search.mode == "hybrid"
                        else DEFAULT_SIMILARITY_THRESHOLD
                    )
                    if action == "preview:default"
                    else round(
                        min(
                            1,
                            max(
                                0,
                                search.threshold_for(chat)
                                + (-0.05 if action == "preview:more" else 0.05),
                            ),
                        ),
                        3,
                    )
                )
                await callback.answer()
                await threshold_preview(
                    bot,
                    chats,
                    solutions,
                    search,
                    settings_sessions,
                    chat_id,
                    user_id,
                    menu_id,
                    proposed,
                )
                return
            elif action in {"question", "threshold"}:
                if settings_sessions.has_input(chat_id, user_id, menu_id, action):
                    await callback.answer(
                        "Запрос уже отправлен. Ответьте на сообщение бота.",
                        show_alert=True,
                    )
                    return
                text = (
                    "Напишите проверяемый вопрос в ответ на это сообщение."
                    if action == "question"
                    else "Напишите число от 0 до 1 в ответ "
                    "на это сообщение. До подтверждения настройка не изменится."
                )
                sent = await bot.send_message(
                    chat_id,
                    f'{PROMPTS[action]}\n<a href="tg://user?id={user_id}">'
                    "Администратор</a>, " + text + "\nЗапрос действует 5 минут.",
                    parse_mode=ParseMode.HTML,
                    message_thread_id=message.message_thread_id,
                    reply_markup=ForceReply(
                        selective=True, input_field_placeholder="Ответьте здесь"
                    ),
                )
                settings_sessions.add_input(
                    chat_id, sent.message_id, user_id, menu_id, action
                )
                await callback.answer("Ответьте на сообщение бота.")
                return
            elif action.startswith("bank:") and action[5:].isdigit():
                text, markup = await bank_page(solutions, chat_id, int(action[5:]))
            elif action == "status":
                report, _ = await connection(bot, message.chat, solutions)
                state = "включены" if chat.suggestions_enabled else "на паузе"
                text = (
                    escape(report) + f"\n\nПодсказки: {state}.\n"
                    f"Поиск: {escape(search.mode_label)}; "
                    f"порог {search.threshold_for(chat):.3f}."
                )
                markup = back()
            elif action == "help":
                text, markup = (
                    escape(HELP) + "\n\n/settings — настройки этой группы.",
                    back(),
                )
            else:
                await callback.answer(
                    "Кнопка устарела. Откройте /settings заново.", show_alert=True
                )
                return
        await edit_menu(bot, chat_id, menu_id, text, markup)
        await callback.answer()
    except Exception as error:
        log_operation_error("settings_callback", chat_id, menu_id, error)
        await callback.answer(
            "Настройка сохранена. Откройте /settings, чтобы увидеть состояние."
            if changed
            else "Не удалось выполнить действие. Откройте /settings заново.",
            show_alert=True,
        )


@router.message(F.text, F.reply_to_message, ~F.text.startswith("/"))
async def receive_input(
    message: Message,
    bot: Bot,
    chats: ChatRepository,
    solutions: SolutionRepository,
    search: SearchService,
    settings_sessions: SettingsSessions,
):
    reply = message.reply_to_message
    if (
        message.chat.type not in GROUP_TYPES
        or message.from_user is None
        or getattr(message.from_user, "is_bot", False)
    ):
        raise SkipHandler
    request = settings_sessions.get_input(message.chat.id, reply.message_id)
    if request is None:
        if (
            reply.from_user
            and reply.from_user.id == bot.id
            and any((reply.text or "").startswith(p) for p in PROMPTS.values())
        ):
            await message.reply("Запрос устарел. Откройте /settings заново.")
            return
        raise SkipHandler
    try:
        allowed = request.user_id == message.from_user.id and await is_chat_admin(
            bot, message.chat.id, message.from_user.id
        )
    except Exception as error:
        log_operation_error(
            "settings_input_permissions", message.chat.id, message.message_id, error
        )
        await message.reply("Не удалось проверить права. Попробуйте ещё раз.")
        return
    if not allowed:
        await message.reply(
            "Ответить на этот запрос может только открывший его администратор."
        )
        return
    try:
        if request.kind == "threshold":
            try:
                value = float(message.text.strip().replace(",", "."))
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError
            except ValueError:
                await message.reply(
                    "Нужно число от 0 до 1. Ответьте ещё раз на запрос бота."
                )
                return
            await threshold_preview(
                bot,
                chats,
                solutions,
                search,
                settings_sessions,
                request.chat_id,
                request.user_id,
                request.menu_id,
                value,
            )
        else:
            await question_result(
                bot, chats, search, request.chat_id, request.menu_id, message.text
            )
        settings_sessions.remove_input(message.chat.id, reply.message_id)
    except Exception as error:
        log_operation_error(
            "settings_input", message.chat.id, message.message_id, error
        )
        await message.reply("Проверка не удалась. Откройте /settings заново.")
