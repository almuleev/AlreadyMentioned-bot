"""Administrator FAQ drafts, CSV previews and confirmed publication."""

from html import escape
from time import time

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ChatType, ParseMode
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
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
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.faq import FaqRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.bank_sessions import BankSessions
from already_mentioned.services.faq_data import (
    MAX_FILE,
    TEMPLATE,
    FaqPair,
    LimitedBuffer,
    parse_csv,
    review,
    valid_text,
)
from already_mentioned.services.faq_publisher import FaqPublisher, publication_text
from already_mentioned.services.faq_sessions import FaqSessions
from already_mentioned.services.settings_sessions import SettingsSessions

router = Router()
PROMPT = "FAQ · Ввод"


def controls(*rows):
    markup = keyboard(*rows, prefix="faq")
    markup.inline_keyboard.append(
        [InlineKeyboardButton(text="К настройкам", callback_data="cfg:home")]
    )
    return markup


def target(thread_id):
    return f"тема №{thread_id} этой группы" if thread_id else "основной чат этой группы"


async def request(bot, draft, kind):
    if draft.kind == kind and draft.prompt_id:
        return
    draft.review = review((), ())
    instruction = {
        "question": "Напишите полный типовой вопрос.",
        "answer": "Напишите полный ответ на вопрос.",
        "file": "Прикрепите CSV в UTF-8 с колонками вопрос и ответ. "
        "До 50 пар и 512 КиБ.",
    }[kind]
    sent = await bot.send_message(
        draft.chat_id,
        f'{PROMPT}\n<a href="tg://user?id={draft.user_id}">'
        f"Администратор</a>, {instruction}\nОтветьте на это сообщение. "
        "Запрос действует 15 минут.",
        parse_mode=ParseMode.HTML,
        message_thread_id=draft.thread_id,
        reply_markup=ForceReply(selective=True),
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )
    draft.kind, draft.prompt_id = kind, sent.message_id


def preview(draft, page=0):
    result, token = draft.review, draft.token
    pages = max(1, (len(result.pairs) + 2) // 3)
    page = min(max(0, page), pages - 1)
    lines = [
        "<b>Проверка FAQ перед публикацией</b>",
        f"Будет добавлено: {len(result.pairs)}. Дубли: {result.duplicates}. "
        f"Ошибки: {len(result.issues)}.",
        f"Место: {target(draft.thread_id)}. Каждая пара создаст два сообщения "
        "с настоящими ссылками. Модель будет искать по сохранённым текстам.",
    ]
    rows = []
    for index in range(page * 3, min(len(result.pairs), page * 3 + 3)):
        p = result.pairs[index]
        lines.append(
            f"<b>Строка {p.row}</b>\n{clip(p.question, 140)}\n{clip(p.answer, 200)}"
        )
        rows.append([(f"Прочитать строку {p.row}", f"pair:{token}:{index}:0")])
    if result.issues:
        lines.append(
            "Ошибочные строки не будут загружены. "
            "Откройте список ошибок перед подтверждением."
        )
        rows.append([("Посмотреть ошибки", f"issues:{token}:0")])
    nav = []
    if page:
        nav.append(("Назад", f"preview:{token}:{page - 1}"))
    if page < pages - 1:
        nav.append(("Далее", f"preview:{token}:{page + 1}"))
    if nav:
        rows.append(nav)
    if result.pairs:
        rows.append([(f"Опубликовать FAQ: {len(result.pairs)}", f"save:{token}")])
    if draft.question:
        rows.append(
            [
                ("Исправить вопрос", f"question:{token}"),
                ("Исправить ответ", f"answer:{token}"),
            ]
        )
    else:
        rows.append([("Другой CSV", f"file:{token}")])
    rows.append([("Отмена", f"cancel:{token}")])
    return "\n\n".join(lines), controls(*rows)


async def job_report(bot, repo, job_id, chat_id, user_id, menu_id):
    job = await repo.get(job_id, chat_id, user_id)
    if job is None:
        await edit_menu(
            bot, chat_id, menu_id, "Загрузка отменена или удалена.", controls()
        )
        return
    items = await repo.items(job_id)
    done = sum(i.status == "done" for i in items)
    skipped = sum(i.status == "skipped" for i in items)
    pending = next((i for i in items if i.status not in {"done", "skipped"}), None)
    lines = [
        f"<b>Загрузка FAQ</b> — {target(job.thread_id)}",
        f"Сохранено: {done}/{len(items)}. Пропущено: {skipped}.",
        "Завершена."
        if job.status == "done"
        else "Отменена. Сохранённые пары остаются в банке."
        if job.status == "canceled"
        else "Приостановлена."
        if job.status != "running"
        else "Идёт публикация.",
    ]
    rows = []
    if job.status == "running":
        rows.append([("Приостановить", f"pause:{job_id}")])
    elif pending and job.status != "canceled":
        lines.append(f"Следующая строка: {pending.row_number}.")
        if pending.status.startswith("sending_"):
            lines.append(
                "Telegram не подтвердил отправку. Проверьте выбранную тему. "
                "Если сообщение уже есть, ответьте на него командой /faqsource. "
                "Если его нет, явно подтвердите повторную отправку."
            )
            rows.append([("Сообщения нет — отправить повторно", f"retry:{job_id}")])
        else:
            if pending.error_type:
                lines.append(
                    "Не удалось выполнить строку. Проверьте права бота, "
                    "доступность темы и соединение; затем продолжите. "
                    f"Тип ошибки: {escape(pending.error_type)}."
                )
            if pending.retry_at and pending.retry_at > time():
                seconds = int(pending.retry_at - time()) + 1
                lines.append(f"Telegram просит подождать ещё {seconds} сек.")
            rows.append([("Продолжить загрузку", f"resume:{job_id}")])
        rows.append([("Пропустить проблемную строку", f"skip:{job_id}")])
        lines.append("Пропуск оставляет уже опубликованные сообщения в Telegram.")
    rows += [[("Сохранённые ответы", "bank")], [("Проверить вопрос", "probe")]]
    await edit_menu(bot, chat_id, menu_id, "\n\n".join(lines), controls(*rows))


@router.callback_query(F.data.startswith("faq:"))
async def navigate(
    callback: CallbackQuery,
    bot: Bot,
    chats: ChatRepository,
    solutions: SolutionRepository,
    faq: FaqRepository,
    faq_sessions: FaqSessions,
    faq_publisher: FaqPublisher,
    bank_sessions: BankSessions | None = None,
    settings_sessions: SettingsSessions | None = None,
):
    message = callback.message
    if (
        message is None
        or isinstance(message, InaccessibleMessage)
        or message.chat.type != ChatType.SUPERGROUP
    ):
        await callback.answer(
            "FAQ загружается администратором в супергруппе.", show_alert=True
        )
        return
    chat_id, user_id, menu_id = (
        message.chat.id,
        callback.from_user.id,
        message.message_id,
    )
    try:
        if not await is_chat_admin(bot, chat_id, user_id):
            await callback.answer(
                "FAQ доступен администратору группы.", show_alert=True
            )
            return
        parts = (callback.data or "").split(":")[1:]
        if bank_sessions is not None:
            bank_sessions.discard_menu(chat_id, menu_id)
        if settings_sessions is not None:
            settings_sessions.discard_previews(chat_id, menu_id)
            for key, pending in list(settings_sessions.inputs.items()):
                if (pending.chat_id, pending.menu_id) == (chat_id, menu_id):
                    settings_sessions.inputs.pop(key, None)
        action = parts[0]
        if action in {"bank", "probe"}:
            faq_sessions.discard_menu(chat_id, menu_id)
            text = "Откройте нужный пункт настроек."
            markup = keyboard(
                [("Сохранённые ответы", "bank:1"), ("Проверить вопрос", "question")]
            )
        elif action == "home":
            active = await faq.active_for_chat(chat_id)
            if active:
                if active.user_id == user_id:
                    await job_report(bot, faq, active.id, chat_id, user_id, menu_id)
                else:
                    await edit_menu(
                        bot,
                        chat_id,
                        menu_id,
                        "В группе уже есть незавершённая загрузка. Продолжить её "
                        "может создавший её администратор.",
                        controls(
                            [
                                (
                                    "Отменить незавершённую загрузку",
                                    f"abandon:{active.id}",
                                )
                            ]
                        ),
                    )
                await callback.answer()
                return
            draft = faq_sessions.create(
                chat_id,
                user_id,
                menu_id,
                message.message_thread_id if message.message_thread_id != 1 else None,
            )
            text = (
                "<b>Начальный FAQ</b>\n\nДобавьте одну пару или загрузите CSV. "
                f"Место публикации: {target(draft.thread_id)}.\n"
                "Для другой темы откройте /settings непосредственно в ней. "
                "Тема должна быть доступна боту. "
                "Вопросы и ответы публикуются только после подтверждения."
            )
            rows = [
                [("Добавить вопрос и ответ", f"question:{draft.token}")],
                [("Загрузить FAQ из CSV", f"file:{draft.token}")],
                [("Скачать шаблон CSV", f"template:{draft.token}")],
            ]
            if message.message_thread_id not in {None, 1}:
                rows.append([("Публиковать в основном чате", f"main:{draft.token}")])
            rows.append([("Начать без FAQ", f"cancel:{draft.token}")])
            markup = controls(*rows)
        elif action == "abandon" and len(parts) == 2:
            active = await faq.active_for_chat(chat_id)
            if active is None or active.id != parts[1]:
                raise ValueError("Загрузка уже завершена или отменена.")
            draft = faq_sessions.create(chat_id, user_id, menu_id, None)
            draft.kind, draft.question = "abandon", active.id
            text = (
                "Отменить незавершённую загрузку этой группы? "
                "Уже сохранённые пары и опубликованные сообщения останутся. "
                "Дальнейшая публикация остановится после текущей отправки."
            )
            markup = controls(
                [("Подтвердить отмену", f"abandonyes:{draft.token}")],
                [("Назад", "home")],
            )
        elif action in {"pause", "resume", "retry", "skip"} and len(parts) == 2:
            job = await faq.get(parts[1], chat_id, user_id)
            if job is None or job.status not in {"paused", "running"}:
                raise ValueError("Загрузка недоступна или уже завершена.")
            if action == "pause":
                await faq.state(job.id, "paused")
            elif chat_id in faq_publisher.active:
                await callback.answer("Загрузка уже выполняется.", show_alert=True)
                return
            else:
                item = next(
                    (
                        i
                        for i in await faq.items(job.id)
                        if i.status not in {"done", "skipped"}
                    ),
                    None,
                )
                if action == "retry" and item and item.status.startswith("sending_"):
                    await faq.update(
                        job.id,
                        item.row_number,
                        status="pending"
                        if item.status == "sending_question"
                        else "question",
                        error_type=None,
                    )
                elif action == "skip" and item:
                    await faq.update(
                        job.id,
                        item.row_number,
                        status="skipped",
                        question_vector=None,
                        answer_vector=None,
                    )
                await callback.answer("Продолжаю публикацию.")
                await faq_publisher.run(
                    job.id,
                    chat_id,
                    user_id,
                    lambda: job_report(bot, faq, job.id, chat_id, user_id, menu_id),
                )
                return
            await job_report(bot, faq, job.id, chat_id, user_id, menu_id)
            await callback.answer()
            return
        else:
            draft = (
                faq_sessions.get(parts[1], chat_id, user_id, menu_id)
                if len(parts) >= 2
                else None
            )
            if draft is None:
                raise ValueError(
                    "Запрос устарел или открыт другим администратором. "
                    "Откройте FAQ заново."
                )
            if action == "cancel":
                faq_sessions.remove(draft.token)
                text, markup = "Добавление FAQ отменено. Банк не изменён.", controls()
            elif action == "abandonyes" and draft.kind == "abandon":
                active = await faq.active_for_chat(chat_id)
                if active is None or active.id != draft.question:
                    raise ValueError("Загрузка уже завершена или отменена.")
                await faq.state(active.id, "canceled")
                faq_sessions.remove(draft.token)
                text, markup = (
                    "Загрузка отменена. Уже сохранённые пары остаются в банке.",
                    controls(),
                )
            elif action == "main":
                draft.thread_id = None
                text, markup = (
                    f"Место: {target(None)}. Выберите способ добавления.",
                    controls(
                        [("Добавить пару", f"question:{draft.token}")],
                        [("Загрузить CSV", f"file:{draft.token}")],
                    ),
                )
            elif action == "template":
                await bot.send_document(
                    chat_id,
                    BufferedInputFile(TEMPLATE, filename="FAQ_шаблон.csv"),
                    message_thread_id=draft.thread_id,
                    caption="Замените пример настоящими вопросами и ответами. "
                    "Сохраните файл в UTF-8; затем нажмите «Загрузить FAQ из CSV».",
                )
                await callback.answer()
                return
            elif action in {"question", "answer", "file"}:
                if action == "answer" and not draft.question:
                    raise ValueError("Сначала введите вопрос.")
                await request(bot, draft, action)
                text, markup = (
                    "Ответьте на специальный запрос бота. "
                    "До подтверждения банк не изменится.",
                    controls([("Отмена", f"cancel:{draft.token}")]),
                )
            elif action == "preview":
                text, markup = preview(draft, int(parts[2]))
            elif action in {"pair", "issues"}:
                if action == "pair":
                    pair = draft.review.pairs[int(parts[2])]
                    body, part = (
                        f"Вопрос:\n{pair.question}\n\nОтвет:\n{pair.answer}",
                        int(parts[3]),
                    )
                    prefix = f"pair:{draft.token}:{parts[2]}"
                else:
                    body, part = "\n".join(draft.review.issues), int(parts[2])
                    prefix = f"issues:{draft.token}"
                chunks = [body[i : i + 1400] for i in range(0, len(body), 1400)] or [
                    "Ошибок нет."
                ]
                if not 0 <= part < len(chunks):
                    raise ValueError("Такая часть недоступна.")
                text = f"<b>Просмотр FAQ</b> — {part + 1}/{len(chunks)}\n\n" + escape(
                    chunks[part]
                )
                rows = []
                if part:
                    rows.append([("Назад", f"{prefix}:{part - 1}")])
                if part < len(chunks) - 1:
                    rows.append([("Далее", f"{prefix}:{part + 1}")])
                rows.append([("К подтверждению", f"preview:{draft.token}:0")])
                markup = controls(*rows)
            elif action == "save":
                if not draft.review.pairs:
                    raise ValueError("Нет новых корректных пар для загрузки.")
                faq_sessions.remove(draft.token)
                await chats.ensure_chat(chat_id, message.chat.title or "")
                job_id = await faq.create(
                    chat_id, user_id, draft.thread_id, draft.review.pairs
                )
                if job_id is None:
                    raise ValueError(
                        "В этой группе уже выполняется загрузка. Откройте FAQ заново."
                    )
                await callback.answer("Начинаю публикацию подтверждённых пар.")
                await faq_publisher.run(
                    job_id,
                    chat_id,
                    user_id,
                    lambda: job_report(bot, faq, job_id, chat_id, user_id, menu_id),
                )
                return
            else:
                raise ValueError("Кнопка устарела. Откройте FAQ заново.")
        await edit_menu(bot, chat_id, menu_id, text, markup)
        await callback.answer()
    except ValueError as error:
        await callback.answer(str(error), show_alert=True)
    except Exception as error:
        log_operation_error("faq_callback", chat_id, menu_id, error)
        await callback.answer(
            "Действие не выполнено. Откройте FAQ заново через /settings.",
            show_alert=True,
        )


async def draft_for(message, bot, sessions):
    if (
        message.chat.type != ChatType.SUPERGROUP
        or message.from_user is None
        or message.from_user.is_bot
    ):
        raise SkipHandler
    draft = sessions.input(message.chat.id, message.reply_to_message.message_id)
    if draft is None:
        reply = message.reply_to_message
        if (
            reply.from_user
            and reply.from_user.id == bot.id
            and (reply.text or "").startswith(PROMPT)
        ):
            await message.reply(
                "Запрос FAQ устарел или отменён. Откройте его заново через /settings."
            )
            return None
        raise SkipHandler
    if draft.user_id != message.from_user.id or not await is_chat_admin(
        bot, message.chat.id, message.from_user.id
    ):
        await message.reply("Ответить может только открывший запрос администратор.")
        return None
    if (
        sessions.get(draft.token, draft.chat_id, draft.user_id, draft.menu_id)
        is not draft
    ):
        await message.reply("Запрос FAQ отменён или устарел. Откройте его заново.")
        return None
    return draft


@router.message(F.reply_to_message, F.text, ~F.text.startswith("/"))
async def receive_text(
    message: Message, bot: Bot, solutions: SolutionRepository, faq_sessions: FaqSessions
):
    try:
        draft = await draft_for(message, bot, faq_sessions)
        if draft is None:
            return
        if draft.kind not in {"question", "answer"}:
            await message.reply("Прикрепите CSV в ответ на запрос файла.")
            return
        if not valid_text(message.text):
            await message.reply(
                "Нужен непустой текст до 3800 знаков UTF-16. Ответьте ещё раз."
            )
            return
        setattr(draft, draft.kind, message.text.strip())
        draft.prompt_id, draft.kind = None, ""
        if not draft.answer:
            await request(bot, draft, "answer")
            return
        draft.review = review(
            [FaqPair(1, draft.question, draft.answer)],
            await solutions.list_for_chat(message.chat.id),
        )
        text, markup = preview(draft)
        await edit_menu(bot, draft.chat_id, draft.menu_id, text, markup)
    except SkipHandler:
        raise
    except Exception as error:
        log_operation_error("faq_input", message.chat.id, message.message_id, error)
        await message.reply("Не удалось проверить пару. Откройте FAQ заново.")


@router.message(F.reply_to_message, F.document)
async def receive_file(
    message: Message, bot: Bot, solutions: SolutionRepository, faq_sessions: FaqSessions
):
    try:
        draft = await draft_for(message, bot, faq_sessions)
        if draft is None:
            return
        doc = message.document
        if draft.kind != "file":
            await message.reply("На этот запрос нужен текст вопроса или ответа.")
            return
        if (
            not (doc.file_name or "").casefold().endswith(".csv")
            or (doc.file_size or 0) > MAX_FILE
        ):
            raise ValueError("Прикрепите CSV до 512 КиБ.")
        buffer = LimitedBuffer()
        await bot.download(doc, destination=buffer)
        pairs, issues = parse_csv(buffer.getvalue())
        if (
            faq_sessions.get(draft.token, draft.chat_id, draft.user_id, draft.menu_id)
            is not draft
        ):
            await message.reply("Запрос отменён или устарел. Файл не загружен.")
            return
        draft.review = review(
            pairs, await solutions.list_for_chat(message.chat.id), issues
        )
        draft.prompt_id, draft.kind = None, ""
        draft.question, draft.answer = "", ""
        text, markup = preview(draft)
        await edit_menu(bot, draft.chat_id, draft.menu_id, text, markup)
    except SkipHandler:
        raise
    except ValueError as error:
        await message.reply(str(error))
    except Exception as error:
        log_operation_error("faq_csv", message.chat.id, message.message_id, error)
        await message.reply(
            "Не удалось проверить файл. Прикрепите его повторно к запросу."
        )


@router.message(Command("faqsource"))
async def reconcile(message: Message, bot: Bot, faq: FaqRepository):
    try:
        await reconcile_source(message, bot, faq)
    except Exception as error:
        log_operation_error("faq_source", message.chat.id, message.message_id, error)
        await message.reply("Не удалось связать источник. Повторите /faqsource.")


async def reconcile_source(message: Message, bot: Bot, faq: FaqRepository):
    if message.chat.type != ChatType.SUPERGROUP or message.from_user is None:
        return
    if not await is_chat_admin(bot, message.chat.id, message.from_user.id):
        await message.reply("Связывать FAQ может администратор группы.")
        return
    job = await faq.active_for_chat(message.chat.id)
    item = (
        next(
            (i for i in await faq.items(job.id) if i.status.startswith("sending_")),
            None,
        )
        if job
        else None
    )
    reply = message.reply_to_message
    if (
        job is None
        or job.status != "paused"
        or job.user_id != message.from_user.id
        or item is None
        or reply is None
        or reply.from_user is None
        or reply.from_user.id != bot.id
    ):
        await message.reply(
            "При неопределённой отправке ответьте /faqsource "
            "на опубликованный ботом вопрос или ответ."
        )
        return
    field = item.status.removeprefix("sending_")
    thread = reply.message_thread_id if reply.message_thread_id != 1 else None
    wrong_parent = field == "answer" and (
        reply.reply_to_message is None
        or reply.reply_to_message.message_id != item.question_message_id
    )
    if (
        reply.text != publication_text(item, field)
        or thread != job.thread_id
        or wrong_parent
    ):
        await message.reply(
            "Это другой текст или другая тема. Выберите нужное сообщение FAQ."
        )
        return
    await faq.update(
        job.id,
        item.row_number,
        **{
            field + "_message_id": reply.message_id,
            "status": field,
            "error_type": None,
        },
    )
    await message.reply(
        "Источник связан. Откройте FAQ через /settings и продолжите загрузку."
    )
