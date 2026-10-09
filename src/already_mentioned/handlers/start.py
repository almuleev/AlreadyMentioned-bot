"""Private introduction and a read-only, chat-scoped connection wizard."""

from aiogram import Bot, F, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    ChatMemberUpdated,
    InaccessibleMessage,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from already_mentioned.bot.permissions import ADMIN_STATUSES, is_chat_admin
from already_mentioned.error_logging import log_operation_error
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.welcome import WelcomeMessages

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
INTRO = (
    "AlreadyMentioned находит готовые ответы на повторные вопросы в вашей группе.\n\n"
    "Базу можно заполнить заранее — загрузите FAQ. Когда появится похожий вопрос, "
    "бот подскажет подходящий ответ. Новые ответы не сочиняет, "
    "старую переписку автоматически не загружает.\n\n"
    "Автоматически сохраняет ответы администраторов.\n\n"
    "Добавьте бота в группу, чтобы начать."
)
HOW = (
    "Как работает бот\n\n"
    "1. Участник задаёт вопрос в группе.\n"
    "2. Администратор отвечает на него через «Ответить» (reply). "
    "Подходящая пара сохраняется без служебного сообщения.\n"
    "3. Когда появляется похожий вопрос, бот показывает один сохранённый ответ "
    "со ссылкой на оригинал.\n\n"
    "Полезен и полный, и содержательный частичный ответ. "
    "Подсказка может ошибаться: 👍 или 👎 под ней может нажать автор вопроса."
)
EXAMPLE = (
    "Первый сохранённый ответ\n\n"
    "1. Попросите участника задать реальный повторяющийся вопрос, например: "
    "«Где найти материалы встречи?»\n"
    "2. Администратор нажимает «Ответить» на этот вопрос и пишет настоящую "
    "инструкцию. Нужны текстовые вопрос и ответ в супергруппе.\n"
    "3. Нажмите «Проверить подключение»: там видно число сохранённых ответов. "
    "Автосохранение не публикует подтверждения в группу.\n"
    "4. Задайте похожий вопрос отдельным сообщением. "
    "Бот предложит ответ, если найдёт подходящее совпадение.\n\n"
    "Если пара не сохранилась, администратор может ответить командой /solve "
    "на своё сообщение с ответом в течение 6 часов."
)
HELP = (
    "Помощь с подключением\n\n"
    "• Используйте супергруппу: в обычной группе нет нужных ссылок на сообщения.\n"
    "• Разрешите боту отправлять сообщения.\n"
    "• Для обычного бота владелец отключает Privacy Mode в @BotFather: "
    "/setprivacy → выбрать бота → Disable, затем удаляет и добавляет бота заново. "
    "Бот-администратор получает обычные сообщения и при включённом Privacy Mode.\n"
    "• Ответ администратора должен быть reply на текстовый вопрос.\n\n"
    "/start — открыть подключение; /status — состояние и число ответов; "
    "/help — все команды. Управление подключением в группе доступно "
    "администраторам. Если бот совсем не отвечает, сначала проверьте запрет "
    "на отправку сообщений в правах группы."
)


def button(text, action):
    return InlineKeyboardButton(text=text, callback_data=f"onboard:{action}")


def group_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [button("Проверить подключение", "check")],
            [InlineKeyboardButton(text="Настройки", callback_data="cfg:home")],
            [button("Первый ответ", "example")],
            [
                InlineKeyboardButton(
                    text="Добавить / загрузить FAQ", callback_data="faq:home"
                )
            ],
            [button("Как работает", "how"), button("Помощь", "help")],
        ]
    )


async def private_menu(bot):
    me = await bot.get_me()
    rows = []
    text = INTRO
    if me.username and me.can_join_groups is not False:
        rows.append(
            [
                InlineKeyboardButton(
                    text="Добавить в группу",
                    url=f"https://t.me/{me.username}?startgroup=connect",
                )
            ]
        )
    else:
        text += (
            "\n\nДобавление в группы сейчас недоступно. Владелец бота может "
            "включить его в @BotFather: /setjoingroups → выбрать бота → Enable. "
            "После изменения откройте /start заново."
        )
    rows.append([button("Как работает", "how"), button("Помощь", "help")])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


async def connection(bot, chat, solutions):
    """Read fresh Telegram permissions; never change settings or save examples."""
    me = await bot.get_me()
    member = await bot.get_chat_member(chat.id, me.id)
    full_chat = await bot.get_chat(chat.id)
    admin = member.status in ADMIN_STATUSES
    present = is_present(member)
    if member.status == ChatMemberStatus.RESTRICTED:
        can_send = present and member.can_send_messages
    elif admin:
        can_send = True
    elif present:
        can_send = (
            full_chat.permissions.can_send_messages
            if full_chat.permissions is not None
            else None
        )
    else:
        can_send = False
    can_read = present and (admin or me.can_read_all_group_messages is True)
    issues = []
    if chat.type != ChatType.SUPERGROUP:
        issues.append(
            "Нужна супергруппа: в обычной группе нет ссылок на ответы. "
            "Преобразуйте её в супергруппу или используйте существующую."
        )
    if not present:
        issues.append("Добавьте бота в эту группу заново.")
    elif not can_read:
        issues.append(
            "Получение обычных сообщений выключено. Владелец бота: "
            "@BotFather → /setprivacy → выбрать бота → Disable. "
            "Затем удалите и добавьте бота в группу заново."
        )
    if can_send is False:
        issues.append("Разрешите боту отправлять текстовые сообщения в правах группы.")
    elif can_send is None:
        issues.append(
            "Telegram не сообщил права отправки. Проверьте разрешение "
            "на текстовые сообщения в настройках группы."
        )
    count = await solutions.count_for_chat(chat.id)
    title = "Подключение готово" if not issues else "Что нужно исправить"
    lines = [title, *issues, f"Сохранённых ответов в этом чате: {count}."]
    if not issues:
        lines.append(
            "Нажмите «Первый ответ», чтобы сохранить пару вопрос–ответ."
            if not count
            else "Задайте похожий вопрос отдельным сообщением, "
            "чтобы проверить подсказку."
        )
        if not admin:
            lines.append(
                "Если Privacy Mode меняли после добавления бота, "
                "удалите и добавьте его заново. Реальную доставку "
                "сообщений проверьте первым вопросом и ответом."
            )
    return "\n\n".join(lines), can_send


async def welcome(bot, chat, solutions, send):
    text, can_send = await connection(bot, chat, solutions)
    if can_send is not False:
        await send(
            "Я помогу находить сохранённые ответы в этой группе.\n\n" + text,
            reply_markup=group_menu(),
            parse_mode=None,
        )
        return True
    return False


@router.message(CommandStart())
async def start(
    message: Message,
    bot: Bot,
    command: CommandObject,
    solutions: SolutionRepository,
    welcome_messages: WelcomeMessages,
):
    try:
        if message.chat.type == ChatType.PRIVATE:
            text, keyboard = await private_menu(bot)
            await message.answer(text, reply_markup=keyboard, parse_mode=None)
        elif message.chat.type in GROUP_TYPES:
            if message.from_user is None or not await is_chat_admin(
                bot, message.chat.id, message.from_user.id
            ):
                await message.answer(
                    "Подключение в группе открывает администратор. "
                    "Справка для участников: /help."
                )
                return

            async def send():
                return await welcome(bot, message.chat, solutions, message.answer)

            if command.args == "connect":
                await welcome_messages.send_once(message.chat.id, send)
            else:
                await send()
    except TelegramAPIError as error:
        log_operation_error(
            "onboarding_start", message.chat.id, message.message_id, error
        )
        if message.chat.type == ChatType.PRIVATE:
            await message.answer(
                INTRO + "\n\nНе удалось получить ссылку добавления. "
                "Повторите /start чуть позже.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [button("Как работает", "how"), button("Помощь", "help")]
                    ]
                ),
                parse_mode=None,
            )


def is_present(member):
    return (
        member.is_member
        if member.status == ChatMemberStatus.RESTRICTED
        else member.status not in {ChatMemberStatus.LEFT, ChatMemberStatus.KICKED}
    )


@router.my_chat_member()
async def joined(
    event: ChatMemberUpdated,
    bot: Bot,
    solutions: SolutionRepository,
    welcome_messages: WelcomeMessages,
):
    if (
        event.chat.type not in GROUP_TYPES
        or is_present(event.old_chat_member)
        or not is_present(event.new_chat_member)
    ):
        return

    async def send():
        async def post(text, **kwargs):
            await bot.send_message(event.chat.id, text, **kwargs)

        return await welcome(bot, event.chat, solutions, post)

    try:
        await welcome_messages.send_once(event.chat.id, send)
    except TelegramAPIError as error:
        log_operation_error("onboarding_join", event.chat.id, 0, error)


@router.callback_query(F.data.startswith("onboard:"))
async def navigate(callback: CallbackQuery, bot: Bot, solutions: SolutionRepository):
    message = callback.message
    action = (callback.data or "").removeprefix("onboard:")
    if (
        message is None
        or isinstance(message, InaccessibleMessage)
        or action not in {"home", "check", "example", "how", "help"}
    ):
        await callback.answer(
            "Кнопка устарела. Откройте /start заново.", show_alert=True
        )
        return
    private = message.chat.type == ChatType.PRIVATE
    if not private and message.chat.type not in GROUP_TYPES:
        await callback.answer(
            "Подключите бота к группе через личный /start.", show_alert=True
        )
        return
    try:
        if not private and not await is_chat_admin(
            bot, message.chat.id, callback.from_user.id
        ):
            await callback.answer(
                "Подключение доступно администратору группы.", show_alert=True
            )
            return
        if private and action in {"check", "example"}:
            await callback.answer("Откройте /start в нужной группе.", show_alert=True)
            return
        if action in {"home", "check"}:
            if private:
                text, keyboard = await private_menu(bot)
            else:
                text, can_send = await connection(bot, message.chat, solutions)
                if can_send is False:
                    await callback.answer(
                        "Бот не может писать в группу. "
                        "Проверьте его права и добавьте заново, "
                        "если он был удалён.",
                        show_alert=True,
                    )
                    return
                keyboard = group_menu()
        else:
            text = {"how": HOW, "help": HELP, "example": EXAMPLE}[action]
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        button(
                            "Назад" if private else "Проверить подключение",
                            "home" if private else "check",
                        )
                    ]
                ]
            )
        if message.text != text or message.reply_markup != keyboard:
            await message.edit_text(text, reply_markup=keyboard, parse_mode=None)
        await callback.answer()
    except TelegramAPIError as error:
        log_operation_error(
            "onboarding_callback", message.chat.id, message.message_id, error
        )
        await callback.answer(
            "Не удалось проверить подключение. Попробуйте ещё раз или откройте /start.",
            show_alert=True,
        )
