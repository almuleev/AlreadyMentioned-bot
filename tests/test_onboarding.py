import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from aiogram.enums import ChatMemberStatus as Status
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandObject
from aiogram.methods import GetMe

from already_mentioned.handlers import start as ui
from already_mentioned.handlers.messages import handle_text
from already_mentioned.services.reply_cache import ReplyCache
from already_mentioned.services.welcome import WelcomeMessages


def message(kind=ChatType.SUPERGROUP, chat_id=-1001):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type=kind),
        from_user=SimpleNamespace(id=7),
        message_id=50,
        text="/start",
        reply_markup=None,
        answer=AsyncMock(),
        edit_text=AsyncMock(),
    )


def bot(*, privacy=False, status=Status.MEMBER, can_send=True, admin=True):
    client = SimpleNamespace(
        get_me=AsyncMock(
            return_value=SimpleNamespace(
                id=42,
                username="AlreadyMentionedTestBot",
                can_join_groups=True,
                can_read_all_group_messages=not privacy,
            )
        ),
        get_chat=AsyncMock(
            return_value=SimpleNamespace(
                permissions=SimpleNamespace(can_send_messages=can_send)
            )
        ),
        send_message=AsyncMock(),
        admin=admin,
        member=SimpleNamespace(
            status=status, is_member=True, can_send_messages=can_send
        ),
    )

    async def member(chat_id, user_id):
        return (
            client.member
            if user_id == 42
            else SimpleNamespace(
                status=Status.ADMINISTRATOR if client.admin else Status.MEMBER
            )
        )

    client.get_chat_member = AsyncMock(side_effect=member)
    return client


def callback(action="check", **kwargs):
    return SimpleNamespace(
        message=message(**kwargs),
        data=f"onboard:{action}",
        from_user=SimpleNamespace(id=7),
        answer=AsyncMock(),
    )


def solutions(count=0):
    return SimpleNamespace(count_for_chat=AsyncMock(return_value=count))


@pytest.mark.asyncio
async def test_private_start_has_real_add_link_and_no_group_data():
    msg, client, bank = message(ChatType.PRIVATE, 7), bot(), solutions()
    await ui.start(msg, client, CommandObject(command="start"), bank, WelcomeMessages())
    rows = msg.answer.await_args.kwargs["reply_markup"].inline_keyboard
    assert rows[0][0].url == "https://t.me/AlreadyMentionedTestBot?startgroup=connect"
    assert [b.text for b in rows[1]] == ["Как работает", "Помощь"]
    bank.count_for_chat.assert_not_awaited()
    client.get_chat_member.assert_not_awaited()
    client.get_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_disabled_groups_and_network_error_have_actionable_help():
    client, bank = bot(), solutions()
    client.get_me.return_value.can_join_groups = False
    msg = message(ChatType.PRIVATE, 7)
    await ui.start(msg, client, CommandObject(command="start"), bank, WelcomeMessages())
    assert "/setjoingroups" in msg.answer.await_args.args[0]
    assert all(
        b.url is None
        for row in msg.answer.await_args.kwargs["reply_markup"].inline_keyboard
        for b in row
    )
    client.get_me.side_effect = TelegramBadRequest(GetMe(), "temporary failure")
    msg = message(ChatType.PRIVATE, 7)
    await ui.start(msg, client, CommandObject(command="start"), bank, WelcomeMessages())
    assert "Повторите /start" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,privacy,status,can_send,expected",
    [
        (ChatType.GROUP, False, Status.MEMBER, True, "Нужна супергруппа"),
        (ChatType.SUPERGROUP, True, Status.MEMBER, True, "/setprivacy"),
        (ChatType.SUPERGROUP, True, Status.ADMINISTRATOR, False, "Подключение готово"),
        (ChatType.SUPERGROUP, False, Status.RESTRICTED, False, "Разрешите боту"),
        (ChatType.SUPERGROUP, False, Status.MEMBER, None, "не сообщил права"),
    ],
)
async def test_connection_reports_real_permissions(
    kind, privacy, status, can_send, expected
):
    client = bot(privacy=privacy, status=status, can_send=can_send)
    text, allowed = await ui.connection(client, message(kind).chat, solutions(3))
    assert expected in text
    assert "этом чате: 3" in text
    if status == Status.RESTRICTED:
        assert allowed is False


@pytest.mark.asyncio
async def test_navigation_rechecks_admin_and_never_leaks_bank_to_nonadmin():
    client, bank = bot(), solutions(3)
    cb = callback()
    await ui.navigate(cb, client, bank)
    assert "Подключение готово" in cb.message.edit_text.await_args.args[0]
    bank.count_for_chat.assert_awaited_once_with(-1001)
    client.admin = False
    bank.count_for_chat.reset_mock()
    cb.message.edit_text.reset_mock()
    await ui.navigate(cb, client, bank)
    assert "администратору" in cb.answer.await_args.args[0]
    cb.message.edit_text.assert_not_awaited()
    bank.count_for_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_checks_are_fresh_and_duplicate_edits_are_skipped():
    client, bank, cb = bot(), solutions(), callback()
    await ui.navigate(cb, client, bank)
    cb.message.text = cb.message.edit_text.await_args.args[0]
    cb.message.reply_markup = cb.message.edit_text.await_args.kwargs["reply_markup"]
    cb.message.edit_text.reset_mock()
    await ui.navigate(cb, client, bank)
    cb.message.edit_text.assert_not_awaited()
    client.get_me.return_value.can_read_all_group_messages = False
    await ui.navigate(cb, client, bank)
    assert "/setprivacy" in cb.message.edit_text.await_args.args[0]


@pytest.mark.asyncio
async def test_blocked_bot_uses_alert_and_stale_private_callbacks_are_safe():
    cb = callback()
    await ui.navigate(cb, bot(status=Status.RESTRICTED, can_send=False), solutions())
    cb.message.edit_text.assert_not_awaited()
    assert "не может писать" in cb.answer.await_args.args[0]
    for action in ("check", "example", "old"):
        cb = callback(action, kind=ChatType.PRIVATE, chat_id=7)
        bank = solutions()
        await ui.navigate(cb, bot(), bank)
        assert cb.answer.await_args.kwargs["show_alert"]
        cb.message.edit_text.assert_not_awaited()
        bank.count_for_chat.assert_not_awaited()
    cb = callback()
    cb.message = None
    await ui.navigate(cb, bot(), solutions())
    assert "устарела" in cb.answer.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("join_first", (True, False))
async def test_join_and_deep_link_produce_one_greeting_in_either_order(join_first):
    client, bank, state, msg = bot(), solutions(), WelcomeMessages(), message()
    event = SimpleNamespace(
        chat=msg.chat,
        old_chat_member=SimpleNamespace(status=Status.LEFT),
        new_chat_member=SimpleNamespace(status=Status.MEMBER),
    )
    calls = [
        ui.joined(event, client, bank, state),
        ui.start(
            msg, client, CommandObject(command="start", args="connect"), bank, state
        ),
    ]
    await asyncio.gather(*(calls if join_first else reversed(calls)))
    assert client.send_message.await_count + msg.answer.await_count == 1
    # A later explicit /start remains available even in the suppression window.
    await ui.start(msg, client, CommandObject(command="start"), bank, state)
    assert client.send_message.await_count + msg.answer.await_count == 2
    event.old_chat_member = event.new_chat_member
    await ui.joined(event, client, bank, state)
    assert client.send_message.await_count + msg.answer.await_count == 2


@pytest.mark.asyncio
async def test_unsent_greeting_is_retryable_and_cache_is_bounded():
    now = [0.0]
    state = WelcomeMessages(clock=lambda: now[0], capacity=2)
    send = AsyncMock(return_value=False)
    await state.send_once(-1, send)
    send.return_value = True
    await state.send_once(-1, send)
    await state.send_once(-1, send)
    assert send.await_count == 2
    await state.send_once(-2, send)
    await state.send_once(-3, send)
    assert len(state.sent) == 2
    now[0] = 61
    await state.send_once(-3, send)
    assert len(state.sent) == 1


@pytest.mark.asyncio
async def test_wizard_keeps_real_chat_settings_data_and_scope(tmp_path):
    from already_mentioned.database.connection import connect_database
    from already_mentioned.database.schema import initialize_database
    from already_mentioned.repositories.chats import ChatRepository
    from already_mentioned.repositories.solutions import SolutionRepository

    db = await connect_database(tmp_path / "onboarding.db")
    try:
        await initialize_database(db)
        chats, bank = ChatRepository(db), SolutionRepository(db)
        for chat_id in (-1001, -1002):
            await chats.ensure_chat(chat_id, "Тест")
            await chats.set_threshold(chat_id, 0.7)
            await chats.set_threshold(chat_id, 0.4, hybrid=True)
        before = await chats.get_chat(-1001)
        text, _ = await ui.connection(bot(), message().chat, bank)
        assert "этом чате: 0" in text
        question = SimpleNamespace(
            text="Где найти вход в кабинет?",
            message_id=1,
            from_user=None,
            get_url=lambda: "https://t.me/c/1/1",
        )
        answer = message()
        answer.text, answer.message_id = "Ссылка на вход в описании группы.", 2
        answer.chat.title = "Тест"
        answer.from_user.is_bot = False
        answer.reply_to_message = question
        answer.get_url = lambda: "https://t.me/c/1/2"
        answer.reply = AsyncMock()
        search = SimpleNamespace(find_best=AsyncMock())
        await handle_text(
            answer,
            bot(),
            ReplyCache(),
            search,
            chats,
            bank,
            SimpleNamespace(
                embed_passage=AsyncMock(
                    return_value=np.array([1.0, 0.0], dtype=np.float32)
                )
            ),
            SimpleNamespace(schedule=AsyncMock()),
        )
        answer.answer.assert_not_awaited()
        answer.reply.assert_not_awaited()
        search.find_best.assert_not_awaited()
        for cb in (callback(), callback("example"), callback(chat_id=-1002)):
            await ui.navigate(cb, bot(), bank)
        assert "этом чате: 0" in cb.message.edit_text.await_args.args[0]
        text, _ = await ui.connection(bot(), message().chat, bank)
        assert "этом чате: 1" in text
        assert await chats.get_chat(-1001) == before
        assert await bank.count_for_chat(-1001) == 1
        assert await bank.count_for_chat(-1002) == 0
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,heading",
    [
        ("how", "Как работает бот"),
        ("help", "Помощь с подключением"),
        ("home", "AlreadyMentioned"),
    ],
)
async def test_private_menu_navigation_and_back(action, heading):
    cb, client, bank = (
        callback(action, kind=ChatType.PRIVATE, chat_id=7),
        bot(),
        solutions(),
    )
    await ui.navigate(cb, client, bank)
    assert cb.message.edit_text.await_args.args[0].startswith(heading)
    bank.count_for_chat.assert_not_awaited()
    client.get_chat_member.assert_not_awaited()
    cb.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_nonadmin_start_and_permission_api_failure_are_closed():
    msg, client, bank = message(), bot(admin=False), solutions()
    await ui.start(msg, client, CommandObject(command="start"), bank, WelcomeMessages())
    assert "администратор" in msg.answer.await_args.args[0]
    bank.count_for_chat.assert_not_awaited()
    client.get_me.assert_not_awaited()
    cb = callback()
    client.get_chat_member.side_effect = TelegramBadRequest(GetMe(), "private details")
    await ui.navigate(cb, client, bank)
    assert "Не удалось" in cb.answer.await_args.args[0]
    cb.message.edit_text.assert_not_awaited()
    bank.count_for_chat.assert_not_awaited()
