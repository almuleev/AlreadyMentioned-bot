"""Settings use temporary SQLite and fake Telegram/model clients."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText
from aiogram.types import Chat, Message, Update, User

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers import settings as ui
from already_mentioned.handlers.messages import handle_text
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.reply_cache import ReplyCache
from already_mentioned.services.search import SearchService
from already_mentioned.services.settings_sessions import SettingsSessions


def message(text="/settings", *, chat_id=-1001, user_id=7, message_id=50):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type=ChatType.SUPERGROUP, title="Тест"),
        from_user=SimpleNamespace(id=user_id, is_bot=False),
        message_id=message_id,
        text=text,
        message_thread_id=None,
        reply_to_message=None,
        answer=AsyncMock(),
        reply=AsyncMock(),
        get_url=lambda: f"https://t.me/c/1/{message_id}",
    )


def callback(action, **kwargs):
    msg = message(**kwargs)
    return SimpleNamespace(
        message=msg, from_user=msg.from_user, data=f"cfg:{action}", answer=AsyncMock()
    )


def client():
    return SimpleNamespace(
        id=42,
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.ADMINISTRATOR)
        ),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=80)),
        edit_message_text=AsyncMock(),
    )


class Encoder:
    def __init__(self):
        self.embed_query = AsyncMock(
            return_value=np.array([1.0, 0.0], dtype=np.float32)
        )
        self.embed_passage = AsyncMock(
            return_value=np.array([1.0, 0.0], dtype=np.float32)
        )


class Scorer:
    def predict(self, question, passages):
        return [0.8] * len(passages)


@pytest_asyncio.fixture
async def state(tmp_path):
    db = await connect_database(tmp_path / "settings.db")
    await initialize_database(db)
    chats, bank = ChatRepository(db), SolutionRepository(db)
    await chats.ensure_chat(-1001, "Тест")
    await chats.ensure_chat(-1002, "Другая группа")
    encoder = Encoder()
    search = SearchService(encoder, chats, bank, mode="hybrid", reranker=Scorer())
    yield SimpleNamespace(
        db=db,
        chats=chats,
        bank=bank,
        encoder=encoder,
        search=search,
        bot=client(),
        sessions=SettingsSessions(),
    )
    await db.close()


async def add(state, chat_id=-1001, index=1, text="Как войти в кабинет?"):
    vector = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    return await state.bank.add_solution(
        chat_id=chat_id,
        question_message_id=index * 2,
        answer_message_id=index * 2 + 1,
        question_text=text,
        answer_text="Откройте <кабинет> & войдите.",
        question_embedding=vector,
        answer_embedding=vector,
        question_link="https://t.me/c/1/2",
        answer_link="https://t.me/c/1/3",
    )


async def navigate(state, action, **kwargs):
    cb = callback(action, **kwargs)
    await ui.navigate(
        cb, state.bot, state.chats, state.bank, state.search, state.sessions
    )
    return cb


def rendered(state):
    return state.bot.edit_message_text.await_args.args[0]


def apply_action(state):
    return (
        state.bot.edit_message_text.await_args.kwargs["reply_markup"]
        .inline_keyboard[0][0]
        .callback_data.removeprefix("cfg:")
    )


async def receive(state, msg):
    await ui.receive_input(
        msg, state.bot, state.chats, state.bank, state.search, state.sessions
    )


@pytest.mark.asyncio
async def test_open_menu_and_permission_checks_precede_bank_access(state):
    msg = message()
    await ui.settings_command(msg, state.bot, state.chats, state.bank)
    assert "Настройки этой группы" in msg.answer.await_args.args[0]
    rows = msg.answer.await_args.kwargs["reply_markup"].inline_keyboard
    assert any(b.callback_data == "cfg:question" for row in rows for b in row)
    state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
    await navigate(state, "bank:1", chat_id=-1999)
    state.bot.edit_message_text.assert_not_awaited()
    assert await state.chats.get_chat(-1999) is None
    msg.chat.type = ChatType.PRIVATE
    state.bot.get_chat_member.reset_mock()
    await ui.settings_command(msg, state.bot, state.chats, state.bank)
    state.bot.get_chat_member.assert_not_awaited()
    assert "в нужной группе" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_pause_skips_inference_but_keeps_silent_autosave_and_other_chat(state):
    await add(state, -1002)
    await navigate(state, "set:0")
    await navigate(state, "set:0")  # Repeated delivery must not toggle back on.
    assert await state.search.find_best(-1001, "Как войти?") is None
    state.encoder.embed_query.assert_not_awaited()
    question = message("Как войти в кабинет?", message_id=10)
    answer = message("Откройте страницу входа.", message_id=11)
    answer.reply_to_message = question
    await handle_text(
        answer,
        state.bot,
        ReplyCache(),
        state.search,
        state.chats,
        state.bank,
        state.encoder,
        None,
    )
    assert await state.bank.count_for_chat(-1001) == 1
    answer.answer.assert_not_awaited()
    answer.reply.assert_not_awaited()
    assert not (await state.chats.get_chat(-1001)).suggestions_enabled
    assert await state.search.find_best(-1002, "Как войти?") is not None
    await navigate(state, "set:1")
    assert await state.search.find_best(-1001, "Как войти?") is not None


@pytest.mark.asyncio
async def test_persistent_pause_refresh_forget_and_restart(state):
    await state.chats.set_threshold(-1001, 0.7)
    await state.chats.set_threshold(-1001, 0.6, hybrid=True)
    await state.chats.set_suggestions_enabled(-1001, False)
    await state.chats.ensure_chat(-1001, "Новое название")
    await add(state)
    await state.chats.forget_chat_data(-1001)
    async with state.db.execute("PRAGMA database_list") as cursor:
        path = (await cursor.fetchone())[2]
    reopened = await connect_database(path)
    try:
        await initialize_database(reopened)
        chat = await ChatRepository(reopened).get_chat(-1001)
        assert (
            chat.title,
            chat.similarity_threshold,
            chat.hybrid_threshold,
            chat.suggestions_enabled,
        ) == ("Новое название", 0.7, 0.6, False)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_preview_requires_owner_confirmation_and_never_changes_other_mode(state):
    await add(state)
    await navigate(state, "preview:more")
    assert "не проверка качества" in rendered(state)
    assert "Ближайший ответ" in rendered(state)
    assert "&lt;кабинет&gt; &amp;" in rendered(state)
    assert (await state.chats.get_chat(-1001)).hybrid_threshold is None
    action = apply_action(state)
    for kwargs in ({"user_id": 8}, {"chat_id": -1002}, {"message_id": 51}):
        cb = await navigate(state, action, **kwargs)
        assert cb.answer.await_args.kwargs["show_alert"]
        assert (await state.chats.get_chat(-1001)).hybrid_threshold is None
    await navigate(state, action)
    chat = await state.chats.get_chat(-1001)
    assert chat.hybrid_threshold == 0.321
    assert chat.similarity_threshold == 0.458
    assert (await state.chats.get_chat(-1002)).hybrid_threshold is None
    cb = await navigate(state, action)
    assert "устарел" in cb.answer.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", ["change", "navigate", "expire", "restart", "mode"])
async def test_stale_preview_cannot_overwrite_settings(state, stale):
    now = [0.0]
    state.sessions = SettingsSessions(clock=lambda: now[0])
    await navigate(state, "preview:less")
    action = apply_action(state)
    if stale == "change":
        await state.chats.set_threshold(-1001, 0.9, hybrid=True)
    elif stale == "navigate":
        await navigate(state, "home")
    elif stale == "expire":
        now[0] = 301
    elif stale == "restart":
        state.sessions = SettingsSessions()
    else:
        state.search.mode = "baseline"
    cb = await navigate(state, action)
    assert "устарел" in cb.answer.await_args.args[0]
    assert (await state.chats.get_chat(-1001)).hybrid_threshold == (
        0.9 if stale == "change" else None
    )


@pytest.mark.asyncio
async def test_custom_input_rechecks_rights_validates_and_previews(state):
    await navigate(state, "threshold")
    await navigate(state, "threshold")
    state.bot.send_message.assert_awaited_once()
    assert state.bot.send_message.await_args.kwargs["reply_markup"].selective
    msg = message("0,35")
    msg.reply_to_message = message(message_id=80)
    msg.from_user.id = 8
    await receive(state, msg)
    assert "только открывший" in msg.reply.await_args.args[0]
    msg.from_user.id = 7
    state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
    await receive(state, msg)
    state.bot.edit_message_text.assert_not_awaited()
    state.bot.get_chat_member.return_value.status = ChatMemberStatus.ADMINISTRATOR
    for invalid in ("nan", "inf", "-1", "1.01", "не число"):
        msg.text = invalid
        await receive(state, msg)
        assert "число от 0 до 1" in msg.reply.await_args.args[0]
        assert state.sessions.get_input(-1001, 80) is not None
    msg.text = "0,35"
    await receive(state, msg)
    assert "0.371 → 0.350" in rendered(state)
    assert state.sessions.get_input(-1001, 80) is None
    assert (await state.chats.get_chat(-1001)).hybrid_threshold is None
    await navigate(state, apply_action(state))
    assert (await state.chats.get_chat(-1001)).hybrid_threshold == 0.35


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pause,threshold,query,reason",
    [
        (False, 0.3, "Как войти в кабинет?", "предложил бы"),
        (True, 0.3, "Как войти в кабинет?", "подсказки на паузе"),
        (False, 0.99, "Как войти в кабинет?", "не достиг порога"),
        (False, 0.3, "Спасибо за помощь", "не прошло фильтр"),
    ],
)
async def test_question_check_explains_real_decision_without_writes(
    state, pause, threshold, query, reason
):
    await add(state)
    await add(state, -1002, text="ЧУЖОЙ ВОПРОС")
    await state.chats.set_suggestions_enabled(-1001, not pause)
    await state.chats.set_threshold(-1001, threshold, hybrid=True)
    before = state.db.total_changes
    await navigate(state, "question")
    msg = message(query)
    msg.reply_to_message = message(message_id=80)
    await receive(state, msg)
    assert reason in rendered(state)
    assert "ЧУЖОЙ" not in rendered(state)
    assert "&lt;кабинет&gt;" in rendered(state)
    assert state.db.total_changes == before
    assert await state.bank.count_for_chat(-1001) == 1


@pytest.mark.asyncio
async def test_expired_input_is_not_searched_but_ordinary_reply_falls_through(state):
    msg = message("Как войти?")
    msg.reply_to_message = message("Проверка вопроса\nАдминистратор", user_id=42)
    await receive(state, msg)
    assert "устарел" in msg.reply.await_args.args[0]
    state.encoder.embed_query.assert_not_awaited()
    msg.reply_to_message = message("Обычный вопрос", user_id=5)
    with pytest.raises(SkipHandler):
        await receive(state, msg)


@pytest.mark.asyncio
async def test_bank_pages_are_scoped_bounded_and_escape_html(state):
    for index in range(4):
        await add(state, index=index, text="<вопрос>" + "А" * 600)
    await add(state, -1002, text="ЧУЖОЙ ВОПРОС")
    await navigate(state, "bank:1")
    text = rendered(state)
    assert "1/2, всего 4" in text and "ЧУЖОЙ" not in text
    assert "&lt;вопрос&gt;" in text and len(text) < 4096
    await navigate(state, "bank:2")
    assert "2/2" in rendered(state)
    await navigate(state, "bank:999")
    assert "устарела" in rendered(state)


@pytest.mark.asyncio
async def test_changed_setting_is_reported_when_menu_edit_fails_and_errors_are_safe(
    state, caplog
):
    state.bot.edit_message_text.side_effect = TelegramBadRequest(
        EditMessageText(text="x", chat_id=-1001, message_id=50), "SECRET"
    )
    cb = await navigate(state, "set:0")
    assert not (await state.chats.get_chat(-1001)).suggestions_enabled
    assert "Настройка сохранена" in cb.answer.await_args.args[0]
    assert "SECRET" not in caplog.text


def test_sessions_are_bounded_expire_and_isolate_owners():
    now = [0.0]
    sessions = SettingsSessions(clock=lambda: now[0], capacity=2, ttl=5)
    for i in range(3):
        sessions.add_input(-1, i, 7, 50, "question")
    assert sessions.get_input(-1, 0) is None
    assert sessions.get_input(-2, 1) is None
    token = sessions.preview(-1, 7, 50, "hybrid", None, 0.4)
    assert sessions.take_preview(token, -1, 8, 50) is None
    assert sessions.take_preview(token, -1, 7, 50).proposed == 0.4
    now[0] = 5
    assert sessions.get_input(-1, 2) is None


@pytest.mark.asyncio
async def test_baseline_confirmation_preserves_hybrid_override(state):
    state.search.mode = "baseline"
    await state.chats.set_threshold(-1001, .65, hybrid=True)
    await navigate(state, "preview:more")
    assert "0.458 → 0.408" in rendered(state)
    await navigate(state, apply_action(state))
    chat = await state.chats.get_chat(-1001)
    assert (chat.similarity_threshold, chat.hybrid_threshold) == (.408, .65)


@pytest.mark.asyncio
async def test_aiogram_routing_passes_normal_replies_and_commands_to_next_handler(
    state,
):
    # Real router filters + SkipHandler must leave autosave and commands reachable.
    dispatcher = Dispatcher()
    settings_router, fallback = Router(), Router()
    for handler in ui.router.message.handlers:
        settings_router.message.register(
            handler.callback, *[item.callback for item in handler.filters]
        )
    seen = []

    @fallback.message()
    async def next_handler(message):
        seen.append(message.text)

    dispatcher.include_routers(settings_router, fallback)
    bot = Bot("42:TEST_TOKEN_FOR_LOCAL_TESTS")
    try:
        base = dict(
            date=datetime.now(UTC),
            chat=Chat(id=-1001, type="supergroup"),
            from_user=User(id=7, is_bot=False, first_name="Тест"),
        )
        reply = Message(message_id=1, text="Как войти?", **base)
        for i, text in enumerate(("Откройте кабинет.", "/solve"), 2):
            msg = Message(message_id=i, text=text, reply_to_message=reply, **base)
            await dispatcher.feed_update(
                bot,
                Update(update_id=i, message=msg),
                chats=state.chats,
                solutions=state.bank,
                search=state.search,
                settings_sessions=state.sessions,
            )
        assert seen == ["Откройте кабинет.", "/solve"]
    finally:
        await bot.session.close()
