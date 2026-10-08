"""Administrator bank flows use temporary storage and fake inference/Telegram."""

import asyncio
from datetime import UTC, datetime
from html import unescape
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText
from aiogram.types import Chat, Message, Update, User

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers import bank as ui
from already_mentioned.handlers import settings
from already_mentioned.models.entities import SolutionRevision
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.feedback import FeedbackRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.bank_sessions import BankSessions
from already_mentioned.services.reply_cache import PendingAnswer, ReplyCache
from already_mentioned.services.settings_sessions import SettingsSessions


def message(text="Новый текст", *, chat_id=-1001, user_id=7, message_id=50):
    return SimpleNamespace(
        text=text,
        message_id=message_id,
        message_thread_id=2,
        chat=SimpleNamespace(id=chat_id, type=ChatType.SUPERGROUP, title="Тест"),
        from_user=SimpleNamespace(id=user_id, is_bot=False),
        reply_to_message=None,
        reply=AsyncMock(),
        answer=AsyncMock(),
    )


@pytest_asyncio.fixture
async def state(tmp_path):
    db = await connect_database(tmp_path / "bank.db")
    await initialize_database(db)
    chats, bank, votes = (
        ChatRepository(db),
        SolutionRepository(db),
        FeedbackRepository(db),
    )
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    bot = SimpleNamespace(
        id=42,
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.ADMINISTRATOR)
        ),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=80)),
        edit_message_text=AsyncMock(),
    )
    yield SimpleNamespace(
        db=db,
        chats=chats,
        bank=bank,
        votes=votes,
        bot=bot,
        sessions=BankSessions(),
        cache=ReplyCache(),
        embeddings=SimpleNamespace(
            embed_passage=AsyncMock(return_value=np.array([0.0, 1.0], dtype=np.float32))
        ),
    )
    await db.close()


async def add(
    s,
    *,
    chat_id=-1001,
    index=1,
    question="Как оплатить?",
    answer="Откройте <оплату> & войдите.",
):
    vector = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    return await s.bank.add_solution(
        chat_id=chat_id,
        question_message_id=index * 2,
        answer_message_id=index * 2 + 1,
        question_text=question,
        answer_text=answer,
        question_embedding=vector,
        answer_embedding=vector,
        question_link=f"https://t.me/c/1/{index * 2}",
        answer_link=f"https://t.me/c/1/{index * 2 + 1}",
    )


async def navigate(s, action, **kwargs):
    msg = message(**kwargs)
    cb = SimpleNamespace(
        message=msg, from_user=msg.from_user, data=f"bank:{action}", answer=AsyncMock()
    )
    await ui.navigate(cb, s.bot, s.bank, s.sessions, s.cache)
    return cb


async def receive(s, text, **kwargs):
    msg = message(text, **kwargs)
    msg.reply_to_message = message(ui.PROMPTS["question"], user_id=42, message_id=80)
    await ui.receive_input(msg, s.bot, s.bank, s.embeddings, s.sessions)
    return msg


def rendered(s):
    return s.bot.edit_message_text.await_args.args[0]


def actions(s):
    return [
        b.callback_data.removeprefix("bank:")
        for row in s.bot.edit_message_text.await_args.kwargs[
            "reply_markup"
        ].inline_keyboard
        for b in row
    ]


@pytest.mark.asyncio
async def test_search_is_literal_unicode_case_insensitive_and_chat_scoped(state):
    first = await add(state, question="Где ОПЛАТА?", answer="Скидка 50%_тест")
    await add(state, index=2, answer="ОПЛАТА через кабинет")
    await add(state, chat_id=-1002, question="ОПЛАТА в чужой группе")
    for query, count in (
        ("оплата", 2),
        ("ОпЛаТа", 2),
        ("50%_тест", 1),
        ("%", 1),
        ("' OR 1=1 --", 0),
        ("несуществующее", 0),
    ):
        assert await state.bank.count_for_chat(-1001, query=query) == count
        rows = await state.bank.page_for_chat(-1001, limit=3, offset=0, query=query)
        assert len(rows) == count
    await state.votes.add_feedback(
        solution_id=first, query_message_id=1, user_id=7, vote="helpful"
    )
    row = (await state.bank.page_for_chat(-1001, limit=1, offset=0, query="50%"))[0]
    assert row.helpful == 1
    assert await state.bank.summary_for_chat(-1002, first) is None


@pytest.mark.asyncio
async def test_search_prompt_pages_cards_back_and_owner_scoping(state):
    for index in range(4):
        await add(state, index=index, question=f"Где оплата {index}?")
    await add(state, chat_id=-1002, question="ЧУЖОЙ ВОПРОС")
    before = state.db.total_changes
    await navigate(state, "search")
    await navigate(state, "search")
    state.bot.send_message.assert_awaited_once()
    assert state.bot.send_message.await_args.kwargs["message_thread_id"] == 2
    await receive(state, "ОПЛАТА")
    assert "1/2, всего 4" in rendered(state) and "ЧУЖОЙ" not in rendered(state)
    next_page = next(
        a for a in actions(state) if a.startswith("list:") and a.endswith(":2")
    )
    cb = await navigate(state, next_page, user_id=8)
    assert "другим администратором" in cb.answer.await_args.args[0]
    await navigate(state, next_page)
    assert "2/2" in rendered(state)
    card = next(a for a in actions(state) if a.startswith("card:"))
    await navigate(state, card)
    assert "Исходный вопрос" in rendered(state) and "Исходный ответ" in rendered(state)
    back = next(a for a in actions(state) if a.startswith("list:"))
    await navigate(state, back)
    assert "Поиск: ОПЛАТА" in rendered(state) and "2/2" in rendered(state)
    assert state.db.total_changes == before


@pytest.mark.asyncio
async def test_long_card_can_be_read_fully_with_safe_html_and_callback_lengths(state):
    question, answer = "<вопрос>&😀" * 280, "<ответ>&😀" * 400
    sid = await add(state, question=question, answer=answer)
    chunks, part = [], 0
    while True:
        await navigate(state, f"card:{sid}:-:1:{part}")
        text = rendered(state)
        chunks.append(unescape(text.split("<blockquote>")[1].split("</blockquote>")[0]))
        assert len(unescape(text).encode("utf-16-le")) // 2 < 4096
        assert all(len(a.encode()) + 5 <= 64 for a in actions(state))
        markup = state.bot.edit_message_text.await_args.kwargs["reply_markup"]
        if not any(
            b.text == "Следующая часть" for row in markup.inline_keyboard for b in row
        ):
            break
        part += 1
    assert "".join(chunks) == f"Вопрос:\n{question}\n\nОтвет:\n{answer}"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["question", "answer"])
async def test_edit_changes_only_selected_text_and_document_vector(state, field):
    sid = await add(state)
    await state.votes.add_feedback(
        solution_id=sid, query_message_id=3, user_id=7, vote="helpful"
    )
    before = await state.bank.get_for_chat(-1001, sid)
    await navigate(state, f"{field}:{sid}:-:1")
    await navigate(state, f"{field}:{sid}:-:1")
    state.bot.send_message.assert_awaited_once()
    await receive(state, " Новый <текст> & инструкция ")
    after = await state.bank.get_for_chat(-1001, sid)
    other = "answer" if field == "question" else "question"
    assert getattr(after, field + "_text") == "Новый <текст> & инструкция"
    assert getattr(after, other + "_text") == getattr(before, other + "_text")
    assert getattr(after, other + "_embedding") == getattr(before, other + "_embedding")
    assert getattr(after, field + "_embedding") != getattr(before, field + "_embedding")
    assert (after.id, after.question_link, after.answer_link) == (
        before.id,
        before.question_link,
        before.answer_link,
    )
    assert (await state.bank.summary_for_chat(-1001, sid)).helpful == 1
    state.embeddings.embed_passage.assert_awaited_once_with(
        "Новый <текст> & инструкция"
    )
    assert "Изменение сохранено" in rendered(state)
    await receive(state, "Повторная доставка")
    state.embeddings.embed_passage.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked", ["other_user", "revoked", "wrong_chat", "empty", "cancel"]
)
async def test_edit_input_rejects_wrong_context_and_keeps_bank(state, blocked):
    sid = await add(state)
    before = await state.bank.get_for_chat(-1001, sid)
    await navigate(state, f"answer:{sid}:-:1")
    kwargs = {}
    if blocked == "other_user":
        kwargs["user_id"] = 8
    elif blocked == "revoked":
        state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
    elif blocked == "wrong_chat":
        kwargs["chat_id"] = -1002
    elif blocked == "cancel":
        await navigate(state, f"card:{sid}:-:1:0")
    msg = await receive(state, "   " if blocked == "empty" else "Правка", **kwargs)
    msg.reply.assert_awaited_once()
    state.embeddings.embed_passage.assert_not_awaited()
    assert await state.bank.get_for_chat(-1001, sid) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancel", "edit", "delete", "revoke", "duplicate"])
async def test_inflight_edit_cannot_undo_concurrent_changes_or_cancellation(
    state, change
):
    sid = await add(state)
    before = await state.bank.get_for_chat(-1001, sid)
    await navigate(state, f"answer:{sid}:-:1")
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed(text):
        started.set()
        await release.wait()
        return np.array([0.0, 1.0], dtype=np.float32)

    state.embeddings.embed_passage.side_effect = delayed
    task = asyncio.create_task(receive(state, "Первый ввод"))
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        if change == "cancel":
            await navigate(state, f"card:{sid}:-:1:0")
        elif change == "edit":
            await state.bank.update_question(
                -1001, sid, before.answer_message_id, "Чужая правка", b"new"
            )
        elif change == "delete":
            await state.bank.delete_revision(SolutionRevision.of(before))
        elif change == "revoke":
            state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
        else:
            await receive(state, "Повторный ввод")
            state.embeddings.embed_passage.assert_awaited_once()
    finally:
        release.set()
    await asyncio.wait_for(task, timeout=2)
    after = await state.bank.get_for_chat(-1001, sid)
    if change == "delete":
        assert after is None
    elif change == "duplicate":
        assert after.answer_text == "Первый ввод"
    else:
        assert after.answer_text == before.answer_text


@pytest.mark.asyncio
async def test_delete_is_single_pair_confirmed_scoped_and_cascades_votes(state):
    sid = await add(state)
    other = await add(state, index=2)
    foreign = await add(state, chat_id=-1002)
    for value in (sid, other, foreign):
        await state.votes.add_feedback(
            solution_id=value, query_message_id=5, user_id=7, vote="helpful"
        )
    state.cache.remember(
        PendingAnswer(-1001, 3, 7, 2, "Как оплатить?", "q", state.cache.clock())
    )
    await navigate(state, f"delete:{sid}:-:1")
    assert "Удалить пару" in rendered(state)
    assert await state.bank.count_for_chat(-1001) == 2
    action = next(a for a in actions(state) if a.startswith("confirm:"))
    for kwargs in ({"user_id": 8}, {"chat_id": -1002}, {"message_id": 51}):
        cb = await navigate(state, action, **kwargs)
        assert cb.answer.await_args.kwargs["show_alert"]
    await navigate(state, action)
    assert await state.bank.get_for_chat(-1001, sid) is None
    assert await state.bank.get_for_chat(-1001, other) is not None
    assert len(await state.votes.list_for_chat(-1001)) == 1
    assert len(await state.votes.list_for_chat(-1002)) == 1
    assert state.cache.get(-1001, 3) is None
    cb = await navigate(state, action)
    assert "устарело" in cb.answer.await_args.args[0]
    new_id = await add(state)
    assert new_id > sid  # Resaving works; an old card never targets the replacement.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale", ["cancel", "navigate", "change", "expire", "restart", "revoke", "settings"]
)
async def test_delete_rejects_stale_previews(state, stale):
    sid = await add(state)
    now = [0.0]
    state.sessions = BankSessions(clock=lambda: now[0])
    await navigate(state, f"delete:{sid}:-:1")
    action = next(a for a in actions(state) if a.startswith("confirm:"))
    if stale == "cancel":
        await navigate(
            state, next(a for a in actions(state) if a.startswith("cancel:"))
        )
    elif stale == "navigate":
        await navigate(state, "list:-:1")
    elif stale == "change":
        await state.bank.update_answer(-1001, sid, 3, "Новый ответ", b"new")
    elif stale == "expire":
        now[0] = 301
    elif stale == "restart":
        state.sessions = BankSessions()
    elif stale == "settings":
        cb = SimpleNamespace(
            message=message(),
            from_user=message().from_user,
            data="cfg:home",
            answer=AsyncMock(),
        )
        await settings.navigate(
            cb,
            state.bot,
            state.chats,
            state.bank,
            None,
            SettingsSessions(),
            state.sessions,
        )
    else:
        state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
    await navigate(state, action)
    assert await state.bank.get_for_chat(-1001, sid) is not None


@pytest.mark.asyncio
async def test_nonadmin_foreign_ids_and_malformed_buttons_expose_no_data(state):
    sid = await add(state, chat_id=-1002, question="СЕКРЕТ ДРУГОЙ ГРУППЫ")
    await navigate(state, f"card:{sid}:-:1:0")
    state.bot.edit_message_text.assert_not_awaited()
    for action in (
        f"question:{sid}:-:1",
        f"delete:{sid}:-:1",
        "card:bad:-:1:0",
        "list:-:0",
    ):
        await navigate(state, action)
        assert "СЕКРЕТ" not in str(state.bot.edit_message_text.await_args)
    state.bot.edit_message_text.reset_mock()
    state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
    await navigate(state, "search")
    state.bot.send_message.assert_not_awaited()
    state.bot.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_failure_and_post_commit_telegram_failure_are_reported_safely(
    state, caplog
):
    sid = await add(state)
    await navigate(state, f"answer:{sid}:-:1")
    before = await state.bank.get_for_chat(-1001, sid)
    state.embeddings.embed_passage.side_effect = RuntimeError("SECRET_TOKEN")
    msg = await receive(state, "Правка")
    assert "не выполнено" in msg.reply.await_args.args[0]
    assert await state.bank.get_for_chat(-1001, sid) == before
    state.embeddings.embed_passage.side_effect = None
    await navigate(state, f"answer:{sid}:-:1")
    state.bot.edit_message_text.side_effect = TelegramBadRequest(
        EditMessageText(text="x", chat_id=-1001, message_id=50), "SECRET_TOKEN"
    )
    msg = await receive(state, "Успешная правка")
    assert "Изменение сохранено" in msg.reply.await_args.args[0]
    assert (await state.bank.get_for_chat(-1001, sid)).answer_text == "Успешная правка"
    assert "SECRET_TOKEN" not in caplog.text


def test_bank_sessions_expire_bound_each_collection_and_claim_once():
    now = [0.0]
    sessions = BankSessions(clock=lambda: now[0], ttl=5, capacity=2)
    for index in range(3):
        sessions.add_input(index, -1, 7, 50, "search")
        sessions.query(-1, 7, 50, str(index))
        sessions.deletion(-1, 7, 50, None, "-", 1)
    assert all(
        len(c) == 2 for c in (sessions.inputs, sessions.views, sessions.deletions)
    )
    assert sessions.input(-1, 0) is None
    assert sessions.claim_input(1, -1, 8, 50) is None
    request = sessions.claim_input(1, -1, 7, 50)
    assert request is not None and sessions.claim_input(1, -1, 7, 50) is None
    sessions.discard_menu(-1, 50)
    assert not sessions.input_is_active(1, request) and not sessions.processing
    token = sessions.query(-1, 7, 50, "оплата")
    now[0] = 5
    assert sessions.view(token, -1, 7, 50) is None


@pytest.mark.asyncio
async def test_real_aiogram_routes_bank_input_without_autosave_and_keeps_other_replies(
    state,
):
    sid = await add(state)
    await navigate(state, f"answer:{sid}:-:1")
    dispatcher, seen = Dispatcher(), []
    for source in (ui.router, settings.router):
        clone = Router()
        for handler in source.message.handlers:
            clone.message.register(
                handler.callback, *[f.callback for f in handler.filters]
            )
        dispatcher.include_router(clone)
    fallback = Router()

    @fallback.message()
    async def normal(message):
        seen.append(message.text)

    dispatcher.include_router(fallback)
    bot = Bot("42:LOCAL_TEST_TOKEN")
    bot.get_chat_member = state.bot.get_chat_member
    bot.edit_message_text = state.bot.edit_message_text
    try:
        base = dict(
            date=datetime.now(UTC),
            chat=Chat(id=-1001, type="supergroup"),
            from_user=User(id=7, is_bot=False, first_name="Тест"),
        )
        original = Message(message_id=80, text=ui.PROMPTS["answer"], **base)
        for index, text in enumerate(("Правка из reply", "Обычный ответ", "/solve")):
            if index:
                original = original.model_copy(update={"message_id": 9})
            msg = Message(
                message_id=100 + index, text=text, reply_to_message=original, **base
            )
            await dispatcher.feed_update(
                bot,
                Update(update_id=index, message=msg),
                solutions=state.bank,
                embeddings=state.embeddings,
                bank_sessions=state.sessions,
                chats=state.chats,
                search=None,
                settings_sessions=SettingsSessions(),
            )
        assert seen == ["Обычный ответ", "/solve"]
        assert (
            await state.bank.get_for_chat(-1001, sid)
        ).answer_text == "Правка из reply"
    finally:
        await bot.session.close()
