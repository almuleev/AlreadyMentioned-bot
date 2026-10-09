"""FAQ import uses temporary FRIDA-contract SQLite and fake Telegram/model IO."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from already_mentioned.database.connection import connect_database
from already_mentioned.database.embedding_contract import require_frida
from already_mentioned.database.schema import initialize_database
from already_mentioned.handlers import faq as ui
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.faq import FaqRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.faq_data import (
    MAX_FILE,
    TEMPLATE,
    FaqPair,
    LimitedBuffer,
    parse_csv,
    review,
)
from already_mentioned.services.faq_publisher import FaqPublisher
from already_mentioned.services.faq_sessions import FaqSessions
from already_mentioned.services.settings_sessions import SettingsSessions


def message(text="", *, mid=50, user=7, chat=-1001, thread=10):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat, type=ChatType.SUPERGROUP, title="Группа"),
        from_user=SimpleNamespace(id=user, is_bot=False),
        message_id=mid,
        message_thread_id=thread,
        text=text,
        document=None,
        reply_to_message=None,
        reply=AsyncMock(),
        answer=AsyncMock(),
    )


@pytest_asyncio.fixture
async def state(tmp_path):
    db = await connect_database(tmp_path / "faq.db")
    await initialize_database(db)
    await require_frida(db, adopt_empty=True)
    chats, solutions, repo = (
        ChatRepository(db),
        SolutionRepository(db),
        FaqRepository(db),
    )
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    sent = []

    async def send(chat_id, text, **kwargs):
        result = message(
            text,
            mid=len(sent) + 100,
            user=42,
            chat=chat_id,
            thread=kwargs.get("message_thread_id"),
        )
        sent.append(result)
        return result

    bot = SimpleNamespace(
        id=42,
        send_message=AsyncMock(side_effect=send),
        send_document=AsyncMock(),
        edit_message_text=AsyncMock(),
        download=AsyncMock(),
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.ADMINISTRATOR)
        ),
    )
    vector = np.zeros(1536, dtype=np.float32)
    vector[0] = 1
    embeddings = SimpleNamespace(embed_passage=AsyncMock(return_value=vector))
    publisher = FaqPublisher(bot, repo, solutions, embeddings, interval=0)
    yield SimpleNamespace(
        db=db,
        chats=chats,
        solutions=solutions,
        faq=repo,
        bot=bot,
        sessions=FaqSessions(),
        publisher=publisher,
        embeddings=embeddings,
        sent=sent,
    )
    await db.close()


async def nav(s, action="home", **kwargs):
    msg = message(**kwargs)
    cb = SimpleNamespace(
        message=msg, from_user=msg.from_user, data="faq:" + action, answer=AsyncMock()
    )
    await ui.navigate(cb, s.bot, s.chats, s.solutions, s.faq, s.sessions, s.publisher)
    return cb


def draft(s):
    return next(reversed(s.sessions.drafts.values()))


async def text_input(s, text, *, user=7):
    d = draft(s)
    msg = message(text, user=user)
    msg.reply_to_message = message(ui.PROMPT, mid=d.prompt_id, user=42)
    await ui.receive_text(msg, s.bot, s.solutions, s.sessions)
    return msg


def actions(s):
    return [
        b.callback_data
        for row in s.bot.edit_message_text.await_args.kwargs[
            "reply_markup"
        ].inline_keyboard
        for b in row
    ]


@pytest.mark.parametrize(
    "data",
    [
        TEMPLATE,
        'вопрос,ответ\n"Как?","Текст, с запятой"\n'.encode(),
        'ВОПРОС;ОТВЕТ\n"Первая\nвторая";Ответ\n'.encode("utf-8-sig"),
    ],
)
def test_csv_supported_encodings_headers_and_quoted_fields(data):
    pairs, issues = parse_csv(data)
    assert len(pairs) == 1 and not issues


@pytest.mark.parametrize(
    "data",
    [
        b"question;answer\nx;y",
        b"\xff",
        b"x" * (MAX_FILE + 1),
        ("вопрос;ответ\n" + "Вопрос;Ответ\n" * 51).encode(),
        'вопрос;ответ\n"Не закрыто;Ответ'.encode(),
    ],
    ids=["headers", "encoding", "size", "rows", "malformed"],
)
def test_csv_rejects_unsupported_or_over_limit_files(data):
    with pytest.raises(ValueError):
        parse_csv(data)


def test_csv_row_errors_literal_duplicate_review_and_bounded_download():
    pairs, issues = parse_csv(
        "вопрос;ответ\nКак?;Ответ\n;Пустой вопрос\nЛишний;текст;колонка\n".encode()
    )
    assert len(pairs) == 1 and len(issues) == 2
    result = review((*pairs, FaqPair(5, " КАК? ", "Ответ")), [])
    assert len(result.pairs) == 1 and result.duplicates == 1
    buffer = LimitedBuffer()
    with pytest.raises(ValueError):
        buffer.write(b"x" * (MAX_FILE + 1))


@pytest.mark.asyncio
async def test_manual_draft_requires_confirmation_scope_and_keeps_topic(state):
    await nav(state)
    token = draft(state).token
    await nav(state, "question:" + token)
    await text_input(state, "Как войти?")
    await text_input(state, "Откройте <кабинет> & войдите.")
    assert await state.solutions.count_for_chat(-1001) == 0
    assert state.embeddings.embed_passage.await_count == 0
    for kwargs in ({"user": 8}, {"chat": -1002}, {"mid": 51}):
        cb = await nav(state, "save:" + token, **kwargs)
        assert cb.answer.await_args.kwargs["show_alert"]
    await asyncio.gather(nav(state, "save:" + token), nav(state, "save:" + token))
    rows = await state.solutions.list_for_chat(-1001)
    assert len(rows) == 1 and rows[0].answer_text == "Откройте <кабинет> & войдите."
    assert rows[0].question_link.startswith("https://t.me/c/1/")
    publications = [
        m
        for m in state.sent
        if m.text.startswith("FAQ · Вопрос") or m.text.startswith("FAQ · Ответ")
    ]
    assert len(publications) == 2 and all(
        m.message_thread_id == 10 for m in publications
    )
    state.embeddings.embed_passage.assert_any_await("Как войти?")
    assert await state.solutions.count_for_chat(-1002) == 0


@pytest.mark.asyncio
async def test_csv_preview_errors_duplicates_and_confirmed_only_publication(state):
    await nav(state)
    d = draft(state)
    await nav(state, "file:" + d.token)
    data = (
        "вопрос;ответ\nКак?;Инструкция\nКАК?;инструкция\n;ошибка\nДругой?;Ответ"
    ).encode()

    async def download(document, destination):
        destination.write(data)

    state.bot.download.side_effect = download
    msg = message()
    msg.document = SimpleNamespace(file_name="FAQ.csv", file_size=len(data))
    msg.reply_to_message = message(ui.PROMPT, mid=d.prompt_id, user=42)
    await ui.receive_file(msg, state.bot, state.solutions, state.sessions)
    assert (
        len(d.review.pairs) == 2
        and d.review.duplicates == 1
        and len(d.review.issues) == 1
    )
    assert await state.solutions.count_for_chat(-1001) == 0
    assert any(a.startswith("faq:issues:") for a in actions(state))
    await nav(state, "save:" + d.token)
    assert await state.solutions.count_for_chat(-1001) == 2
    result = review(parse_csv(data)[0], await state.solutions.list_for_chat(-1001))
    assert not result.pairs and result.duplicates == 3


@pytest.mark.asyncio
async def test_nonadmin_template_target_choice_cancel_and_expiry(state):
    state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
    await nav(state)
    assert not state.sessions.drafts
    state.bot.get_chat_member.return_value.status = ChatMemberStatus.ADMINISTRATOR
    await nav(state)
    token = draft(state).token
    await nav(state, "template:" + token)
    state.bot.send_document.assert_awaited_once()
    await nav(state, "main:" + token)
    assert draft(state).thread_id is None
    await nav(state, "question:" + token)
    msg = await text_input(state, "Чужой ввод", user=8)
    assert "открывший" in msg.reply.await_args.args[0]
    await nav(state, "cancel:" + token)
    cb = await nav(state, "save:" + token)
    assert "устарел" in cb.answer.await_args.args[0]
    assert await state.solutions.count_for_chat(-1001) == 0


async def job(s, pairs=None):
    return await s.faq.create(
        -1001, 7, 10, pairs or [FaqPair(2, "Как войти?", "Откройте кабинет")]
    )


@pytest.mark.asyncio
async def test_definite_send_failure_resumes_only_missing_answer_and_waits_for_telegram(
    state,
):
    jid = await job(state)
    original = state.bot.send_message.side_effect

    async def send(chat_id, text, **kwargs):
        if text.startswith("FAQ · Ответ"):
            raise TelegramRetryAfter(
                SendMessage(chat_id=chat_id, text=text), "wait", 60
            )
        return await original(chat_id, text, **kwargs)

    state.bot.send_message.side_effect = send
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    row = (await state.faq.items(jid))[0]
    assert row.status == "question" and row.question_message_id is not None
    assert row.retry_at is not None
    state.bot.send_message.side_effect = original
    count = state.bot.send_message.await_count
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    assert state.bot.send_message.await_count == count
    await state.faq.update(jid, 2, retry_at=0)
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    assert state.bot.send_message.await_count == count + 1
    assert (await state.faq.get(jid, -1001, 7)).status == "done"
    assert await state.solutions.count_for_chat(-1001) == 1
    assert state.embeddings.embed_passage.await_count == 2


@pytest.mark.asyncio
async def test_unknown_send_result_never_resends_without_source_or_explicit_retry(
    state, caplog
):
    jid = await job(state)
    original = state.bot.send_message.side_effect
    published = []

    async def uncertain(chat_id, text, **kwargs):
        sent = await original(chat_id, text, **kwargs)
        if text.startswith("FAQ · Ответ"):
            published.append(sent)
            raise OSError("PRIVATE_TOKEN_AND_TEXT")
        return sent

    state.bot.send_message.side_effect = uncertain
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    assert (await state.faq.items(jid))[0].status == "sending_answer"
    count = state.bot.send_message.await_count
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    assert state.bot.send_message.await_count == count
    msg = message("/faqsource")
    msg.reply_to_message = published[0]
    published[0].reply_to_message = state.sent[0]
    await ui.reconcile(msg, state.bot, state.faq)
    assert "связан" in msg.reply.await_args.args[0]
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    assert state.bot.send_message.await_count == count
    assert await state.solutions.count_for_chat(-1001) == 1
    assert "PRIVATE_TOKEN_AND_TEXT" not in caplog.text


@pytest.mark.asyncio
async def test_partial_load_is_durable_and_saved_rows_are_skipped_after_reopen(state):
    jid = await job(
        state, [FaqPair(2, "Первый?", "Ответ 1"), FaqPair(3, "Второй?", "Ответ 2")]
    )
    original = state.bot.send_message.side_effect

    async def fail(chat_id, text, **kwargs):
        if "Второй?" in text:
            raise TelegramBadRequest(SendMessage(chat_id=chat_id, text=text), "invalid")
        return await original(chat_id, text, **kwargs)

    state.bot.send_message.side_effect = fail
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    assert await state.solutions.count_for_chat(-1001) == 1
    assert (await state.faq.items(jid))[0].status == "done"
    async with state.db.execute("PRAGMA database_list") as cursor:
        path = (await cursor.fetchone())[2]
    reopened = await connect_database(path)
    try:
        state.bot.send_message.side_effect = original
        repo = FaqRepository(reopened)
        await FaqPublisher(
            state.bot, repo, SolutionRepository(reopened), state.embeddings, interval=0
        ).run(jid, -1001, 7, AsyncMock())
        assert await state.solutions.count_for_chat(-1001) == 2
        assert len([m for m in state.sent if "Первый?" in m.text]) == 1
    finally:
        await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["pause", "forget", "revoke", "cancel"])
async def test_control_changes_during_encoding_block_publication(state, change):
    jid = await job(state)
    started, release = asyncio.Event(), asyncio.Event()
    original = state.embeddings.embed_passage.return_value

    async def delayed(text):
        started.set()
        await release.wait()
        return original

    state.embeddings.embed_passage.side_effect = delayed
    task = asyncio.create_task(state.publisher.run(jid, -1001, 7, AsyncMock()))
    await asyncio.wait_for(started.wait(), timeout=2)
    if change == "forget":
        await state.chats.forget_chat_data(-1001)
    elif change == "revoke":
        state.bot.get_chat_member.return_value.status = ChatMemberStatus.MEMBER
    else:
        await state.faq.state(jid, "paused" if change == "pause" else "canceled")
    release.set()
    await asyncio.wait_for(task, timeout=2)
    assert await state.solutions.count_for_chat(-1001) == 0
    state.bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_vectors_and_progress_failures_do_not_corrupt_bank(state):
    jid = await job(state)
    state.embeddings.embed_passage.return_value = np.array([1.0, 0.0], dtype=np.float32)
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    assert await state.solutions.count_for_chat(-1001) == 0
    state.bot.send_message.assert_not_awaited()
    vector = np.zeros(1536, dtype=np.float32)
    vector[0] = 1
    state.embeddings.embed_passage.return_value = vector
    await state.publisher.run(
        jid, -1001, 7, AsyncMock(side_effect=RuntimeError("SECRET"))
    )
    assert (await state.faq.get(jid, -1001, 7)).status == "done"
    assert await state.solutions.count_for_chat(-1001) == 1


@pytest.mark.asyncio
async def test_corrupt_cached_vectors_block_publication_before_sending(state):
    jid = await job(state)
    await state.faq.update(jid, 2, question_vector=b"invalid", answer_vector=b"invalid")
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    state.bot.send_message.assert_not_awaited()
    assert await state.solutions.count_for_chat(-1001) == 0


def test_drafts_are_bounded_scoped_and_expire():
    now = [0.0]
    sessions = FaqSessions(clock=lambda: now[0], capacity=2, ttl=5)
    first = sessions.create(-1, 7, 1, None)
    sessions.create(-1, 7, 2, None)
    sessions.create(-1, 7, 3, None)
    assert sessions.get(first.token, -1, 7, 1) is None
    last = next(reversed(sessions.drafts.values()))
    assert sessions.get(last.token, -2, 7, 3) is None
    now[0] = 5
    assert sessions.get(last.token, -1, 7, 3) is None


@pytest.mark.asyncio
async def test_other_admin_cancels_only_after_bound_preview_and_cannot_resume(state):
    jid = await job(state)
    await nav(state, user=8)
    assert await state.faq.active_for_chat(-1001) is not None
    await nav(state, "abandon:" + jid, user=8)
    token = draft(state).token
    for kwargs in ({"user": 7}, {"chat": -1002, "user": 8}, {"mid": 51, "user": 8}):
        cb = await nav(state, "abandonyes:" + token, **kwargs)
        assert cb.answer.await_args.kwargs["show_alert"]
    await nav(state, "abandonyes:" + token, user=8)
    assert (await state.faq.get(jid, -1001, 7)).status == "canceled"
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    state.bot.send_message.assert_not_awaited()
    assert not await state.faq.state(jid, "running")
    assert await job(state) is not None


@pytest.mark.asyncio
async def test_competing_confirmations_keep_one_complete_job_and_atomic_items(state):
    results = await asyncio.gather(job(state), job(state))
    assert sum(result is not None for result in results) == 1
    jid = next(result for result in results if result is not None)
    assert len(await state.faq.items(jid)) == 1
    await state.faq.state(jid, "canceled")
    assert (
        await state.faq.create(
            -1001, 7, None, [FaqPair(1, "Q", "A"), FaqPair(1, "Q2", "A2")]
        )
        is None
    )
    assert await state.faq.active_for_chat(-1001) is None


@pytest.mark.asyncio
async def test_unknown_question_requires_matching_source_or_explicit_repeat(state):
    jid = await job(state)
    original = state.bot.send_message.side_effect
    state.bot.send_message.side_effect = OSError("connection")
    await state.publisher.run(jid, -1001, 7, AsyncMock())
    item = (await state.faq.items(jid))[0]
    assert item.status == "sending_question"
    for reply in (
        message("FAQ · Вопрос\n" + item.question, mid=99, user=42, thread=11),
        message("Другой текст", mid=99, user=42),
        message("FAQ · Вопрос\n" + item.question, mid=99, user=7),
    ):
        msg = message("/faqsource")
        msg.reply_to_message = reply
        await ui.reconcile(msg, state.bot, state.faq)
        assert (await state.faq.items(jid))[0].question_message_id is None
    state.bot.send_message.side_effect = original
    await nav(state, "retry:" + jid)
    assert await state.solutions.count_for_chat(-1001) == 1
    assert state.bot.send_message.await_count == 3


@pytest.mark.asyncio
async def test_real_router_consumes_only_faq_input_and_navigation_cancels_settings(
    state,
):
    await nav(state)
    d = draft(state)
    await nav(state, "question:" + d.token)
    dispatcher, seen = Dispatcher(), []
    clone = Router()
    for handler in ui.router.message.handlers:
        clone.message.register(handler.callback, *[f.callback for f in handler.filters])
    dispatcher.include_router(clone)
    fallback = Router()

    @fallback.message()
    async def normal(message):
        seen.append(message.text)

    dispatcher.include_router(fallback)
    bot = Bot("42:LOCAL_TEST_TOKEN")
    bot.get_chat_member = state.bot.get_chat_member
    bot.send_message = state.bot.send_message
    bot.edit_message_text = state.bot.edit_message_text
    try:
        base = dict(
            date=datetime.now(UTC),
            chat=Chat(id=-1001, type="supergroup"),
            from_user=User(id=7, is_bot=False, first_name="Тест"),
        )
        prompt = Message(message_id=d.prompt_id, text=ui.PROMPT, **base)
        for index, text in enumerate(("Как добавить?", "Обычный ответ", "/solve")):
            reply = (
                prompt if index == 0 else prompt.model_copy(update={"message_id": 9})
            )
            msg = Message(
                message_id=200 + index, text=text, reply_to_message=reply, **base
            )
            await dispatcher.feed_update(
                bot,
                Update(update_id=index, message=msg),
                solutions=state.solutions,
                faq_sessions=state.sessions,
                faq=state.faq,
            )
        assert seen == ["Обычный ответ", "/solve"]
        assert draft(state).question == "Как добавить?"
        assert await state.solutions.count_for_chat(-1001) == 0
        settings = SettingsSessions()
        settings.add_input(-1001, 999, 7, 50, "threshold")
        cb = SimpleNamespace(
            message=message(),
            from_user=message().from_user,
            data="faq:home",
            answer=AsyncMock(),
        )
        await ui.navigate(
            cb,
            state.bot,
            state.chats,
            state.solutions,
            state.faq,
            state.sessions,
            state.publisher,
            settings_sessions=settings,
        )
        assert settings.get_input(-1001, 999) is None
    finally:
        await bot.session.close()
