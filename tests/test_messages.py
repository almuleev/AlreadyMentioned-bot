from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from aiogram.enums import ChatMemberStatus, ChatType, ParseMode

from already_mentioned.handlers.messages import format_answer_preview, handle_text
from already_mentioned.models.entities import Solution
from already_mentioned.services.reply_cache import ReplyCache
from already_mentioned.services.search import SearchMatch


def make_message(text: str, *, message_id: int = 50) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        message_id=message_id,
        chat=SimpleNamespace(id=-1001, type=ChatType.SUPERGROUP),
        from_user=SimpleNamespace(id=123, is_bot=False),
        reply_to_message=None,
        answer=AsyncMock(),
        reply=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_strong_question_gets_inline_quoted_reply_and_source_button() -> None:
    solution = Solution(
        id=7,
        chat_id=-1001,
        question_message_id=1,
        answer_message_id=2,
        question_text="Как войти?",
        answer_text="Откройте страницу входа.",
        question_embedding=b"vector",
        question_link="https://t.me/c/1/1",
        answer_link="https://t.me/c/1/2",
    )
    search = SimpleNamespace(
        find_best=AsyncMock(return_value=SearchMatch(solution, 0.95))
    )
    message = make_message("Где найти вход в личный кабинет?")
    await handle_text(
        message, SimpleNamespace(), ReplyCache(), search, None, None, None
    )

    search.find_best.assert_awaited_once_with(-1001, message.text)
    message.answer.assert_not_awaited()
    message.reply.assert_awaited_once()
    text = message.reply.await_args.args[0]
    assert "<blockquote>Откройте страницу входа.</blockquote>" in text
    assert message.reply.await_args.kwargs["parse_mode"] == ParseMode.HTML
    rows = message.reply.await_args.kwargs["reply_markup"].inline_keyboard
    buttons = rows[0]
    assert [button.text for button in buttons] == ["👍 Помогло", "👎 Не подходит"]
    assert [button.callback_data for button in buttons] == [
        "vote:7:50:helpful",
        "vote:7:50:not_helpful",
    ]
    assert rows[1][0].text == "📎 Оригинал"
    assert rows[1][0].url == "https://t.me/c/1/2"


def test_answer_preview_escapes_html_and_truncates_long_text() -> None:
    assert "&lt;b&gt;&amp;&lt;/b&gt;" in format_answer_preview("<b>&</b>")
    preview = format_answer_preview("A" * 4000)
    assert preview.count("A") == 3000
    assert "Ответ сокращён" in preview


@pytest.mark.asyncio
async def test_ordinary_text_and_weak_question_are_silent() -> None:
    search = SimpleNamespace(find_best=AsyncMock(return_value=None))
    ordinary = make_message("Спасибо за помощь")
    await handle_text(
        ordinary, SimpleNamespace(), ReplyCache(), search, None, None, None
    )
    search.find_best.assert_not_awaited()
    ordinary.answer.assert_not_awaited()
    ordinary.reply.assert_not_awaited()

    weak = make_message("Где оплатить подписку?")
    await handle_text(weak, SimpleNamespace(), ReplyCache(), search, None, None, None)
    search.find_best.assert_awaited_once()
    weak.answer.assert_not_awaited()
    weak.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_reply_is_saved_without_a_bot_message() -> None:
    question = make_message("Как войти в кабинет?", message_id=10)
    question.get_url = lambda: "https://t.me/c/1/10"
    answer = make_message("Откройте страницу входа.", message_id=11)
    answer.chat.title = "Тест"
    answer.from_user.id = 200
    answer.reply_to_message = question
    answer.get_url = lambda: "https://t.me/c/1/11"
    bot = SimpleNamespace(
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status=ChatMemberStatus.ADMINISTRATOR)
        )
    )
    chats = SimpleNamespace(ensure_chat=AsyncMock())
    solutions = SimpleNamespace(
        get_by_pair=AsyncMock(return_value=None), add_solution=AsyncMock(return_value=1)
    )
    embeddings = SimpleNamespace(
        embed_passage=AsyncMock(return_value=np.array([1.0, 0.0], dtype=np.float32))
    )
    search = SimpleNamespace(find_best=AsyncMock())

    await handle_text(answer, bot, ReplyCache(), search, chats, solutions, embeddings)

    solutions.add_solution.assert_awaited_once()
    assert solutions.add_solution.await_args.kwargs["question_text"] == question.text
    assert solutions.add_solution.await_args.kwargs["answer_text"] == answer.text
    search.find_best.assert_not_awaited()
    answer.answer.assert_not_awaited()
    answer.reply.assert_not_awaited()
