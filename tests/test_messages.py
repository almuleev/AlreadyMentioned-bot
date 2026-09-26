from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.enums import ChatType

from already_mentioned.handlers.messages import handle_text
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
    )


@pytest.mark.asyncio
async def test_strong_question_offers_one_link_with_feedback_buttons() -> None:
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
    await handle_text(message, SimpleNamespace(), ReplyCache(), search)

    search.find_best.assert_awaited_once_with(-1001, message.text)
    assert "https://t.me/c/1/2" in message.answer.await_args.args[0]
    buttons = message.answer.await_args.kwargs["reply_markup"].inline_keyboard[0]
    assert [button.text for button in buttons] == ["👍 Помогло", "👎 Не подходит"]
    assert [button.callback_data for button in buttons] == [
        "vote:7:50:helpful",
        "vote:7:50:not_helpful",
    ]


@pytest.mark.asyncio
async def test_ordinary_text_and_weak_question_are_silent() -> None:
    search = SimpleNamespace(find_best=AsyncMock(return_value=None))
    ordinary = make_message("Спасибо за помощь")
    await handle_text(ordinary, SimpleNamespace(), ReplyCache(), search)
    search.find_best.assert_not_awaited()
    ordinary.answer.assert_not_awaited()

    weak = make_message("Где оплатить подписку?")
    await handle_text(weak, SimpleNamespace(), ReplyCache(), search)
    search.find_best.assert_awaited_once()
    weak.answer.assert_not_awaited()
