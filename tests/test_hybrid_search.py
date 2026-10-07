import asyncio
import math
import threading
import time

import numpy as np
import pytest
import pytest_asyncio

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.search import SearchService


@pytest_asyncio.fixture(name="storage")
async def hybrid_storage(tmp_path):
    connection = await connect_database(tmp_path / "hybrid.db")
    await initialize_database(connection)
    try:
        yield ChatRepository(connection), SolutionRepository(connection)
    finally:
        await connection.close()


class Encoder:
    def __init__(self):
        self.calls = 0

    async def embed_query(self, text):
        self.calls += 1
        return np.array([1., 0.], dtype=np.float32)


class Scorer:
    def __init__(self):
        self.calls = []
        self.active = self.peak = 0

    def predict(self, query, passages):
        assert threading.current_thread() is not threading.main_thread()
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append(passages)
        time.sleep(.01)
        self.active -= 1
        return [.9 if "Ответ: Ответ 1" in text else 0 for text in passages]


async def prepare(storage):
    chats, solutions = storage
    await chats.ensure_chat(-1, "Первый")
    await chats.ensure_chat(-2, "Второй")
    ids = []
    for index, cosine in enumerate((.95, .85, .75, .65, .55, .45)):
        v = np.array([cosine, math.sqrt(1-cosine**2)], dtype=np.float32).tobytes()
        ids.append(await solutions.add_solution(
            chat_id=-1, question_message_id=index*2+1, answer_message_id=index*2+2,
            question_text=f"Вопрос {index}", answer_text=f"Ответ {index}",
            question_embedding=v, answer_embedding=v,
            question_link="q", answer_link="a",
        ))
    await solutions.add_solution(
        chat_id=-2, question_message_id=1, answer_message_id=2,
        question_text="Чужой вопрос", answer_text="Чужой ответ",
        question_embedding=np.array([1., 0.], dtype=np.float32).tobytes(),
        answer_embedding=np.array([1., 0.], dtype=np.float32).tobytes(),
        question_link="q", answer_link="a",
    )
    return chats, solutions, ids


@pytest.mark.asyncio
async def test_hybrid_reranks_same_chat_top5_with_separate_threshold(storage):
    chats, solutions, ids = await prepare(storage)
    await chats.set_threshold(-1, 1.)
    encoder, scorer = Encoder(), Scorer()
    search = SearchService(encoder, chats, solutions, mode="hybrid", reranker=scorer)
    match = await search.find_best(-1, "Запрос?")
    assert match.solution.id == ids[1]
    assert match.similarity == pytest.approx(.75*.85+.25*.9)
    assert match.question_similarity == pytest.approx(.85)
    assert encoder.calls == 1
    assert len(scorer.calls[0]) == 5
    assert scorer.calls[0][0] == "Вопрос: Вопрос 0\nОтвет: Ответ 0"
    assert all("Чужой" not in text for text in scorer.calls[0])
    await chats.set_threshold(-1, .99, hybrid=True)
    await chats.ensure_chat(-1, "Обновлён")
    chat = await chats.get_chat(-1)
    assert chat.similarity_threshold == 1.
    assert chat.hybrid_threshold == .99
    assert await search.find_best(-1, "Запрос?") is None
    assert await chats.get_chat(-2) is not None
    assert (await chats.get_chat(-2)).hybrid_threshold is None


@pytest.mark.asyncio
async def test_requests_serialize_reranker_without_blocking_loop(storage):
    chats, solutions, _ = await prepare(storage)
    scorer = Scorer()
    search = SearchService(Encoder(), chats, solutions, mode="hybrid", reranker=scorer)
    await asyncio.gather(*(search.find_best(-1, "Запрос?") for _ in range(4)))
    assert len(scorer.calls) == 4 and scorer.peak == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [[float('nan')]*5, [.1], [2.]*5])
async def test_bad_reranker_output_never_falls_back_to_unchecked_answer(storage, bad):
    chats, solutions, _ = await prepare(storage)
    scorer = Scorer()
    scorer.predict = lambda *_: bad
    search = SearchService(Encoder(), chats, solutions, mode="hybrid", reranker=scorer)
    with pytest.raises(ValueError, match="оценки MiniLM"):
        await search.find_best(-1, "Запрос?")


def test_hybrid_requires_scorer():
    with pytest.raises(ValueError):
        SearchService(None, None, None, mode="hybrid")
