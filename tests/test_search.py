from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from tests.fixtures.questions import (
    ALL_MESSAGES,
    NO_SOLUTION,
    ORDINARY_MESSAGES,
    SAME_SOLUTION,
    WORD_OVERLAP_DIFFERENT,
)

from already_mentioned.database.connection import connect_database
from already_mentioned.database.schema import initialize_database
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.embeddings import EmbeddingService
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.search import SearchService


class FakeEmbeddings(EmbeddingService):
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.passages: list[str] = []

    async def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        if text in SAME_SOLUTION:
            return np.array([0.99, 0.1, 0.0, 0.0], dtype=np.float32)
        if text in WORD_OVERLAP_DIFFERENT[:10]:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if text in WORD_OVERLAP_DIFFERENT[10:]:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    async def embed_passage(self, text: str) -> np.ndarray:
        self.passages.append(text)
        if text in {
            "Откройте страницу входа.",
            "Используйте восстановление пароля.",
            "Откройте настройки профиля.",
            "Ответ другого чата.",
        }:
            return np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        if text == "Как войти в личный кабинет?":
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if text == "Как сбросить пароль?":
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


@pytest_asyncio.fixture
async def storage(
    tmp_path: Path,
) -> AsyncIterator[tuple[ChatRepository, SolutionRepository]]:
    connection = await connect_database(tmp_path / "search.db")
    await initialize_database(connection)
    try:
        yield ChatRepository(connection), SolutionRepository(connection)
    finally:
        await connection.close()


def test_eighty_synthetic_messages_and_question_filter() -> None:
    assert len(ALL_MESSAGES) == 80
    assert len(set(ALL_MESSAGES)) == 80
    assert all(map(is_question_candidate, SAME_SOLUTION))
    assert all(map(is_question_candidate, WORD_OVERLAP_DIFFERENT))
    assert all(map(is_question_candidate, NO_SOLUTION))
    assert not any(map(is_question_candidate, ORDINARY_MESSAGES))


@pytest.mark.asyncio
async def test_search_is_scoped_and_returns_one_strongest_match(
    storage: tuple[ChatRepository, SolutionRepository],
) -> None:
    chats, solutions = storage
    embeddings = FakeEmbeddings()
    search = SearchService(embeddings, chats, solutions)
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    login_id = await solutions.add_solution(
        chat_id=-1001,
        question_message_id=1,
        answer_message_id=2,
        question_text="Как войти в личный кабинет?",
        answer_text="Откройте страницу входа.",
        question_embedding=b"",
        question_link="https://t.me/c/1/1",
        answer_link="https://t.me/c/1/2",
    )
    reset_id = await solutions.add_solution(
        chat_id=-1001,
        question_message_id=3,
        answer_message_id=4,
        question_text="Как сбросить пароль?",
        answer_text="Используйте восстановление пароля.",
        question_embedding=b"",
        question_link="https://t.me/c/1/3",
        answer_link="https://t.me/c/1/4",
    )
    email_id = await solutions.add_solution(
        chat_id=-1001,
        question_message_id=5,
        answer_message_id=6,
        question_text="Как изменить почту?",
        answer_text="Откройте настройки профиля.",
        question_embedding=b"",
        question_link="https://t.me/c/1/5",
        answer_link="https://t.me/c/1/6",
    )
    other_id = await solutions.add_solution(
        chat_id=-1002,
        question_message_id=1,
        answer_message_id=2,
        question_text="Как войти в личный кабинет?",
        answer_text="Ответ другого чата.",
        question_embedding=b"",
        question_link="https://t.me/c/2/1",
        answer_link="https://t.me/c/2/2",
    )

    match = await search.find_best(-1001, SAME_SOLUTION[1])
    assert match is not None and match.solution.id == login_id
    assert match.similarity >= 0.88
    assert len(embeddings.passages) == 6  # Both vectors are backfilled.
    assert (await solutions.get_for_chat(-1001, login_id)).question_embedding
    assert (await solutions.get_for_chat(-1001, login_id)).answer_embedding

    wrong_words = await search.find_best(-1001, WORD_OVERLAP_DIFFERENT[0])
    assert wrong_words is not None and wrong_words.solution.id == reset_id
    other_words = await search.find_best(-1001, WORD_OVERLAP_DIFFERENT[10])
    assert other_words is not None and other_words.solution.id == email_id
    assert await search.find_best(-1001, NO_SOLUTION[0]) is None
    assert await search.find_best(-9999, SAME_SOLUTION[0]) is None

    other_match = await search.find_best(-1002, SAME_SOLUTION[1])
    assert other_match is not None and other_match.solution.id == other_id
    assert other_match.solution.answer_text == "Ответ другого чата."
    assert await solutions.get_for_chat(-1001, other_id) is None

    await chats.set_threshold(-1001, 1.0)
    assert await search.find_best(-1001, SAME_SOLUTION[1]) is None


@pytest.mark.asyncio
async def test_search_finds_answer_content_and_backfills_old_answer(
    storage: tuple[ChatRepository, SolutionRepository],
) -> None:
    chats, solutions = storage
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    answer_text = "Настройте двухфакторную аутентификацию в профиле."
    answer_vector = np.array([0.0, 1.0], dtype=np.float32)
    question_vector = np.array([1.0, 0.0], dtype=np.float32)
    solution_id = await solutions.add_solution(
        chat_id=-1001,
        question_message_id=1,
        answer_message_id=2,
        question_text="Как защитить аккаунт?",
        answer_text=answer_text,
        question_embedding=question_vector.tobytes(),
        question_link="https://t.me/c/1/1",
        answer_link="https://t.me/c/1/2",
    )
    await solutions.add_solution(
        chat_id=-1002,
        question_message_id=1,
        answer_message_id=2,
        question_text="Как защитить аккаунт?",
        answer_text="Другой ответ",
        question_embedding=question_vector.tobytes(),
        answer_embedding=answer_vector.tobytes(),
        question_link="https://t.me/c/2/1",
        answer_link="https://t.me/c/2/2",
    )

    class AnswerEmbeddings(EmbeddingService):
        def __init__(self) -> None:
            self.passages = []

        async def embed_query(self, text: str) -> np.ndarray:
            return answer_vector

        async def embed_passage(self, text: str) -> np.ndarray:
            self.passages.append(text)
            return answer_vector

    embeddings = AnswerEmbeddings()
    search = SearchService(embeddings, chats, solutions)
    match = await search.find_best(-1001, "Где включить двухфакторную аутентификацию?")
    assert match is not None and match.solution.id == solution_id
    assert embeddings.passages == [answer_text]
    assert (
        await solutions.get_for_chat(-1001, solution_id)
    ).answer_embedding == answer_vector.tobytes()
    await search.find_best(-1001, "Как включить второй фактор?")
    assert embeddings.passages == [answer_text]


@pytest.mark.asyncio
async def test_top_candidates_are_unique_scoped_and_available_below_threshold(storage):
    chats, solutions = storage
    await chats.ensure_chat(-1001, "Первый")
    await chats.ensure_chat(-1002, "Второй")
    await chats.set_threshold(-1001, 1.0)
    query = np.array([0.8, 0.6], dtype=np.float32)

    class Embeddings(EmbeddingService):
        calls = 0

        async def embed_query(self, text):
            self.calls += 1
            return query

        async def embed_passage(self, text):
            raise AssertionError("Векторы уже сохранены")

    ids = []
    for chat, q, a in (
        (-1001, [1., 0.], [1., 0.]),
        (-1001, [0., 1.], [0., 1.]),
        (-1002, [0.8, 0.6], [0.8, 0.6]),
    ):
        ids.append(await solutions.add_solution(
            chat_id=chat, question_message_id=1, answer_message_id=len(ids) + 2,
            question_text="Где вход?", answer_text="На сайте",
            question_embedding=np.array(q, dtype=np.float32).tobytes(),
            answer_embedding=np.array(a, dtype=np.float32).tobytes(),
            question_link="https://t.me/c/1/1", answer_link="https://t.me/c/1/2",
        ))
    embeddings = Embeddings()
    service = SearchService(embeddings, chats, solutions)
    ranked = await service.find_candidates(-1001, "Где вход?", backfill=False)
    assert [m.solution.id for m in ranked] == ids[:2]
    assert embeddings.calls == 1
    assert ranked[0].question_similarity == pytest.approx(0.8)
    assert ranked[0].answer_similarity == pytest.approx(0.8)
    assert await service.find_best(-1001, "Где вход?") is None
    with pytest.raises(ValueError):
        await service.find_candidates(-1001, "Где вход?", limit=0)
