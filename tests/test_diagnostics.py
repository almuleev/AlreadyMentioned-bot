"""The manual score check is read-only and reports the current decision."""

import numpy as np
import pytest

from already_mentioned import diagnostics
from already_mentioned.database.connection import connect_database
from already_mentioned.database.embedding_contract import require_frida
from already_mentioned.database.schema import initialize_database
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository


@pytest.mark.asyncio
async def test_inspect_question_reports_score_without_changing_database(
    tmp_path, monkeypatch, capsys
) -> None:
    database_path = tmp_path / "bot.db"
    connection = await connect_database(database_path)
    await initialize_database(connection)
    await require_frida(connection, adopt_empty=True)
    chats = ChatRepository(connection)
    await chats.ensure_chat(-1001, "Test")
    await chats.set_threshold(-1001, 0.9)
    await SolutionRepository(connection).add_solution(
        chat_id=-1001,
        question_message_id=1,
        answer_message_id=2,
        question_text="Где находится личный кабинет?",
        answer_text="На сайте.",
        question_embedding=np.pad(
            np.array([1.0, 0.0], dtype=np.float32), (0, 1534)
        ).tobytes(),
        answer_embedding=np.pad(
            np.array([0.0, 1.0], dtype=np.float32), (0, 1534)
        ).tobytes(),
        question_link="https://t.me/c/1/1",
        answer_link="https://t.me/c/1/2",
    )
    await connection.close()
    original_bytes = database_path.read_bytes()

    class FakeEmbeddings:
        async def embed_query(self, _text):
            return np.pad(
                np.array([0.88, np.sqrt(1 - 0.88**2)], dtype=np.float32), (0, 1534)
            )

        async def embed_passage(self, _text):
            return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(diagnostics, "FridaEmbeddingService", FakeEmbeddings)
    result = await diagnostics.inspect_question(
        "Где открыть кабинет?", None, database_path
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Лучшее совпадение: 0.8800" in output
    assert "Порог чата: 0.90" in output
    assert "бот промолчал бы из-за порога" in output
    assert "https://t.me/c/1/2" in output
    assert "Кандидаты FRIDA до порога: 1" in output
    assert "вопрос=0.8800; ответ=0.4750" in output
    assert database_path.read_bytes() == original_bytes
