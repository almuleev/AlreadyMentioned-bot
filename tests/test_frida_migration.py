"""Contract migration tests use fictional messages and temporary databases only."""

import json
import sqlite3
import sys

import numpy as np
import pytest

from already_mentioned import diagnostics
from already_mentioned.database import reembed
from already_mentioned.database.backup import verify_backup
from already_mentioned.database.connection import connect_database
from already_mentioned.database.embedding_contract import (
    read_contract,
    require_frida,
    validate_blob,
)
from already_mentioned.database.process_lock import database_process_lock
from already_mentioned.database.schema import initialize_database
from already_mentioned.services.embeddings import E5_CONTRACT, FRIDA_CONTRACT


def vector(index=0):
    result = np.zeros(1536, dtype=np.float32)
    result[index] = 1
    return result


class FakeFrida:
    def __init__(self, fail_at=None):
        self.seen = []
        self.fail_at = fail_at

    async def embed_passage(self, text):
        self.seen.append(text)
        if len(self.seen) == self.fail_at:
            raise RuntimeError("Ошибка модели с приватным текстом")
        return vector(len(self.seen))


async def old_database(path):
    connection = await connect_database(path)
    await initialize_database(connection)
    await connection.executemany(
        "INSERT INTO chats (telegram_chat_id, title, similarity_threshold) "
        "VALUES (?, ?, ?)",
        [(-1, "Первый", 0.73), (-2, "Второй", 0.94)],
    )
    await connection.executemany(
        "INSERT INTO solutions (chat_id, question_message_id, answer_message_id, "
        "question_text, answer_text, question_embedding, question_link, answer_link) "
        "VALUES (?, 1, 2, ?, ?, ?, 'q', 'a')",
        [(-1, "Вопрос А", "Ответ А", b"old-a"), (-2, "Вопрос Б", "Ответ Б", b"old-b")],
    )
    await connection.execute(
        "INSERT INTO feedback (solution_id, query_message_id, user_id, vote) "
        "VALUES (1, 3, 4, 'helpful')"
    )
    await connection.execute("DROP TABLE embedding_state")
    await connection.execute("PRAGMA user_version = 2")
    await connection.commit()
    await connection.close()


@pytest.mark.asyncio
async def test_full_migration_preserves_chats_votes_texts_and_is_idempotent(tmp_path):
    path, backup = tmp_path / "old.db", tmp_path / "backup.db"
    await old_database(path)
    encoder = FakeFrida()
    assert await reembed.migrate_embeddings(path, backup, encoder) == 2
    assert encoder.seen == ["Вопрос А", "Ответ А", "Вопрос Б", "Ответ Б"]
    assert verify_backup(backup) == 2
    with sqlite3.connect(backup) as old:
        assert old.execute("SELECT question_embedding FROM solutions").fetchall() == [
            (b"old-a",),
            (b"old-b",),
        ]
    connection = await connect_database(path)
    try:
        await require_frida(connection)
        assert await read_contract(connection) == FRIDA_CONTRACT
        async with connection.execute(
            "SELECT telegram_chat_id, similarity_threshold "
            "FROM chats ORDER BY telegram_chat_id"
        ) as c:
            assert await c.fetchall() == [(-2, 0.94), (-1, 0.73)]
        async with connection.execute("SELECT vote FROM feedback") as c:
            assert await c.fetchall() == [("helpful",)]
        async with connection.execute(
            "SELECT question_text, answer_text FROM solutions"
        ) as c:
            assert await c.fetchall() == [
                ("Вопрос А", "Ответ А"),
                ("Вопрос Б", "Ответ Б"),
            ]
    finally:
        await connection.close()
    before = path.read_bytes()
    no_encode = FakeFrida(fail_at=1)
    assert (
        await reembed.migrate_embeddings(path, tmp_path / "second.db", no_encode) == 2
    )
    assert no_encode.seen == []
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_mid_migration_failure_rolls_back_all_vectors_and_contract(tmp_path):
    path, backup = tmp_path / "old.db", tmp_path / "backup.db"
    await old_database(path)
    with pytest.raises(RuntimeError):
        await reembed.migrate_embeddings(path, backup, FakeFrida(fail_at=4))
    connection = await connect_database(path)
    try:
        assert await read_contract(connection) == E5_CONTRACT
        async with connection.execute(
            "SELECT question_embedding, answer_embedding FROM solutions"
        ) as c:
            assert await c.fetchall() == [(b"old-a", None), (b"old-b", None)]
        with pytest.raises(RuntimeError, match="другой контракт"):
            await require_frida(connection, adopt_empty=True)
    finally:
        await connection.close()
    assert verify_backup(backup) == 2


@pytest.mark.asyncio
async def test_empty_database_adopts_frida_and_rejects_same_shape_wrong_contract(
    tmp_path,
):
    path = tmp_path / "new.db"
    connection = await connect_database(path)
    try:
        await initialize_database(connection)
        await require_frida(connection, adopt_empty=True)
        await require_frida(connection)
        wrong = {**FRIDA_CONTRACT, "query_prefix": "paraphrase:"}
        await connection.execute(
            "UPDATE embedding_state SET contract = ?", (json.dumps(wrong),)
        )
        await connection.commit()
        with pytest.raises(RuntimeError):
            await require_frida(connection)
    finally:
        await connection.close()


@pytest.mark.parametrize(
    "blob",
    [
        None,
        b"bad",
        np.zeros(1536, dtype=np.float32).tobytes(),
        np.full(1536, np.nan, dtype=np.float32).tobytes(),
    ],
)
def test_vector_validation_rejects_missing_wrong_dimension_and_invalid_values(blob):
    with pytest.raises(ValueError):
        validate_blob(blob)


def test_process_lock_excludes_polling_and_migration_and_releases(tmp_path):
    path = tmp_path / "bot.db"
    with database_process_lock(path), pytest.raises(RuntimeError):
        with database_process_lock(path):
            pytest.fail("Не должна быть доступна вторая блокировка")
    with database_process_lock(path):
        pass


@pytest.mark.asyncio
async def test_diagnostics_rejects_legacy_before_loading_any_model(
    tmp_path, monkeypatch
):
    path = tmp_path / "old.db"
    await old_database(path)
    before = path.read_bytes()
    monkeypatch.setattr(
        diagnostics, "FridaEmbeddingService", lambda: pytest.fail("Модель")
    )
    assert await diagnostics.inspect_question("Как войти?", -1, path) == 1
    assert path.read_bytes() == before


def test_cli_requires_stopped_bot_and_redacts_model_errors(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["reembed", "missing.db", "--backup", "copy.db"])
    with pytest.raises(SystemExit) as error:
        reembed.main()
    assert error.value.code == 2
    monkeypatch.setattr(
        sys, "argv", ["reembed", "missing.db", "--backup", "copy.db", "--bot-stopped"]
    )

    async def fail(*args):
        raise RuntimeError("secret text")

    monkeypatch.setattr(reembed, "migrate_embeddings", fail)
    with pytest.raises(SystemExit) as error:
        reembed.main()
    assert error.value.code == 1
    assert "secret text" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_polling_warms_frida_and_legacy_database_fails_before_model(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    from already_mentioned import main
    from already_mentioned.config import Settings

    events = []

    class Encoder:
        async def embed_query(self, text):
            events.append("warm")
            return vector()

    class Dispatcher:
        def include_router(self, router):
            pass

        async def start_polling(self, bot, **kwargs):
            assert isinstance(kwargs["embeddings"], Encoder)
            assert kwargs["search"].embeddings is kwargs["embeddings"]
            events.append("polling")

    async def close():
        events.append("close")

    monkeypatch.setattr(
        main,
        "create_bot",
        lambda settings: SimpleNamespace(
            session=SimpleNamespace(close=close),
        ),
    )
    monkeypatch.setattr(main, "Dispatcher", Dispatcher)
    monkeypatch.setattr(main, "FridaEmbeddingService", Encoder)
    await main._run(Settings(None, tmp_path / "new.db"))
    assert events == ["warm", "polling", "close"]
    events.clear()
    await old_database(tmp_path / "legacy.db")
    with pytest.raises(RuntimeError, match="другой контракт"):
        await main._run(Settings(None, tmp_path / "legacy.db"))
    assert events == ["close"]
