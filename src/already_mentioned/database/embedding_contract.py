"""Fail closed before comparing persisted vectors from incompatible encoders."""

import json

import aiosqlite
import numpy as np

from already_mentioned.services.embeddings import FRIDA_CONTRACT, FRIDA_DIMENSION


def contract_json(contract: dict) -> str:
    return json.dumps(contract, sort_keys=True)


async def read_contract(connection: aiosqlite.Connection) -> dict:
    try:
        async with connection.execute(
            "SELECT contract FROM embedding_state WHERE id = 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            return json.loads(row[0])
    except (aiosqlite.Error, ValueError):
        pass
    raise RuntimeError("Нет контракта embeddings; выполните миграцию FRIDA")


def validate_blob(data: bytes | None) -> None:
    if data is None or len(data) != FRIDA_DIMENSION * 4:
        raise ValueError("Сохранённый вектор не соответствует FRIDA")
    vector = np.frombuffer(data, dtype="<f4")
    if not np.all(np.isfinite(vector)) or not np.isclose(
        np.linalg.norm(vector), 1.0, atol=1e-4
    ):
        raise ValueError("Сохранённый вектор FRIDA повреждён")


async def verify_vectors(connection: aiosqlite.Connection) -> int:
    count = 0
    async with connection.execute(
        "SELECT question_embedding, answer_embedding FROM solutions"
    ) as cursor:
        async for question, answer in cursor:
            validate_blob(question)
            validate_blob(answer)
            count += 1
    return count


async def require_frida(
    connection: aiosqlite.Connection, *, adopt_empty: bool = False
) -> None:
    if adopt_empty:
        await connection.execute("BEGIN IMMEDIATE")
    try:
        contract = await read_contract(connection)
        if contract != FRIDA_CONTRACT:
            async with connection.execute("SELECT COUNT(*) FROM solutions") as cursor:
                count = (await cursor.fetchone())[0]
            if not adopt_empty or count:
                raise RuntimeError(
                    "База содержит другой контракт embeddings. Остановите бота и "
                    "выполните python -m already_mentioned.database.reembed"
                )
            await connection.execute(
                "UPDATE embedding_state SET contract = ? WHERE id = 1",
                (contract_json(FRIDA_CONTRACT),),
            )
        await verify_vectors(connection)
        if adopt_empty:
            await connection.commit()
    except BaseException:
        if adopt_empty:
            await connection.rollback()
        raise
