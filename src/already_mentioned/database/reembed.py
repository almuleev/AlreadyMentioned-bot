"""Explicit, backed-up, all-or-nothing migration of all saved vectors to FRIDA."""

import argparse
import asyncio
from pathlib import Path

from already_mentioned.database.backup import create_backup, verify_backup
from already_mentioned.database.connection import connect_database
from already_mentioned.database.embedding_contract import (
    contract_json,
    read_contract,
    validate_blob,
    verify_vectors,
)
from already_mentioned.database.process_lock import database_process_lock
from already_mentioned.database.schema import initialize_database
from already_mentioned.services.embeddings import (
    E5_CONTRACT,
    FRIDA_CONTRACT,
    EmbeddingService,
    FridaEmbeddingService,
)


async def migrate_embeddings(
    database_path: Path, backup_path: Path, embeddings: EmbeddingService
) -> int:
    """Caller must stop old polling processes too; they do not share our lock."""
    if not database_path.is_file():
        raise ValueError("Рабочая база не найдена")
    with database_process_lock(database_path):
        # Backup predates both schema adoption and vector migration.
        await asyncio.to_thread(create_backup, database_path, backup_path)
        connection = await connect_database(database_path)
        try:
            await initialize_database(connection)
            await connection.execute("BEGIN EXCLUSIVE")
            try:
                contract = await read_contract(connection)
                if contract not in (E5_CONTRACT, FRIDA_CONTRACT):
                    raise RuntimeError("Неизвестный исходный контракт embeddings")
                if contract == FRIDA_CONTRACT:
                    count = await verify_vectors(connection)
                else:
                    count = 0
                    async with connection.execute(
                        "SELECT id, question_text, answer_text "
                        "FROM solutions ORDER BY id"
                    ) as cursor:
                        rows = await cursor.fetchall()
                    for solution_id, question, answer in rows:
                        q = (
                            (await embeddings.embed_passage(question))
                            .astype("<f4")
                            .tobytes()
                        )
                        a = (
                            (await embeddings.embed_passage(answer))
                            .astype("<f4")
                            .tobytes()
                        )
                        validate_blob(q)
                        validate_blob(a)
                        await connection.execute(
                            "UPDATE solutions SET question_embedding = ?, "
                            "answer_embedding = ? WHERE id = ?",
                            (q, a, solution_id),
                        )
                        count += 1
                    if await verify_vectors(connection) != len(rows):
                        raise RuntimeError("Не все решения проверены после пересчёта")
                    await connection.execute(
                        "UPDATE embedding_state SET contract = ? WHERE id = 1",
                        (contract_json(FRIDA_CONTRACT),),
                    )
                async with connection.execute("PRAGMA foreign_key_check") as cursor:
                    if await cursor.fetchone() is not None:
                        raise RuntimeError("Нарушены внешние ключи")
                async with connection.execute("PRAGMA integrity_check") as cursor:
                    if (await cursor.fetchone())[0] != "ok":
                        raise RuntimeError("Нарушена целостность SQLite")
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        finally:
            await connection.close()
        await asyncio.to_thread(verify_backup, database_path)
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный пересчёт embeddings на FRIDA")
    parser.add_argument("database", type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--bot-stopped",
        required=True,
        action="store_true",
        help="Подтверждение остановки всех процессов бота, включая старую версию",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Разрешить загрузку закреплённой модели при отсутствии кэша",
    )
    args = parser.parse_args()
    embeddings = FridaEmbeddingService(
        args.cache_dir,
        local_files_only=not args.download_model,
    )
    try:
        count = asyncio.run(migrate_embeddings(args.database, args.backup, embeddings))
    except Exception as error:
        # Third-party exceptions may contain texts, so only disclose their type.
        parser.exit(
            1,
            f"Пересчёт не выполнен ({type(error).__name__}); "
            "проверьте кэш, остановку бота и резервную копию.\n",
        )
    print(f"FRIDA: проверены {count} решений во всех чатах; копия: {args.backup}")
    print("Пороги, тексты, ссылки и оценки сохранены. Проверьте качество подсказок.")


if __name__ == "__main__":
    main()
