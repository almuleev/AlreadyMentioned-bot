"""Read-only, on-demand similarity check for a supplied question."""

import argparse
import asyncio
from pathlib import Path

import aiosqlite
import numpy as np

from already_mentioned.config import load_settings
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.embeddings import (
    FastEmbedEmbeddingService,
    normalize_vector,
)
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.similarity import cosine_similarity


async def inspect_question(
    question: str, chat_id: int | None, database_path: Path
) -> int:
    """Print the best saved match without changing the database or sending messages."""
    if not database_path.is_file():
        print(f"База данных не найдена: {database_path}")
        return 1

    database_uri = database_path.resolve().as_uri() + "?mode=ro"
    async with aiosqlite.connect(database_uri, uri=True) as connection:
        chats = ChatRepository(connection)
        solutions = SolutionRepository(connection)
        if chat_id is None:
            async with connection.execute(
                "SELECT telegram_chat_id FROM chats"
            ) as cursor:
                chat_ids = [row[0] for row in await cursor.fetchall()]
            if len(chat_ids) != 1:
                print(
                    "Укажите --chat-id: в базе должно быть ровно одно чат-сообщество "
                    "для автоматического выбора."
                )
                return 1
            chat_id = chat_ids[0]

        chat = await chats.get_chat(chat_id)
        if chat is None:
            print(f"Чат {chat_id} не найден в базе.")
            return 1
        candidates = await solutions.list_for_chat(chat_id)
        if not candidates:
            print(f"В чате {chat_id} пока нет сохранённых решений.")
            return 0

        embeddings = FastEmbedEmbeddingService()
        query = normalize_vector(await embeddings.embed_query(question))
        best = None
        skipped = 0
        for solution in candidates:
            for source, text, data in (
                ("вопрос", solution.question_text, solution.question_embedding),
                ("ответ", solution.answer_text, solution.answer_embedding),
            ):
                try:
                    passage = (
                        np.frombuffer(data, dtype=np.float32)
                        if data
                        else await embeddings.embed_passage(text)
                    )
                    score = cosine_similarity(query, passage)
                except ValueError:
                    skipped += 1
                    continue
                if best is None or score > best[0]:
                    best = (score, solution, source)

    if best is None:
        print("Не удалось прочитать embeddings сохранённых решений.")
        return 1

    score, solution, source = best
    print(f"Чат: {chat_id}")
    print(f"Лучшее совпадение: {score:.4f}")
    print(f"Совпало с: {source}")
    print(f"Порог чата: {chat.similarity_threshold:.2f}")
    if not is_question_candidate(question):
        print("Результат: бот промолчал бы — сообщение не прошло фильтр вопросов.")
    elif score >= chat.similarity_threshold:
        print("Результат: бот предложил бы ссылку.")
    else:
        print("Результат: бот промолчал бы из-за порога.")
    print(f"Сохранённый вопрос: {solution.question_text}")
    print(f"Ссылка на ответ: {solution.answer_link}")
    if skipped:
        print(f"Пропущено решений с повреждённым embedding: {skipped}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Показать score нового вопроса относительно решений одного чата."
    )
    parser.add_argument("question", help="Точный текст вопроса в кавычках")
    parser.add_argument("--chat-id", type=int, help="ID чата, если в базе их несколько")
    args = parser.parse_args()
    settings = load_settings()
    raise SystemExit(
        asyncio.run(
            inspect_question(args.question, args.chat_id, settings.database_path)
        )
    )


if __name__ == "__main__":
    main()
