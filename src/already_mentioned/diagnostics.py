"""Read-only, on-demand similarity check for a supplied question."""

import argparse
import asyncio
from pathlib import Path

import aiosqlite

from already_mentioned.config import load_settings
from already_mentioned.database.embedding_contract import require_frida
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.embeddings import FridaEmbeddingService
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reranking import OnnxReranker
from already_mentioned.services.search import SearchService


async def inspect_question(
    question: str,
    chat_id: int | None,
    database_path: Path,
    *,
    top_k: int = 5,
    mode: str = "baseline",
) -> int:
    """Print the best saved match without changing the database or sending messages."""
    if not database_path.is_file():
        print(f"База данных не найдена: {database_path}")
        return 1

    database_uri = database_path.resolve().as_uri() + "?mode=ro"
    async with aiosqlite.connect(database_uri, uri=True) as connection:
        await connection.execute("BEGIN")
        try:
            await require_frida(connection)
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
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
        embeddings = FridaEmbeddingService()
        reranker = (
            OnnxReranker(local_only=True, batch_size=1) if mode == "hybrid" else None
        )
        search = SearchService(
            embeddings, chats, solutions, mode=mode, reranker=reranker
        )
        ranked = await search.find_candidates(
            chat_id,
            question,
            limit=max(top_k, 5) if mode == "hybrid" else top_k,
            backfill=False,
        )
        if not ranked:
            print(f"В чате {chat_id} пока нет сохранённых решений.")
            return 0
        decision = await search.rerank_candidates(question, ranked[:5])
        threshold = search.threshold_for(chat)
    best = decision[0]
    score, solution = best.similarity, best.solution
    source = (
        "вопрос"
        if best.question_similarity is not None
        and (
            best.answer_similarity is None
            or best.question_similarity >= best.answer_similarity
        )
        else "ответ"
    )
    if mode == "hybrid":
        source = "смесь FRIDA и MiniLM"
    print(f"Чат: {chat_id}")
    print(f"Лучшее совпадение: {score:.4f}")
    print(f"Совпало с: {source}")
    print(f"Поиск: {search.mode_label}")
    print(f"Порог чата: {threshold:.3f}")
    if not is_question_candidate(question):
        print("Результат: бот промолчал бы — сообщение не прошло фильтр вопросов.")
    elif score >= threshold:
        print("Результат: бот предложил бы ссылку.")
    else:
        print("Результат: бот промолчал бы из-за порога.")
    print(f"Сохранённый вопрос: {solution.question_text}")
    print(f"Ссылка на ответ: {solution.answer_link}")
    print(f"Кандидаты FRIDA до порога: {len(ranked[:top_k])}")
    for index, match in enumerate(ranked[:top_k], 1):
        question_score = (
            f"{match.question_similarity:.4f}"
            if match.question_similarity is not None
            else "нет"
        )
        answer_score = (
            f"{match.answer_similarity:.4f}"
            if match.answer_similarity is not None
            else "нет"
        )
        print(
            f"{index}. Решение {match.solution.id}: score={match.similarity:.4f}; "
            f"вопрос={question_score}; ответ={answer_score}"
        )
        print(f"   Вопрос: {match.solution.question_text}")
        print(f"   Ответ: {match.solution.answer_text}")
        print(f"   Источник: {match.solution.answer_link}")
    if len(ranked) > 1:
        print(
            f"Отрыв первого кандидата: "
            f"{ranked[0].similarity - ranked[1].similarity:.4f}"
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Показать score нового вопроса относительно решений одного чата."
    )
    parser.add_argument("question", help="Точный текст вопроса в кавычках")
    parser.add_argument("--chat-id", type=int, help="ID чата, если в базе их несколько")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Число кандидатов до порога (1–100, по умолчанию 5)",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100:
        parser.error("Число кандидатов должно быть от 1 до 100")
    settings = load_settings()
    raise SystemExit(
        asyncio.run(
            inspect_question(
                args.question,
                args.chat_id,
                settings.database_path,
                top_k=args.top_k,
                mode=settings.search_mode,
            )
        )
    )


if __name__ == "__main__":
    main()
