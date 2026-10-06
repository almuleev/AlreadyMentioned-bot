"""Offline evaluation of the question filter and saved-question ranking."""

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from already_mentioned.config import DEFAULT_SIMILARITY_THRESHOLD
from already_mentioned.evaluation_cases import CASES, Case
from already_mentioned.services.embeddings import (
    EmbeddingService,
    FridaEmbeddingService,
    normalize_vector,
)
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.similarity import cosine_similarity


@dataclass(frozen=True)
class Result:
    correct: int
    wrong: int
    missed: int
    correct_silence: int
    filter_missed: int
    ranking_missed: int
    wrong_best: int
    below_threshold: int
    details: tuple[str, ...]


def rank(
    query: np.ndarray, passages: tuple[tuple[str, np.ndarray], ...]
) -> tuple[str | None, float]:
    """Use the bot's cosine score and first-maximum tie rule."""
    best_id: str | None = None
    best_score = -2.0
    for solution_id, passage in passages:
        try:
            score = cosine_similarity(query, passage)
        except ValueError:
            continue
        if score > best_score:
            best_id, best_score = solution_id, score
    return best_id, best_score


async def encode_cases(
    cases: tuple[Case, ...], embeddings: EmbeddingService
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    messages = {case.message for case in cases if is_question_candidate(case.message)}
    saved = {text for case in cases for _, text in case.saved}
    queries = {
        message: normalize_vector(await embeddings.embed_query(message))
        for message in sorted(messages)
    }
    passages = {
        text: normalize_vector(await embeddings.embed_passage(text))
        for text in sorted(saved)
    }
    return queries, passages


def evaluate(
    cases: tuple[Case, ...],
    queries: dict[str, np.ndarray],
    passages: dict[str, np.ndarray],
    threshold: float,
) -> Result:
    counts: Counter[str] = Counter()
    details = []
    for case in cases:
        passed = is_question_candidate(case.message)
        chosen: str | None = None
        best_id: str | None = None
        score = -2.0
        if passed and case.saved:
            best_id, score = rank(
                queries[case.message],
                tuple(
                    (solution_id, passages[text]) for solution_id, text in case.saved
                ),
            )
            chosen = best_id if score >= threshold else None
        if chosen == case.expected:
            counts["correct" if chosen is not None else "correct_silence"] += 1
        else:
            if chosen is not None:
                counts["wrong"] += 1
            if case.expected is not None:
                counts["missed"] += 1
                counts["ranking_missed" if passed else "filter_missed"] += 1
        if passed and case.expected is not None:
            if best_id is not None and best_id != case.expected:
                counts["wrong_best"] += 1
            elif best_id == case.expected and score < threshold:
                counts["below_threshold"] += 1
        stage = "фильтр" if not passed else "ранжирование"
        details.append(
            f"{case.name}: ожидалось {case.expected or 'молчание'}, "
            f"лучшее {best_id or 'нет'}, получено {chosen or 'молчание'}; "
            f"этап {stage}; "
            f"score {score:.4f}"
            if passed
            else f"{case.name}: ожидалось {case.expected or 'молчание'}, "
            f"получено молчание; этап фильтр"
        )
    return Result(
        counts["correct"],
        counts["wrong"],
        counts["missed"],
        counts["correct_silence"],
        counts["filter_missed"],
        counts["ranking_missed"],
        counts["wrong_best"],
        counts["below_threshold"],
        tuple(details),
    )


def benchmark(
    query: np.ndarray,
    passages: tuple[np.ndarray, ...],
    sizes: tuple[int, ...],
    repeats: int,
) -> tuple[tuple[int, float], ...]:
    """Time only the same linear cosine ranking; vectors are already encoded."""
    results = []
    for size in sizes:
        candidates = tuple((str(i), passages[i % len(passages)]) for i in range(size))
        start = perf_counter()
        for _ in range(repeats):
            rank(query, candidates)
        results.append((size, (perf_counter() - start) * 1000 / repeats))
    return tuple(results)


def benchmark_filter(messages: tuple[str, ...], repeats: int) -> float:
    """Measure the preliminary filter without embedding or ranking."""
    start = perf_counter()
    for _ in range(repeats):
        for message in messages:
            is_question_candidate(message)
    return (perf_counter() - start) * 1000 / (repeats * len(messages))


async def run(
    embeddings: EmbeddingService,
    thresholds: tuple[float, ...],
    sizes: tuple[int, ...],
    repeats: int,
) -> None:
    queries, passages = await encode_cases(CASES, embeddings)
    print(
        f"Набор: {len(CASES)} примеров; сохранённые вопросы заданы "
        "отдельно для каждого примера"
    )
    for threshold in thresholds:
        result = evaluate(CASES, queries, passages, threshold)
        print(
            f"Порог {threshold:.2f}: правильных подсказок {result.correct}; "
            f"ошибочных подсказок {result.wrong}; пропущенных подходящих ответов "
            f"{result.missed} (фильтр {result.filter_missed}, "
            f"ранжирование/порог {result.ranking_missed}, "
            f"неверный лидер {result.wrong_best}, "
            f"верный лидер ниже порога {result.below_threshold}); "
            f"верное молчание {result.correct_silence}"
        )
        for detail in result.details:
            print(f"  {detail}")
    query = next(iter(queries.values()))
    filter_ms = benchmark_filter(tuple(case.message for case in CASES), repeats)
    print(f"Время предварительного фильтра: {filter_ms:.4f} мс/сообщение")
    print("Время ранжирования (без фильтра и кодирования; мс/поиск):")
    for size, elapsed in benchmark(query, tuple(passages.values()), sizes, repeats):
        print(f"  {size} решений: {elapsed:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Офлайн-проверка фильтра и поиска на вымышленных примерах"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[DEFAULT_SIMILARITY_THRESHOLD, 0.88],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in args.thresholds):
        parser.error("порог должен быть от 0 до 1")
    if any(value < 1 for value in args.sizes) or args.repeats < 1:
        parser.error("размеры и число повторов должны быть положительными")
    asyncio.run(
        run(
            FridaEmbeddingService(),
            tuple(args.thresholds),
            tuple(args.sizes),
            args.repeats,
        )
    )


if __name__ == "__main__":
    main()
