"""Offline comparison of independent embedding spaces on fictional cases."""

import argparse
import asyncio
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from already_mentioned.evaluation import Result, benchmark, evaluate
from already_mentioned.evaluation_cases import CASES
from already_mentioned.services.embeddings import (
    EmbeddingService,
    FridaEmbeddingService,
    normalize_vector,
)
from already_mentioned.services.questions import is_question_candidate

ALTERNATIVES = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "potion": "minishlab/potion-multilingual-128M",
}


class PlainTextFastEmbed(EmbeddingService):
    """Use an alternative model's own plain-text contract, never bot vectors."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir="models")

    async def embed_query(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._encode, text)

    async def embed_passage(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._encode, text)

    def _encode(self, text: str) -> np.ndarray:
        return normalize_vector(next(iter(self.model.embed([text]))))


@dataclass(frozen=True)
class Comparison:
    name: str
    dimension: int
    baseline: Result
    selected_threshold: float
    selected: Result
    query_ms: float
    passage_ms: float
    rank_100_ms: float
    rank_1000_ms: float


def select_threshold(
    queries: dict[str, np.ndarray], passages: dict[str, np.ndarray]
) -> tuple[float, Result]:
    """Exploratory in-sample sweep; minimize wrong hints, then maximize hits."""
    options = [
        (step / 100, evaluate(CASES, queries, passages, step / 100))
        for step in range(50, 101)
    ]
    return max(
        options,
        key=lambda item: (-item[1].wrong, item[1].correct, item[0]),
    )


async def compare_one(name: str, embeddings: EmbeddingService) -> Comparison:
    # Warm-up includes lazy model loading; timed work below excludes it.
    await embeddings.embed_query(CASES[0].message)
    messages = sorted(
        {case.message for case in CASES if is_question_candidate(case.message)}
    )
    saved = sorted({text for case in CASES for _, text in case.saved})
    start = perf_counter()
    queries = {
        text: normalize_vector(await embeddings.embed_query(text)) for text in messages
    }
    query_ms = (perf_counter() - start) * 1000 / len(messages)
    start = perf_counter()
    passages = {
        text: normalize_vector(await embeddings.embed_passage(text)) for text in saved
    }
    passage_ms = (perf_counter() - start) * 1000 / len(saved)
    threshold, selected = select_threshold(queries, passages)
    query = next(iter(queries.values()))
    timing = benchmark(query, tuple(passages.values()), (100, 1000), 5)
    return Comparison(
        name, query.size, evaluate(CASES, queries, passages, 0.88),
        threshold, selected, query_ms, passage_ms, timing[0][1], timing[1][1],
    )


async def run(names: tuple[str, ...], *, details: bool = False) -> None:
    for name in names:
        try:
            embeddings: EmbeddingService = (
                FridaEmbeddingService() if name == "current"
                else PlainTextFastEmbed(ALTERNATIVES[name])
            )
            result = await compare_one(name, embeddings)
        except Exception as error:
            print(f"{name}: проверка недоступна ({type(error).__name__})")
            continue
        print(
            f"{name}: {result.dimension} измерений; порог 0.88: "
            f"верно {result.baseline.correct}, ошибочно {result.baseline.wrong}, "
            f"пропущено {result.baseline.missed}; подобранный на этом наборе "
            f"порог {result.selected_threshold:.2f}: верно {result.selected.correct}, "
            f"ошибочно {result.selected.wrong}, пропущено {result.selected.missed}; "
            f"кодирование запроса {result.query_ms:.2f} мс, сохранённого вопроса "
            f"{result.passage_ms:.2f} мс; перебор 100/1000: "
            f"{result.rank_100_ms:.2f}/{result.rank_1000_ms:.2f} мс"
        )
        if details:
            for detail in result.selected.details:
                print(f"  {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить embeddings-модели на вымышленных примерах"
    )
    parser.add_argument(
        "--models", nargs="+", choices=("current", *ALTERNATIVES),
        default=("current", *ALTERNATIVES),
    )
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(tuple(args.models), details=args.details))


if __name__ == "__main__":
    main()
