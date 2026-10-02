"""Compare frozen embedding models on the same public held-out triplets."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from already_mentioned.model_comparison import ALTERNATIVES, PlainTextFastEmbed
from already_mentioned.offline_training import (
    DATASET,
    REVISION,
    SHA256,
    SPLITS,
    Triplet,
    download_split,
    load_triplets,
)
from already_mentioned.services.embeddings import (
    EmbeddingService,
    FastEmbedEmbeddingService,
)


def create_model(name: str) -> EmbeddingService:
    if name == "current":
        return FastEmbedEmbeddingService()
    return PlainTextFastEmbed(ALTERNATIVES[name])


def metrics(positive: np.ndarray, negative: np.ndarray, threshold: float) -> dict:
    tp = int(np.count_nonzero(positive >= threshold))
    fn = len(positive) - tp
    fp = int(np.count_nonzero(negative >= threshold))
    tn = len(negative) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def select_threshold(positive: np.ndarray, negative: np.ndarray) -> float:
    """Maximize validation F1; break ties toward fewer false suggestions."""
    scores = np.unique(np.concatenate((positive, negative)))
    options = np.concatenate(([np.nextafter(scores[0], -np.inf)], scores,
                              [np.nextafter(scores[-1], np.inf)]))
    return float(max(
        options,
        key=lambda threshold: (
            metrics(positive, negative, threshold)["f1"],
            -metrics(positive, negative, threshold)["fp"],
            threshold,
        ),
    ))


async def score_rows(
    embeddings: EmbeddingService, rows: tuple[Triplet, ...]
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    start = perf_counter()
    queries = np.stack([await embeddings.embed_query(row.query) for row in rows])
    query_ms = (perf_counter() - start) * 1000 / len(rows)
    start = perf_counter()
    positives = np.stack(
        [await embeddings.embed_passage(row.positive) for row in rows]
    )
    negatives = np.stack(
        [await embeddings.embed_passage(row.negative) for row in rows]
    )
    passage_ms = (perf_counter() - start) * 1000 / (2 * len(rows))
    positive_scores = np.sum(queries * positives, axis=1)
    negative_scores = np.sum(queries * negatives, axis=1)
    return positive_scores, negative_scores, query_ms, passage_ms, queries.shape[1]


async def measure_model(
    name: str, validation_rows: tuple[Triplet, ...], test_rows: tuple[Triplet, ...]
) -> dict:
    embeddings = create_model(name)
    # Lazy model loading is excluded from the timed comparison.
    await embeddings.embed_query(test_rows[0].query)
    val_positive, val_negative, _, _, _ = await score_rows(
        embeddings, validation_rows
    )
    threshold = select_threshold(val_positive, val_negative)
    positive, negative, query_ms, passage_ms, dimension = await score_rows(
        embeddings, test_rows
    )
    correct = int(np.count_nonzero(positive > negative))
    return {
        "model": name,
        "dimension": int(dimension),
        "correct": correct,
        "total": len(test_rows),
        "accuracy": correct / len(test_rows),
        "validation_threshold": threshold,
        "validation_metrics": metrics(val_positive, val_negative, threshold),
        "test_metrics": metrics(positive, negative, threshold),
        "query_ms": round(query_ms, 2),
        "passage_ms": round(passage_ms, 2),
    }


async def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    test_rows = load_triplets(
        download_split(args.split, output / "public_data"), args.limit, args.seed
    )
    validation_rows = load_triplets(
        download_split("val", output / "public_data"),
        args.validation_limit,
        args.seed,
    )
    results = []
    for name in args.models:
        results.append(await measure_model(name, validation_rows, test_rows))
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": args.split,
        "sha256": SHA256[args.split],
        "validation_sha256": SHA256["val"],
        "seed": args.seed,
        "selected_rows": len(test_rows),
        "validation_rows": len(validation_rows),
        "threshold_selection": "max validation F1, then fewer false positives",
        "model_results": results,
        "deployment": "offline_only",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить E5, MiniLM и POTION на одной части ru-HNP"
    )
    parser.add_argument("--output", default="training_runs/ru-hnp")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", nargs="+", choices=("current", *ALTERNATIVES),
        default=("current", *ALTERNATIVES),
    )
    args = parser.parse_args()
    if args.limit < 1 or args.validation_limit < 1:
        parser.error("размеры наборов должны быть положительными")
    if args.split == "val":
        parser.error("тестовый split должен отличаться от val")
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
