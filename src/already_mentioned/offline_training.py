"""Reproducible, offline-only metric adaptation on public Russian triplets."""

import argparse
import asyncio
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from already_mentioned.evaluation import evaluate
from already_mentioned.evaluation_cases import CASES
from already_mentioned.services.embeddings import (
    MODEL_NAME,
    FastEmbedEmbeddingService,
)
from already_mentioned.services.questions import is_question_candidate

DATASET = "deepvk/ru-HNP"
REVISION = "f43fca040b2635de6163b2b4ed16e5e5907f20f8"
SPLITS = ("train", "val", "test")
SHA256 = {
    "train": "210c0a80185c430aa0e2d62dff9e56ca3a142919d064582161d019069e769a7e",
    "val": "78ca16cc7c449eeeb59fcde44d468aa6524019b7eebbc37c4e03460e0e62adb9",
    "test": "6731cb6779e59b531d542229d7fd0f0ce452cc75722151fef3d70ad43c8d92f7",
}


@dataclass(frozen=True)
class Triplet:
    query: str
    positive: str
    negative: str


def download_split(split: str, directory: Path) -> Path:
    """Fetch a pinned public split only; never read chat or SQLite data."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{split}.parquet"
    if target.exists() and file_sha256(target) == SHA256[split]:
        return target
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{split}.parquet"
    temporary = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if file_sha256(temporary) != SHA256[split]:
            raise ValueError(f"Checksum mismatch for public split {split}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_triplets(path: Path, limit: int, seed: int) -> tuple[Triplet, ...]:
    """Choose deterministic rows and one positive/negative from each row."""
    import pyarrow.parquet as pq

    if limit < 1:
        raise ValueError("limit must be positive")
    table = pq.read_table(path, columns=["query", "pos", "neg"])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(table.num_rows)[:limit]
    selected = table.take(indices).to_pylist()
    return tuple(
        Triplet(row["query"], row["pos"][0], row["neg"][0])
        for row in selected
        if row["query"] and row["pos"] and row["neg"]
    )


async def encode_triplets(
    rows: tuple[Triplet, ...], embeddings: FastEmbedEmbeddingService
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.stack([await embeddings.embed_query(row.query) for row in rows])
    positives = np.stack([await embeddings.embed_passage(row.positive) for row in rows])
    negatives = np.stack([await embeddings.embed_passage(row.negative) for row in rows])
    return queries, positives, negatives


def weighted_score_and_gradient(
    queries: np.ndarray, passages: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine after a shared positive diagonal transform, plus its gradient."""
    q_power = np.sum(weights * queries**2, axis=1)
    p_power = np.sum(weights * passages**2, axis=1)
    denominator = np.sqrt(q_power * p_power)
    scores = np.sum(weights * queries * passages, axis=1) / denominator
    gradients = queries * passages / denominator[:, None]
    gradients -= 0.5 * scores[:, None] * (
        queries**2 / q_power[:, None] + passages**2 / p_power[:, None]
    )
    return scores, gradients


def triplet_accuracy(
    encoded: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray
) -> float:
    queries, positives, negatives = encoded
    positive, _ = weighted_score_and_gradient(queries, positives, weights)
    negative, _ = weighted_score_and_gradient(queries, negatives, weights)
    return float(np.mean(positive > negative))


def fit_adapter(
    encoded: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    epochs: int = 20,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, int, float]:
    """Learn 384 nonnegative feature weights; select using validation only."""
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    queries, positives, negatives = encoded
    weights = np.ones(queries.shape[1], dtype=np.float64)
    best = weights.copy()
    best_accuracy = triplet_accuracy(validation, weights)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        pos_score, pos_gradient = weighted_score_and_gradient(
            queries, positives, weights
        )
        neg_score, neg_gradient = weighted_score_and_gradient(
            queries, negatives, weights
        )
        # Smooth pairwise ranking loss; regularization keeps the adapter near E5.
        logits = np.clip((neg_score - pos_score + 0.05) / 0.1, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        gradient = np.mean(
            probability[:, None] * (neg_gradient - pos_gradient) / 0.1, axis=0
        ) + 0.02 * (weights - 1)
        weights = np.clip(weights - learning_rate * gradient, 0.25, 4.0)
        weights /= weights.mean()
        accuracy = triplet_accuracy(validation, weights)
        if accuracy > best_accuracy:
            best, best_accuracy, best_epoch = weights.copy(), accuracy, epoch
    return best.astype(np.float32), best_epoch, best_accuracy


def transform(vector: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scaled = np.asarray(vector) * np.sqrt(weights)
    return scaled / np.linalg.norm(scaled)


async def fictional_result(
    embeddings: FastEmbedEmbeddingService, weights: np.ndarray
) -> tuple[int, int, int]:
    messages = sorted({c.message for c in CASES if is_question_candidate(c.message)})
    saved = sorted({text for c in CASES for _, text in c.saved})
    queries = {
        text: transform(await embeddings.embed_query(text), weights)
        for text in messages
    }
    passages = {
        text: transform(await embeddings.embed_passage(text), weights)
        for text in saved
    }
    result = evaluate(CASES, queries, passages, 0.88)
    return result.correct, result.wrong, result.missed


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    data = output / "public_data"
    rows = {
        split: load_triplets(download_split(split, data), limit, args.seed)
        for split, limit in zip(SPLITS, (args.train, args.val, args.test), strict=True)
    }
    embeddings = FastEmbedEmbeddingService()
    encoded = {}
    times = {}
    for split in SPLITS:
        start = perf_counter()
        encoded[split] = await encode_triplets(rows[split], embeddings)
        times[split] = round(perf_counter() - start, 2)
    ones = np.ones(encoded["train"][0].shape[1], dtype=np.float32)
    baseline = {split: triplet_accuracy(encoded[split], ones) for split in SPLITS}
    weights, selected_epoch, _ = fit_adapter(
        encoded["train"], encoded["val"], epochs=args.epochs
    )
    after = {split: triplet_accuracy(encoded[split], weights) for split in SPLITS}
    fictional_before = await fictional_result(embeddings, ones)
    fictional_after = await fictional_result(embeddings, weights)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "adapter.npy", weights)
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "dataset_sha256": SHA256,
        "embedding_model": MODEL_NAME,
        "embedding_contract": "query:/passage:, float32, L2-normalized",
        "adapter": "shared positive diagonal cosine weights",
        "seed": args.seed,
        "rows": {split: len(rows[split]) for split in SPLITS},
        "selected_epoch": selected_epoch,
        "triplet_accuracy_before": baseline,
        "triplet_accuracy_after": after,
        "fictional_correct_wrong_missed_before": fictional_before,
        "fictional_correct_wrong_missed_after": fictional_after,
        "encoding_seconds": times,
        "deployment": "offline_only",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Офлайн-обучение адаптера E5 на ru-HNP"
    )
    parser.add_argument("--output", default="training_runs/ru-hnp")
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train, args.val, args.test, args.epochs) < 1:
        parser.error("размеры выборок и число эпох должны быть положительными")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
