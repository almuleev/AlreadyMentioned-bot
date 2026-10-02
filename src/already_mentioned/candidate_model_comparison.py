"""CPU benchmark of candidate encoders on pinned public Russian triplets."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from already_mentioned.offline_training import (
    DATASET,
    REVISION,
    SHA256,
    Triplet,
    download_split,
    load_triplets,
)
from already_mentioned.public_model_comparison import metrics, select_threshold

MODELS = {
    "e5-base": "intfloat/multilingual-e5-base",
    "bge-m3": "BAAI/bge-m3",
    "gte-base": "Alibaba-NLP/gte-multilingual-base",
    "arctic-m": "Snowflake/snowflake-arctic-embed-m-v2.0",
    "rubert-tiny2": "cointegrated/rubert-tiny2",
    "e5-large": "intfloat/multilingual-e5-large",
    "rosberta": "ai-forever/ru-en-RoSBERTa",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "frida": "ai-forever/FRIDA",
    "qwen-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}
REVISIONS = {
    "e5-base": "d128750597153bb5987e10b1c3493a34e5a4502a",
    "bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
    "gte-base": "9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
    "arctic-m": "95c2741480856aa9666782eb4afe11959938017f",
    "rubert-tiny2": "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
    "e5-large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    "rosberta": "89fb1651989adbb1cfcfdedafd7d102951ad0555",
    "mpnet": "4328cf26390c98c5e3c738b4460a05b95f4911f5",
    "frida": "850455b605544a944739b25f81ddf812b6e3d0d5",
    "qwen-0.6b": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}
REMOTE_CODE = {"gte-base", "arctic-m"}


def encode(
    model, name: str, texts: list[str], *, query: bool
) -> np.ndarray:
    options = {
        "batch_size": 4 if name == "qwen-0.6b" else 16,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if name in {"e5-base", "e5-large"}:
        prefix = "query: " if query else "passage: "
        texts = [prefix + text for text in texts]
    elif name == "rosberta":
        texts = ["classification: " + text for text in texts]
    elif name in {"arctic-m", "qwen-0.6b"} and query:
        options["prompt_name"] = "query"
    elif name == "frida":
        options["prompt_name"] = "paraphrase"
    return np.asarray(model.encode(texts, **options), dtype=np.float32)


def score_split(
    model, name: str, rows: tuple[Triplet, ...]
) -> tuple[np.ndarray, np.ndarray]:
    queries = encode(model, name, [row.query for row in rows], query=True)
    positives = encode(model, name, [row.positive for row in rows], query=False)
    negatives = encode(model, name, [row.negative for row in rows], query=False)
    return np.sum(queries * positives, axis=1), np.sum(queries * negatives, axis=1)


def compare(
    name: str, output: Path, val_count: int, test_count: int, seed: int
) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(min(8, torch.get_num_threads()))
    data = output / "public_data"
    validation = load_triplets(download_split("val", data), val_count, seed)
    test = load_triplets(download_split("test", data), test_count, seed)
    start = perf_counter()
    model = SentenceTransformer(
        MODELS[name],
        cache_folder="models",
        revision=REVISIONS[name],
        trust_remote_code=name in REMOTE_CODE,
        device="cpu",
        config_kwargs={"use_memory_efficient_attention": False}
        if name == "arctic-m"
        else None,
    )
    load_seconds = perf_counter() - start
    encode(model, name, [test[0].query], query=True)
    val_positive, val_negative = score_split(model, name, validation)
    threshold = select_threshold(val_positive, val_negative)
    start = perf_counter()
    positive, negative = score_split(model, name, test)
    test_seconds = perf_counter() - start
    start = perf_counter()
    for _ in range(5):
        encode(model, name, [test[0].query], query=True)
    single_query_ms = (perf_counter() - start) * 200
    correct = int(np.count_nonzero(positive > negative))
    report = {
        "model": MODELS[name],
        "model_revision": REVISIONS[name],
        "name": name,
        "dataset": DATASET,
        "revision": REVISION,
        "val_sha256": SHA256["val"],
        "test_sha256": SHA256["test"],
        "seed": seed,
        "val_rows": len(validation),
        "test_rows": len(test),
        "dimension": int(model.get_embedding_dimension()),
        "pairwise_correct": correct,
        "pairwise_total": len(test),
        "validation_threshold": threshold,
        "validation_metrics": metrics(val_positive, val_negative, threshold),
        "test_metrics": metrics(positive, negative, threshold),
        "load_seconds": round(load_seconds, 2),
        "test_encoding_seconds": round(test_seconds, 2),
        "single_query_ms": round(single_query_ms, 2),
        "device": "cpu",
        "deployment": "offline_only",
    }
    target = output / "candidates"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить новую модель на одинаковых val/test ru-HNP"
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", default="training_runs/ru-hnp")
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.val < 1 or args.test < 1:
        parser.error("размеры выборок должны быть положительными")
    print(json.dumps(
        compare(args.model, Path(args.output), args.val, args.test, args.seed),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
