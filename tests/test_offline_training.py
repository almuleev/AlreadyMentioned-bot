"""Offline adaptation stays reproducible and independent of bot storage."""

import json
import sys

import numpy as np
import pytest

from already_mentioned import offline_training as training


def test_fit_adapter_improves_separable_triplets() -> None:
    queries = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    positives = np.array([[1.0, -0.4], [1.0, -0.3]], dtype=np.float64)
    negatives = np.array([[-0.4, 1.0], [-0.3, 1.0]], dtype=np.float64)
    train = (queries, positives, negatives)
    weights, epoch, accuracy = training.fit_adapter(
        train, train, epochs=100, learning_rate=1.0
    )
    assert epoch > 0
    assert accuracy == 1.0
    assert weights[0] > weights[1]


def test_triplet_accuracy_and_transform_preserve_identity() -> None:
    identity = np.ones(2, dtype=np.float32)
    vector = np.array([0.6, 0.8], dtype=np.float32)
    np.testing.assert_allclose(training.transform(vector, identity), vector)
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    positive = query.copy()
    negative = query[::-1].copy()
    assert training.triplet_accuracy((query, positive, negative), identity) == 1.0


def test_load_triplets_is_deterministic(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table({
            "query": [f"q{i}" for i in range(10)],
            "pos": [[f"p{i}"] for i in range(10)],
            "neg": [[f"n{i}"] for i in range(10)],
        }),
        path,
    )
    first = training.load_triplets(path, 4, 42)
    assert first == training.load_triplets(path, 4, 42)
    assert len(first) == 4
    assert len({row.query for row in first}) == 4


def test_cli_reports_baseline_and_holdout_without_network(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        training, "download_split", lambda split, directory: tmp_path / split
    )
    monkeypatch.setattr(
        training,
        "load_triplets",
        lambda path, limit, seed: tuple(
            training.Triplet("q", "p", "n") for _ in range(limit)
        ),
    )

    async def fake_encode(rows, embeddings):
        count = len(rows)
        return (
            np.tile([1.0, 0.0], (count, 1)),
            np.tile([1.0, 0.0], (count, 1)),
            np.tile([0.0, 1.0], (count, 1)),
        )

    async def fake_fictional(embeddings, weights):
        return 4, 0, 4

    monkeypatch.setattr(training, "encode_triplets", fake_encode)
    monkeypatch.setattr(training, "fictional_result", fake_fictional)
    monkeypatch.setattr(training, "FastEmbedEmbeddingService", lambda: object())
    monkeypatch.setattr(sys, "argv", ["offline_training", "--output", str(tmp_path)])
    training.main()
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == {"train": 500, "val": 100, "test": 100}
    assert report["triplet_accuracy_before"]["test"] == 1.0
    assert report["deployment"] == "offline_only"
    assert (tmp_path / "adapter.npy").exists()
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report
