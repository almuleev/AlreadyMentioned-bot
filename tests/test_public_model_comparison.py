"""The public benchmark evaluates all models on exactly the same rows."""

import argparse
import json

import numpy as np
import pytest

from already_mentioned import public_model_comparison as comparison
from already_mentioned.offline_training import Triplet


@pytest.mark.asyncio
async def test_measure_model_counts_ranked_triplets(monkeypatch) -> None:
    class FakeEmbeddings:
        async def embed_query(self, text):
            return np.array([1.0, 0.0])

        async def embed_passage(self, text):
            return np.array([1.0, 0.0] if text == "p" else [0.0, 1.0])

    monkeypatch.setattr(comparison, "create_model", lambda name: FakeEmbeddings())
    rows = (Triplet("q", "p", "n"),)
    result = await comparison.measure_model("current", rows, rows)
    assert result["correct"] == result["total"] == 1
    assert result["dimension"] == 2
    assert result["accuracy"] == 1.0
    assert result["test_metrics"]["tp"] == 1
    assert result["test_metrics"]["fp"] == 0


def test_threshold_is_selected_from_validation_only() -> None:
    validation_positive = np.array([0.8, 0.9])
    validation_negative = np.array([0.2, 0.3])
    threshold = comparison.select_threshold(
        validation_positive, validation_negative
    )
    assert 0.3 < threshold <= 0.8
    measured = comparison.metrics(
        np.array([0.7, 0.85]), np.array([0.4, 0.1]), threshold
    )
    assert measured == {
        "tp": 1, "fp": 0, "fn": 1, "tn": 2,
        "precision": 1.0, "recall": 0.5, "f1": 2 / 3,
        "false_positive_rate": 0.0,
    }


@pytest.mark.asyncio
async def test_run_uses_same_holdout_for_every_model(tmp_path, monkeypatch) -> None:
    rows = (Triplet("q1", "p1", "n1"), Triplet("q2", "p2", "n2"))
    monkeypatch.setattr(comparison, "download_split", lambda split, data: tmp_path)
    monkeypatch.setattr(comparison, "load_triplets", lambda path, limit, seed: rows)
    seen = []

    async def fake_measure(name, validation_rows, supplied_rows):
        seen.append((name, validation_rows, supplied_rows))
        return {"model": name, "correct": 2, "total": 2}

    monkeypatch.setattr(comparison, "measure_model", fake_measure)
    args = argparse.Namespace(
        output=str(tmp_path), split="test", limit=2, validation_limit=2, seed=42,
        models=("current", "minilm", "potion"),
    )
    report = await comparison.run(args)
    assert [name for name, _, _ in seen] == list(args.models)
    assert all(
        validation is rows and supplied is rows
        for _, validation, supplied in seen
    )
    assert report["selected_rows"] == 2
    assert json.loads(
        (tmp_path / "model_comparison.json").read_text(encoding="utf-8")
    ) == report
