"""Candidate benchmark contracts without loading optional model dependencies."""

import json
import sys

import numpy as np

from already_mentioned import candidate_model_comparison as candidates


def test_encode_respects_model_specific_query_contract() -> None:
    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return [[1.0, 0.0] for _ in texts]

    model = FakeModel()
    result = candidates.encode(model, "e5-base", ["вопрос"], query=True)
    np.testing.assert_array_equal(result, [[1.0, 0.0]])
    assert model.calls[-1][0] == ["query: вопрос"]
    candidates.encode(model, "e5-base", ["вопрос"], query=False)
    assert model.calls[-1][0] == ["passage: вопрос"]
    candidates.encode(model, "arctic-m", ["вопрос"], query=True)
    assert model.calls[-1][1]["prompt_name"] == "query"
    candidates.encode(model, "arctic-m", ["вопрос"], query=False)
    assert "prompt_name" not in model.calls[-1][1]
    candidates.encode(model, "bge-m3", ["вопрос"], query=True)
    assert model.calls[-1][0] == ["вопрос"]
    candidates.encode(model, "rosberta", ["вопрос"], query=True)
    assert model.calls[-1][0] == ["classification: вопрос"]
    candidates.encode(model, "frida", ["вопрос"], query=False)
    assert model.calls[-1][1]["prompt_name"] == "paraphrase"
    candidates.encode(model, "qwen-0.6b", ["вопрос"], query=True)
    assert model.calls[-1][1]["prompt_name"] == "query"


def test_score_split_uses_same_query_for_both_candidates() -> None:
    class FakeModel:
        def encode(self, texts, **kwargs):
            mapping = {"q": [1.0, 0.0], "p": [1.0, 0.0], "n": [0.0, 1.0]}
            return [mapping[text] for text in texts]

    from already_mentioned.offline_training import Triplet

    positive, negative = candidates.score_split(
        FakeModel(), "bge-m3", (Triplet("q", "p", "n"),)
    )
    np.testing.assert_array_equal(positive, [1.0])
    np.testing.assert_array_equal(negative, [0.0])


def test_cli_selects_one_model_without_loading_it(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        candidates, "compare", lambda name, output, val, test, seed: {
            "name": name, "val_rows": val, "test_rows": test, "seed": seed
        },
    )
    monkeypatch.setattr(sys, "argv", [
        "candidate_model_comparison", "--model", "gte-base", "--val", "20",
        "--test", "30",
    ])
    candidates.main()
    assert json.loads(capsys.readouterr().out) == {
        "name": "gte-base", "val_rows": 20, "test_rows": 30, "seed": 42
    }
