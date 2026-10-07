import argparse
import json

import numpy as np
import pytest
from tests.test_search_experiment import write_csv

from already_mentioned import release_validation as validation


@pytest.mark.asyncio
async def test_production_replay_keeps_unknown_in_execution_but_out_of_quality(
    tmp_path, monkeypatch
):
    write_csv(
        tmp_path / "reviewed_cases_full.csv",
        [
            {
                "chat": "A",
                "message_id": "10",
                "new_message": "Как попасть?",
                "expected_solution_id": "one",
            },
            {
                "chat": "A",
                "message_id": "11",
                "new_message": "Спасибо всем",
                "expected_solution_id": "SILENCE",
            },
            {
                "chat": "A",
                "message_id": "12",
                "new_message": "Где находится вход?",
                "expected_solution_id": "UNRESOLVED",
            },
        ],
    )
    write_csv(
        tmp_path / "chatA_inferred_solution_bank.csv",
        [
            {
                "chat": "A",
                "solution_id": "one",
                "answer_message_id": "1",
                "saved_question": "Как попасть?",
                "saved_answer": "Через дверь",
                "historical_proxy_status": "usable_candidate",
            }
        ],
    )

    class Encoder:
        async def embed_query(self, text):
            return np.array([1.0, 0.0], dtype=np.float32)

        async def embed_passage(self, text):
            return np.array([1.0, 0.0], dtype=np.float32)

    class Scorer:
        def __init__(self, **kwargs):
            assert kwargs == {"local_only": True, "batch_size": 1}

        def predict(self, query, passages):
            return [0.8] * len(passages)

    monkeypatch.setattr(validation, "FridaEmbeddingService", Encoder)
    monkeypatch.setattr(validation, "OnnxReranker", Scorer)
    args = argparse.Namespace(
        dataset=tmp_path,
        output=tmp_path / "result.json",
        mode="hybrid",
        baseline_threshold=0.458,
        hybrid_threshold=0.5,
    )
    await validation.run(args)
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["history_rows"] == 3 and report["labelled"] == 2
    assert report["excluded_unknown"] == 1 and report["all"]["correct"] == 1
    assert len(report["details"]) == 3
    assert report["details"][2]["best"] == "one"
    assert report["details"][2]["labelled"] is False
    assert report["threshold_calibrated_here"] is False
    assert report["candidate_leaders_for_abstentions_recorded"] is False
    assert "below_threshold" not in report["all"]
