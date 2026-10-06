import argparse
import csv
import sys

import numpy as np
import pytest

from already_mentioned import threshold_calibration as calibration
from already_mentioned.historical_evaluation import HistoricalCase, Outcome


def outcome(number, score, positive):
    case = HistoricalCase(
        "chat", number, "Как войти?", frozenset({"one"}) if positive else frozenset()
    )
    return Outcome(case, True, "one", score, None)


def test_small_recall_preference_changes_f1_tie_without_using_later_cases():
    early = (
        outcome(1, 0.8, True),
        outcome(2, 0.6, True),
        outcome(3, 0.7, False),
        outcome(4, 0.65, False),
    )
    assert calibration.select_threshold(early, beta=1) == 0.8
    assert calibration.select_threshold(early, beta=1.1) == 0.6


def test_calibration_requires_positives_and_finite_beta():
    with pytest.raises(ValueError, match="положительные"):
        calibration.select_threshold((outcome(1, 0.8, False),))
    for beta in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            calibration.select_threshold((outcome(1, 0.8, True),), beta)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.asyncio
async def test_frozen_split_unknowns_and_common_threshold_in_both_scenarios(tmp_path):
    rows = [
        {
            "chat": "chat",
            "message_id": str(i),
            "new_message": "Как войти?",
            "expected_solution_id": label,
        }
        for i, label in enumerate(("one", "UNRESOLVED", "SILENCE", "one", "SILENCE"), 1)
    ]
    split = tmp_path / "split.csv"
    write_csv(split, rows)
    folders = []
    for name in ("main", "proxy"):
        folder = tmp_path / name
        folder.mkdir()
        write_csv(folder / "reviewed_cases_full.csv", rows)
        write_csv(
            folder / "chat1_inferred_solution_bank.csv",
            [
                {
                    "chat": "chat",
                    "solution_id": "one",
                    "answer_message_id": "0",
                    "saved_question": "Как зайти?",
                    "saved_answer": "Откройте сайт.",
                    "historical_proxy_status": "usable_candidate",
                }
            ],
        )
        folders.append(folder)

    class FakeEmbeddings:
        async def embed_query(self, text):
            return np.array([0.6, 0.8], dtype=np.float32)

        async def embed_passage(self, text):
            return np.array([1.0, 0.0], dtype=np.float32)

    args = argparse.Namespace(datasets=folders, split_cases=split, beta=1.1)
    report = await calibration.run(args, FakeEmbeddings())
    assert report["selection"]["dataset"] == "main_validation_only"
    assert report["selection"]["threshold"] == 0.6
    for scenario in report["scenarios"]:
        assert scenario["excluded"] == 1
        # Three early source IDs stay early even after ID 2 becomes unknown.
        assert scenario["validation_cases"] == scenario["test_cases"] == 2
        assert [
            r["message_id"] for r in scenario["details"] if r["split"] == "validation"
        ] == [1, 3]
        assert scenario["after"]["test"]["correct"] == 1
    assert report["timing"]["unique_queries"] == 1


def test_cli_rejects_overwriting_report_before_loading_model(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("preserve", encoding="utf8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibration",
            "--datasets",
            "a",
            "b",
            "--split-cases",
            "old.csv",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        calibration, "FridaEmbeddingService", lambda: pytest.fail("Модель")
    )
    with pytest.raises(SystemExit) as error:
        calibration.main()
    assert error.value.code == 2
    assert output.read_text(encoding="utf8") == "preserve"
