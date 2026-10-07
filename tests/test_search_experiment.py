import argparse
import csv
import json

import numpy as np
import pytest

from already_mentioned import search_experiment as experiment
from already_mentioned.historical_evaluation import HistoricalCase, HistoricalSolution


def test_retrieval_excludes_future_other_chat_and_keeps_distinct_pairs():
    case = HistoricalCase("A", 10, "Запрос?", frozenset({"one"}))
    bank = (
        HistoricalSolution("A", "one", 1, "q", "a"),
        HistoricalSolution("A", "future", 10, "q", "a"),
        HistoricalSolution("B", "other", 1, "q", "a"),
        HistoricalSolution("A", "two", 2, "q2", "a2"),
    )
    queries = {case.message: np.array([1., 0.])}
    passages = {"q": np.array([1., 0.]), "a": np.array([0., 1.]),
                "q2": np.array([0., 1.]), "a2": np.array([1., 0.])}
    pairs = {experiment.pair_text(s): np.array([1., 0.]) for s in bank}
    ranking = experiment.retrieve(case, bank, queries, passages, pairs)
    assert [r["id"] for r in ranking] == ["one", "two"]
    assert ranking[0]["question_score"] == 1
    assert ranking[0]["answer_score"] == 0
    outcome = experiment.to_outcome(case, ranking, "baseline")
    assert outcome.best_id == "one" and outcome.gap == 0
    ranking[0]["rerank_score"] = 0.2
    ranking[1]["rerank_score"] = 0.8
    assert experiment.to_outcome(case, ranking, "rerank_top5").best_id == "two"


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], delimiter=";")
        writer.writeheader()
        writer.writerows(records)


@pytest.mark.asyncio
async def test_experiment_freezes_split_preserves_sources_and_uses_main_threshold(
    tmp_path, monkeypatch,
):
    records = [{
        "chat": "A", "message_id": str(i), "new_message": "Где вход?",
        "expected_solution_id": label,
    } for i, label in enumerate(("one", "UNRESOLVED", "SILENCE", "one", "SILENCE"), 10)]
    split = tmp_path / "split.csv"
    write_csv(split, records)
    folders = []
    for name in ("main", "proxy"):
        folder = tmp_path / name
        folder.mkdir()
        write_csv(folder / "reviewed_cases_full.csv", records)
        write_csv(folder / "chat1_inferred_solution_bank.csv", [{
            "chat": "A", "solution_id": "one", "answer_message_id": "1",
            "saved_question": "Открыть кабинет?", "saved_answer": "В меню профиля.",
            "historical_proxy_status": "usable_candidate",
        }])
        folders.append(folder)

    class Embeddings:
        async def embed_query(self, text):
            return np.array([0.6, 0.8])

        async def embed_passage(self, text):
            return np.array([1., 0.])

    class Reranker:
        def predict(self, query, passages):
            return [0.75] * len(passages)

    calls = []
    real_select = experiment.select_threshold

    def select(early, beta):
        calls.append([o.case.message_id for o in early])
        return real_select(early, beta)

    monkeypatch.setattr(experiment, "select_threshold", select)
    args = argparse.Namespace(datasets=folders, split_cases=split,
                              output=tmp_path / "reports")
    reports = await experiment.run(args, Embeddings(), Reranker())
    assert calls == [[10, 12]] * 3  # Only main early cases, frozen before exclusion.
    for strategy, report in reports.items():
        main, proxy = report["scenarios"]
        assert main["threshold"] == proxy["threshold"]
        assert main["dataset"]["excluded"] == 1
        assert [r["message_id"] for r in main["details"]
                if r["split"] == "test"] == [13, 14]
        assert main["candidate_coverage"]["test"]["with_expected_top5"] == 1
        text = (args.output / f"{strategy}.json").read_text(encoding="utf-8")
        assert "В меню профиля" not in text and "Где вход" not in text
        assert json.loads(text)["contract"]["precision"] == "fp32"

    guarded = reports["rerank_recall_guard"]
    assert guarded["scenarios"][0]["validation"]["correct"] >= (
        reports["baseline"]["scenarios"][0]["validation"]["correct"]
    )
    # Changing only late scores/labels cannot change the guarded early threshold.
    altered = json.loads(json.dumps(reports["rerank_top5"]))
    altered_base = json.loads(json.dumps(reports["baseline"]))
    for report in (altered, altered_base):
        for scenario in report["scenarios"]:
            for row in scenario["details"]:
                if row["split"] == "test":
                    row["score"] = 0.999
                    row["expected"] = []
    changed = experiment.recall_guard_report(altered_base, altered)
    assert changed["scenarios"][0]["threshold"] == guarded["scenarios"][0]["threshold"]
