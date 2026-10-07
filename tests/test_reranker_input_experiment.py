import argparse
import json
from pathlib import Path

import pytest
from tests.test_search_experiment import write_csv

from already_mentioned import reranker_input_experiment as experiment
from already_mentioned.services.embeddings import FRIDA_CONTRACT
from already_mentioned.threshold_calibration import file_hash


def fixture(tmp_path):
    folders = []
    scenarios = []
    cases = [{"chat": "A", "message_id": str(i), "new_message": text,
              "expected_solution_id": expected} for i, text, expected in
             [(10, "Вход?", "one"), (11, "Выход?", "SILENCE"),
              (12, "Другой вход?", "one"), (13, "Другой выход?", "SILENCE")]]
    split = tmp_path / "split.csv"
    write_csv(split, cases)
    for name in ("main", "proxy"):
        folder = tmp_path / name
        folder.mkdir()
        paths = [folder / "reviewed_cases_full.csv", folder / "bank.csv"]
        write_csv(paths[0], cases)
        write_csv(paths[1], [{
            "chat": "A", "solution_id": "one", "answer_message_id": "1",
            "saved_question": "Где дверь?", "saved_answer": "Слева",
            "historical_proxy_status": "usable_candidate",
        }, {
            "chat": "A", "solution_id": "two", "answer_message_id": "2",
            "saved_question": "Где окно?", "saved_answer": "Справа",
            "historical_proxy_status": "usable_candidate",
        }])
        folders.append(folder)
        scenarios.append({
            "name": name, "inputs": [file_hash(p) for p in paths], "threshold": .458,
            "dataset": {"cases": 4, "solutions": 2, "excluded": 0},
            "details": [{
                "chat": "A", "message_id": int(c["message_id"]),
                "expected": [] if c["expected_solution_id"] == "SILENCE" else ["one"],
                "split": "validation" if index < 2 else "test",
                "filter_passed": True, "best": "one", "score": .6, "gap": .1,
                "candidates": [
                    {"id": "one", "score": .6, "question_score": .6,
                     "answer_score": .3},
                    {"id": "two", "score": .5, "question_score": .1,
                     "answer_score": .5},
                ],
            } for index, c in enumerate(cases)],
        })
    report = tmp_path / "baseline.json"
    report.write_text(json.dumps({
        "format_version": 1, "name": "frida-baseline", "ranking": "baseline",
        "contract": FRIDA_CONTRACT, "scenarios": scenarios,
        "split_source": file_hash(split),
        "filter_sha256": file_hash(
            Path(experiment.__file__).parent / "services/questions.py"
        ),
    }), encoding="utf-8")
    return argparse.Namespace(candidates_from=report, datasets=folders,
                              split_cases=split, output=tmp_path / "results")


class Scorer:
    batch_size = 1
    max_length = 512
    total_pairs = 0
    truncated_pairs = 0

    def __init__(self, late_value=.9):
        self.calls = []
        self.late_value = late_value

    def predict(self, query, passages):
        self.calls.append((query, passages))
        self.total_pairs += len(passages)
        return [self.late_value if query.startswith("Другой") else
                .8 if query == "Вход?" else .2 for _ in passages]


def test_cached_ablation_preserves_split_and_calibrates_main_early_only(tmp_path):
    args = fixture(tmp_path)
    scorer = Scorer()
    reports = experiment.run(args, scorer)
    # Four queries, two candidates, three formats; proxy reuses every score.
    assert len(scorer.calls) == 24
    assert all(len(passages) == 1 for _, passages in scorer.calls)
    for strategy, report in reports.items():
        main, proxy = report["scenarios"]
        assert main["threshold"] == proxy["threshold"]
        assert [d["split"] for d in main["details"]] == [
            "validation", "validation", "test", "test"
        ]
        assert report["contract"] == FRIDA_CONTRACT
        assert report["timing"]["frida_loaded"] is False
        assert report["reused_candidates"] == file_hash(args.candidates_from)
        saved = (args.output / f"{strategy}.json").read_text(encoding="utf-8")
        assert "Где дверь" not in saved
    args.output = tmp_path / "changed_late"
    changed = experiment.run(args, Scorer(late_value=.01))
    assert {s: r["scenarios"][0]["threshold"] for s, r in reports.items()} == {
        s: r["scenarios"][0]["threshold"] for s, r in changed.items()
    }
    # Scores changed on late cases, demonstrating the threshold check isn't vacuous.
    assert changed["rerank_pair_single"]["scenarios"][0]["test"] != reports[
        "rerank_pair_single"]["scenarios"][0]["test"]


@pytest.mark.parametrize(
    "corruption", ["source", "contract", "missing_candidates", "score", "batch"]
)
def test_invalid_reuse_rejected_before_results(tmp_path, corruption):
    args = fixture(tmp_path)
    scorer = Scorer()
    report = json.loads(args.candidates_from.read_text(encoding="utf-8"))
    if corruption == "source":
        for folder in args.datasets:
            (folder / "reviewed_cases_full.csv").write_text("changed", encoding="utf-8")
    elif corruption == "contract":
        report["contract"] = {"model": "other"}
    elif corruption == "missing_candidates":
        report["scenarios"][0]["details"][0]["candidates"] = []
    elif corruption == "score":
        report["scenarios"][0]["details"][0]["candidates"][0]["score"] = .9
    else:
        scorer.batch_size = 4
    args.candidates_from.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        experiment.run(args, scorer)
    assert not args.output.exists()


def test_formats_and_balanced_rank_do_not_treat_short_answer_as_full_context():
    solution = {"saved_question": "Какая версия ОС?", "saved_answer": "11"}
    assert experiment.passage(solution, "answer") == "11"
    assert experiment.passage(solution, "plain_pair") == "Какая версия ОС?\n11"
    assert experiment.passage(solution, "labelled_pair") == (
        "Вопрос: Какая версия ОС?\nОтвет: 11"
    )
    candidates = [{"id": "a", "score": .8, "question_score": .1, "answer_score": .8},
                  {"id": "b", "score": .6, "question_score": .6, "answer_score": .6}]
    scores = {m: [.2, .7] for m in experiment.FORMATS}
    assert experiment.variants(candidates, scores)["balanced_top5"][0]["id"] == "b"


def test_alternative_scorer_contract_and_selected_formats(tmp_path):
    args = fixture(tmp_path)
    args.strategies = ["rerank_pair_single", "hybrid_top5"]
    scorer = Scorer()
    scorer.model = "alternative/model"
    scorer.revision = "pinned-revision"
    scorer.file = "onnx/model.onnx"
    scorer.experiment_prefix = "frida-alternative"
    reports = experiment.run(args, scorer)
    assert set(reports) == set(args.strategies)
    assert len(scorer.calls) == 8  # Four queries, two candidates; shared with proxy.
    for strategy, report in reports.items():
        assert report["name"] == f"frida-alternative-{strategy}"
        assert report["reranker"]["model"] == scorer.model
        assert report["reranker"]["revision"] == scorer.revision
        assert report["reranker"]["file"] == scorer.file
        assert set(report["timing"]["ablation_inference"]) == {"labelled_pair"}


@pytest.mark.parametrize("strategies", [["unknown"], ["hybrid_top5"] * 2])
def test_invalid_strategy_selection_rejected(tmp_path, strategies):
    args = fixture(tmp_path)
    args.strategies = strategies
    with pytest.raises(ValueError, match="стратегий"):
        experiment.run(args, Scorer())
    assert not args.output.exists()


def test_recall_priority_beta_is_recorded_and_rejects_invalid_value(tmp_path):
    args = fixture(tmp_path)
    args.beta = 2
    result = experiment.run(args, Scorer())
    assert result["hybrid_top5"]["beta"] == 2
    assert "F_beta_2_" in result["hybrid_top5"]["calibration"]
    args.output = tmp_path / "bad_beta"
    args.beta = float("nan")
    with pytest.raises(ValueError, match="Beta"):
        experiment.run(args, Scorer())


@pytest.mark.parametrize("text", [
    "Здравствуйте! Я не успеваю на автобус на 10:00",
    "У нас задерживается рейс, мы не успеем прилететь",
    "Возможно, не успею на шаттл на 11:00",
])
def test_offline_transport_filter_accepts_two_cue_problem_reports(text):
    assert experiment.experimental_transport_filter(text)


@pytest.mark.parametrize("text", [
    "Не успеваю закончить работу", "Трансфер отправится в 11:00",
    "/не успеваю на автобус", "Здравствуйте", "Рейс вылетает в 10",
])
def test_offline_transport_filter_keeps_plain_announcements_and_commands_out(text):
    assert not experiment.experimental_transport_filter(text)
