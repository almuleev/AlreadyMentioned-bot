"""Historical experiments preserve chat/time boundaries and evaluate abstention."""

import csv
import json
import sys

import numpy as np
import pytest

from already_mentioned import historical_evaluation as evaluation
from already_mentioned.services.embeddings import EmbeddingService


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dataset(tmp_path):
    cases = tmp_path / "cases.csv"
    bank = tmp_path / "bank.csv"
    write_csv(
        cases,
        ("chat", "message_id", "new_message", "expected_solution_id"),
        [
            {
                "chat": "A",
                "message_id": 10,
                "new_message": "Как войти в кабинет?",
                "expected_solution_id": "login|alternate",
            },
            {
                "chat": "A",
                "message_id": 20,
                "new_message": "Где оплатить билет?",
                "expected_solution_id": "SILENCE",
            },
            {
                "chat": "A",
                "message_id": 30,
                "new_message": "Где форма входа?",
                "expected_solution_id": "login",
            },
            {
                "chat": "A",
                "message_id": 40,
                "new_message": "Как вернуть билет?",
                "expected_solution_id": "SILENCE",
            },
            {
                "chat": "A",
                "message_id": 50,
                "new_message": "Когда будет ответ?",
                "expected_solution_id": "UNRESOLVED",
            },
        ],
    )
    write_csv(
        bank,
        (
            "chat",
            "solution_id",
            "answer_message_id",
            "saved_question",
            "saved_answer",
            "historical_proxy_status",
        ),
        [
            {
                "chat": "A",
                "solution_id": "login",
                "answer_message_id": 2,
                "saved_question": "Вход в кабинет",
                "saved_answer": "Откройте форму входа",
                "historical_proxy_status": "usable_candidate",
            },
            {
                "chat": "A",
                "solution_id": "alternate",
                "answer_message_id": 3,
                "saved_question": "Авторизация",
                "saved_answer": "Нажмите кнопку входа",
                "historical_proxy_status": "usable_candidate",
            },
            {
                "chat": "B",
                "solution_id": "other",
                "answer_message_id": 1,
                "saved_question": "Другой чат",
                "saved_answer": "Другой ответ",
                "historical_proxy_status": "usable_candidate",
            },
        ],
    )
    return cases, bank


def test_loader_excludes_uncertain_and_accepts_multiple_correct_answers(dataset):
    cases_path, bank_path = dataset
    cases, bank, excluded = evaluation.load_dataset(cases_path, [bank_path])
    assert len(cases) == 4 and len(bank) == 3 and excluded == 1
    assert cases[0].expected == {"login", "alternate"}
    assert cases[1].expected == frozenset()


@pytest.mark.parametrize("label", ["other", "missing", "", "future"])
def test_loader_rejects_missing_other_chat_or_future_solution(dataset, label):
    cases_path, bank_path = dataset
    text = cases_path.read_text(encoding="utf-8-sig")
    cases_path.write_text(text.replace("login|alternate", label), encoding="utf-8-sig")
    if label == "future":
        text = bank_path.read_text(encoding="utf-8-sig")
        bank_path.write_text(
            text.replace("login;2;", "future;10;"), encoding="utf-8-sig"
        )
    with pytest.raises(ValueError):
        evaluation.load_dataset(cases_path, [bank_path])


def test_split_is_chronological_per_chat_and_ignores_input_order():
    cases = tuple(
        evaluation.HistoricalCase(chat, number, "Как войти?", frozenset())
        for chat in ("B", "A")
        for number in (40, 10, 30, 20)
    )
    validation, test = evaluation.split_cases(cases, 0.5)
    assert validation == {("A", 10), ("A", 20), ("B", 10), ("B", 20)}
    assert test == {("A", 30), ("A", 40), ("B", 30), ("B", 40)}


def test_scoring_scopes_chat_and_time_and_gap_uses_distinct_solutions():
    case = evaluation.HistoricalCase("A", 10, "Как войти?", frozenset({"good"}))
    solutions = (
        evaluation.HistoricalSolution("B", "other", 1, "perfect", "perfect"),
        evaluation.HistoricalSolution("A", "future", 10, "perfect", "perfect"),
        evaluation.HistoricalSolution("A", "good", 2, "perfect", "near"),
        evaluation.HistoricalSolution("A", "second", 3, "half", "half"),
    )
    vectors = {
        "perfect": np.array([1.0, 0.0]),
        "near": np.array([0.99, np.sqrt(1 - 0.99**2)]),
        "half": np.array([0.5, np.sqrt(0.75)]),
    }
    scores = evaluation.score_dataset(
        (case,), solutions, {case.message: vectors["perfect"]}, vectors
    )
    outcome = scores["both"][0]
    assert outcome.best_id == "good"
    assert outcome.gap == pytest.approx(0.5)
    assert evaluation.chosen_id(outcome, 0.8, 0.1) == "good"
    assert scores["answer"][0].score == pytest.approx(0.99)
    only = evaluation.score_dataset(
        (case,), solutions[:3], {case.message: vectors["perfect"]}, vectors
    )["both"][0]
    assert only.gap is None
    assert evaluation.chosen_id(only, 0.8, 0.1) is None
    assert evaluation.chosen_id(only, 0.8, 0) == "good"


def test_metrics_attribute_misses_and_count_wrong_positive_as_two_errors():
    def outcome(number, expected, best, score, gap=0.2, passed=True):
        case = evaluation.HistoricalCase("A", number, "Как войти?", frozenset(expected))
        return evaluation.Outcome(case, passed, best, score, gap)

    values = (
        outcome(1, {"good", "alternate"}, "alternate", 0.95),
        outcome(2, {"good"}, "wrong", 0.96),
        outcome(3, (), "wrong", 0.97),
        outcome(4, {"good"}, "good", 0.7),
        outcome(5, {"good"}, "good", 0.99, passed=False),
        outcome(6, {"good"}, "good", 0.99, gap=0.01),
    )
    result = evaluation.metrics(values, 0.8, 0.1)
    assert (result["correct"], result["wrong"], result["missed"]) == (1, 2, 4)
    assert result["false_prompt"] == result["wrong_solution"] == 1
    assert (
        result["filter_missed"]
        == result["wrong_leader"]
        == result["below_threshold"]
        == result["margin_missed"]
        == 1
    )
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 5)
    assert result["f1"] == pytest.approx(2 / 8)


def test_calibration_does_not_consult_test_and_includes_wrong_leader_misses():
    positive = evaluation.HistoricalCase("A", 1, "Как войти?", frozenset({"good"}))
    negative = evaluation.HistoricalCase("A", 2, "Где оплатить?", frozenset())
    validation = (
        evaluation.Outcome(positive, True, "good", 0.8, 0.1),
        evaluation.Outcome(positive, True, "wrong", 0.9, 0.1),
        evaluation.Outcome(negative, True, "wrong", 0.85, 0.1),
    )
    threshold, margin = evaluation.calibrate(validation, (0,))
    assert threshold == 0.8 and margin == 0
    # An unseen high-scoring negative must remain an error after calibration.
    test = (evaluation.Outcome(negative, True, "wrong", 0.99, 0.1),)
    assert evaluation.metrics(test, threshold)["wrong"] == 1


class FakeEmbeddings(EmbeddingService):
    def __init__(self, _cache):
        pass

    async def embed_query(self, text):
        return np.array(
            [1, 0] if "вход" in text or "войти" in text else [0, 1], dtype=np.float32
        )

    async def embed_passage(self, text):
        return np.array([1, 0], dtype=np.float32)


def test_cli_produces_identified_report_without_model_or_database(
    dataset, tmp_path, monkeypatch, capsys
):
    cases, bank = dataset
    output = tmp_path / "report.json"
    monkeypatch.setattr(evaluation, "FastEmbedEmbeddingService", FakeEmbeddings)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "historical_evaluation",
            "--cases",
            str(cases),
            "--banks",
            str(bank),
            "--output",
            str(output),
            "--margins",
            "0",
            "0.1",
        ],
    )
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset"] == {
        "cases": 4,
        "positives": 2,
        "excluded": 1,
        "solutions": 3,
    }
    assert len(report["experiments"]) == 9 and len(report["details"]) == 12
    assert report["split"]["validation"] == report["split"]["test"] == 2
    assert len(report["inputs"][0]["sha256"]) == 64
    assert all("message" not in row for row in report["details"])
    assert "threshold_and_margin" in capsys.readouterr().out


def test_cli_rejects_nonfinite_margin_before_loading_model(dataset, monkeypatch):
    cases, bank = dataset
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "historical_evaluation",
            "--cases",
            str(cases),
            "--banks",
            str(bank),
            "--margins",
            "nan",
        ],
    )
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
