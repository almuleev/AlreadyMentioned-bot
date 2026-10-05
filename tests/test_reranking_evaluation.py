"""Reranking experiments preserve scope, candidates, calibration and labels."""

import asyncio
import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from already_mentioned import historical_evaluation
from already_mentioned import reranking_evaluation as evaluation
from already_mentioned.historical_evaluation import (
    HistoricalCase,
    HistoricalSolution,
    Outcome,
)
from already_mentioned.services.embeddings import EmbeddingService


def test_shortlist_excludes_other_chat_and_future_and_keeps_e5_order():
    case = HistoricalCase("A", 10, "Как войти?", frozenset({"good"}))
    solutions = tuple(
        HistoricalSolution(chat, name, number, text, text)
        for chat, name, number, text in (
            ("B", "other", 1, "perfect"),
            ("A", "future", 10, "perfect"),
            ("A", "good", 2, "half"),
            ("A", "bad", 3, "perfect"),
        )
    )
    vectors = {"perfect": np.array([1.0, 0]), "half": np.array([0.5, np.sqrt(0.75)])}
    candidates = evaluation.shortlist(
        case, solutions, {case.message: vectors["perfect"]}, vectors
    )
    assert [s.id for s, _ in candidates] == ["bad", "good"]


def test_pair_scoring_can_recover_second_candidate_and_does_not_invent_one():
    case = HistoricalCase("A", 10, "Как войти?", frozenset({"good"}))
    candidates = tuple(
        (HistoricalSolution("A", name, 2, "Сохранённый вопрос", answer), score)
        for name, answer, score in (("bad", "Оплата", 0.9), ("good", "Вход", 0.7))
    )
    seen = []

    class Scorer:
        def predict(self, query, passages):
            seen.append((query, passages))
            return [0.9 if "Вход" in text else 0.1 for text in passages]

    result = evaluation.rerank(case, candidates, Scorer(), "answer")
    assert result.best_id == "good" and result.score == 0.9
    assert seen[0] == ("Как войти?", ["Оплата", "Вход"])
    assert evaluation.rerank(case, candidates[:1], Scorer(), "answer").best_id == "bad"
    evaluation.rerank(case, candidates, Scorer(), "question_answer")
    assert seen[-1][1][0] == "Вопрос: Сохранённый вопрос\nОтвет: Оплата"


def test_filter_and_empty_shortlist_skip_pair_inference():
    class Scorer:
        def predict(self, *_):
            raise AssertionError("Модель не должна вызываться")

    case = HistoricalCase("A", 10, "Спасибо за помощь", frozenset())
    candidate = HistoricalSolution("A", "one", 2, "Вопрос", "Ответ")
    assert (
        evaluation.rerank(case, ((candidate, 0.9),), Scorer(), "answer").best_id is None
    )
    question = HistoricalCase("A", 10, "Как войти?", frozenset())
    assert evaluation.rerank(question, (), Scorer(), "answer").best_id is None


def test_threshold_selection_ignores_late_case():
    positive = HistoricalCase("A", 1, "Как войти?", frozenset({"good"}))
    negative = HistoricalCase("A", 2, "Как вернуть оплату?", frozenset())
    late = HistoricalCase("A", 3, "Как вернуть билет?", frozenset())
    outcomes = (
        Outcome(positive, True, "good", 0.8, None),
        Outcome(negative, True, "good", 0.4, None),
        Outcome(late, True, "good", 0.9, None),
    )
    report = evaluation.report_experiment(outcomes, frozenset({("A", 1), ("A", 2)}))
    assert report["threshold"] == 0.8
    assert report["validation"]["wrong"] == 0 and report["test"]["wrong"] == 1


class FakeEmbeddings(EmbeddingService):
    def __init__(self, _cache, **_kwargs):
        pass

    async def embed_query(self, text):
        return np.array([1.0, 0], dtype=np.float32)

    async def embed_passage(self, text):
        return np.array(
            [1.0, 0] if "вход" in text.lower() else [0, 1], dtype=np.float32
        )


class FakeScorer:
    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, query, passages):
        return [
            0.9 if "вход" in query.lower() and "вход" in text.lower() else 0.1
            for text in passages
        ]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dataset(tmp_path):
    cases, bank = tmp_path / "cases.csv", tmp_path / "bank.csv"
    write_csv(
        cases,
        [
            {
                "chat": "A",
                "message_id": n,
                "new_message": text,
                "expected_solution_id": expected,
            }
            for n, text, expected in (
                (10, "Где вход в кабинет?", "login"),
                (20, "Как оплатить билет?", "SILENCE"),
                (30, "Где вход в профиль?", "login"),
                (40, "Где билет?", "SILENCE"),
            )
        ],
    )
    write_csv(
        bank,
        [
            {
                "chat": "A",
                "solution_id": name,
                "answer_message_id": 2,
                "saved_question": question,
                "saved_answer": answer,
                "historical_proxy_status": "usable_candidate",
            }
            for name, question, answer in (
                ("login", "Как открыть вход?", "Откройте форму входа"),
                ("other", "Где урок?", "Запись урока в профиле"),
            )
        ],
    )
    return cases, bank


def test_cli_reports_coverage_quality_timing_and_human_queue(
    dataset, tmp_path, monkeypatch, capsys
):
    cases, bank = dataset
    original = cases.read_bytes()
    output, review = tmp_path / "report.json", tmp_path / "review.csv"
    monkeypatch.setattr(evaluation, "FastEmbedEmbeddingService", FakeEmbeddings)
    monkeypatch.setattr(evaluation, "OnnxReranker", FakeScorer)
    monkeypatch.setattr(
        evaluation, "memory_snapshot", lambda: {"rss_mb": 10, "peak_rss_mb": 12}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reranking_evaluation",
            "--cases",
            str(cases),
            "--banks",
            str(bank),
            "--output",
            str(output),
            "--review-output",
            str(review),
            "--top-k",
            "1",
            "2",
        ],
    )
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["experiments"]) == 4
    assert report["baseline_fixed"]["all"]["correct"] == 2
    assert report["experiments"][0]["candidate_coverage"]["all"] == {
        "positives": 2,
        "retrieved": 2,
    }
    assert report["reranker"]["threads"] == 2 and report["reranker"]["revision"]
    assert all("new_message" not in detail for detail in report["details"])
    rows = list(csv.DictReader(review.open(encoding="utf-8-sig"), delimiter=";"))
    assert rows and all(row["owner_decision"] == "" for row in rows)
    assert cases.read_bytes() == original
    assert "поздняя часть" in capsys.readouterr().out


def test_cli_rejects_overwriting_inputs_before_loading_model(dataset, monkeypatch):
    cases, bank = dataset
    original = cases.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reranking_evaluation",
            "--cases",
            str(cases),
            "--banks",
            str(bank),
            "--review-output",
            str(cases),
        ],
    )
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2 and cases.read_bytes() == original


def test_empty_review_still_has_header(tmp_path):
    path = tmp_path / "empty.csv"
    assert evaluation.write_review(path, (), (), (), 0.8, 10) == 0
    assert "owner_decision" in path.read_text(encoding="utf-8-sig")


def test_negative_review_rotates_chats_and_prioritizes_high_scores(tmp_path):
    cases = tuple(
        Outcome(
            HistoricalCase(chat, n, "Как войти?", frozenset()), True, None, score, None
        )
        for chat, n, score in (("A", 1, 0.9), ("A", 2, 0.99), ("B", 1, 0.95))
    )
    # Use one dummy candidate per chat so the baseline does issue a prompt.
    outcomes = tuple(Outcome(o.case, True, "one", o.score, None) for o in cases)
    path = tmp_path / "review.csv"
    evaluation.write_review(path, outcomes, None, (), 0.88, 3)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig"), delimiter=";"))
    assert [(r["chat"], r["message_id"]) for r in rows] == [
        ("A", "2"),
        ("B", "1"),
        ("A", "1"),
    ]
    assert all(r["reranker_score"] == "" for r in rows)


@pytest.mark.asyncio
async def test_review_only_reuses_verified_report_and_rejects_stale_inputs(
    dataset, tmp_path, monkeypatch, capsys
):
    cases, bank = dataset
    source = tmp_path / "historical.json"
    args = SimpleNamespace(
        cases=cases,
        banks=[bank],
        output=source,
        threshold=0.88,
        margins=[0.0],
        validation_fraction=0.6,
    )
    await historical_evaluation.run(args, FakeEmbeddings(None))
    output, review = tmp_path / "review_metadata.json", tmp_path / "review.csv"
    argv = [
        "reranking_evaluation",
        "--cases",
        str(cases),
        "--banks",
        str(bank),
        "--prepare-review-from",
        str(source),
        "--output",
        str(output),
        "--review-output",
        str(review),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    def fail(*_args, **_kwargs):
        raise AssertionError("Review must never instantiate a model")

    monkeypatch.setattr(evaluation, "OnnxReranker", fail)
    monkeypatch.setattr(evaluation, "FastEmbedEmbeddingService", fail)
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "review_only_no_new_inference"
    assert report["baseline_reused"]["correct"] == 2
    assert "Новых метрик" in capsys.readouterr().out
    rows = list(csv.DictReader(review.open(encoding="utf-8-sig"), delimiter=";"))
    assert len(rows) == 4 and all(row["reranker_decision"] == "" for row in rows)
    original_review = review.read_bytes()
    cases.write_bytes(cases.read_bytes() + b"\n")
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 1 and review.read_bytes() == original_review


@pytest.mark.asyncio
async def test_review_only_rejects_duplicate_results(dataset, tmp_path):
    cases, bank = dataset
    source = tmp_path / "historical.json"
    args = SimpleNamespace(
        cases=cases,
        banks=[bank],
        output=source,
        threshold=0.88,
        margins=[0.0],
        validation_fraction=0.6,
    )
    saved = await historical_evaluation.run(args, FakeEmbeddings(None))
    saved["details"].append(
        next(d for d in saved["details"] if d["strategy"] == "both")
    )
    source.write_text(json.dumps(saved), encoding="utf-8")
    args.prepare_review_from = source
    args.review_output = tmp_path / "review.csv"
    args.review_limit = 60
    with pytest.raises(ValueError, match="ровно один"):
        evaluation.prepare_review(args)
    assert not args.review_output.exists()


def test_cached_top1_scores_real_pairs_without_embedding_service(
    dataset, tmp_path, monkeypatch
):
    cases, bank = dataset
    source = tmp_path / "historical.json"
    args = SimpleNamespace(
        cases=cases,
        banks=[bank],
        output=source,
        threshold=0.88,
        margins=[0.0],
        validation_fraction=0.6,
    )
    asyncio.run(historical_evaluation.run(args, FakeEmbeddings(None)))
    output, review = tmp_path / "result.json", tmp_path / "review.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reranking_evaluation",
            "--cases",
            str(cases),
            "--banks",
            str(bank),
            "--candidates-from",
            str(source),
            "--top-k",
            "1",
            "--output",
            str(output),
            "--review-output",
            str(review),
        ],
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("Cached candidates must not load E5")

    monkeypatch.setattr(evaluation, "FastEmbedEmbeddingService", fail)
    monkeypatch.setattr(evaluation, "OnnxReranker", FakeScorer)
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["retrieval_source"] == "verified_historical_top1"
    assert report["models_coexist"] is False and report["encoding_timing"] is None
    assert report["embedding_threads"] is None
    assert report["experiments"][0]["all"]["correct"] == 2
    assert "with_both_models" not in report["process_memory"]


def test_cached_candidates_reject_unsupported_top_k_before_model_load(
    dataset, tmp_path, monkeypatch
):
    cases, bank = dataset
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reranking_evaluation",
            "--cases",
            str(cases),
            "--banks",
            str(bank),
            "--candidates-from",
            str(tmp_path / "not_loaded.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
