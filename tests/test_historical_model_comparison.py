"""Compare isolated historical banks without models or private messages."""

import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from already_mentioned import historical_model_comparison as comparison
from already_mentioned.historical_evaluation import HistoricalCase, HistoricalSolution


def test_threshold_uses_early_cases_and_counts_filter_and_ranking_misses():
    cases = (
        HistoricalCase("A", 10, "Как войти?", frozenset({"login"})),
        HistoricalCase("A", 20, "Как войти иначе?", frozenset({"login"})),
        HistoricalCase("A", 30, "Где вход?", frozenset()),
        HistoricalCase("A", 40, "Нужен доступ", frozenset({"login"})),
        HistoricalCase("A", 50, "Когда вход?", frozenset()),
    )
    solutions = (HistoricalSolution("A", "login", 1, "q", "a"),)
    queries = {
        c.message: np.array([s, np.sqrt(1 - s**2)])
        for c, s in zip(cases, (0.8, 0.9, 0.85, 0.99, 0.99), strict=True)
    }
    passages = {t: np.array([1., 0.]) for t in ("q", "a")}
    result = comparison.evaluate_vectors(cases, solutions, queries, passages)
    assert result["threshold"] == pytest.approx(0.8)
    # A high-scoring later negative is not used to raise the threshold.
    assert result["test"]["wrong"] == 1
    assert result["test"]["filter_missed"] == 1
    assert result["test"]["correct_leader_without_filter"] == 1
    assert result["test"]["missed"] == 1


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0], delimiter=";")
        writer.writeheader()
        writer.writerows(records)


def test_cli_keeps_scenarios_separate_and_reports_hashes_without_source_text(
    tmp_path, monkeypatch,
):
    folders = [tmp_path / name for name in ("main", "proxy")]
    for folder in folders:
        folder.mkdir()
        write_csv(folder / "reviewed_cases_full.csv", [{
            "chat": "A", "message_id": i, "new_message": "Как войти?",
            "expected_solution_id": "login" if folder.name == "main" else "SILENCE",
        } for i in range(10, 15)])
        write_csv(folder / "chat1_inferred_solution_bank.csv", [{
            "chat": "A", "solution_id": "login", "answer_message_id": 1,
            "saved_question": "Форма входа", "saved_answer": "Откройте форму",
            "historical_proxy_status": "usable_candidate",
        }])
    calls = []

    def fake_loader(name, cache, threads):
        calls.append((name, threads))
        return (lambda texts, query: np.array([[1., 0.]] * len(texts)),
                {"model": "fake", "normalization": "L2"}, 0.)

    memory = SimpleNamespace(rss=100, peak_wset=200)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda: SimpleNamespace(memory_info=lambda: memory),
    ))
    monkeypatch.setattr(comparison, "load_encoder", fake_loader)
    output = tmp_path / "reports"
    monkeypatch.setattr(sys, "argv", [
        "comparison", "--model", "current", "--datasets",
        *map(str, folders), "--output", str(output),
    ])
    comparison.main()
    report_text = (output / "current.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    main, proxy = report["scenarios"]
    assert calls == [("current", 2)]  # Shared encoder, independent calibration.
    assert main["test"]["correct"] == 2
    assert proxy["threshold"] > 1 and proxy["test"]["correct_silence"] == 2
    assert main["inputs"][0]["sha256"] != proxy["inputs"][0]["sha256"]
    assert "Как войти?" not in report_text and "Откройте форму" not in report_text
    assert [r["solutions"] for r in report["timing"]["ranking"]] == [15, 50, 100, 200]
    assert report["contract"]["dimension"] == 2


def test_cli_invalid_threads_fails_before_model_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "comparison", "--model", "current", "--datasets", "missing",
        "--threads", "0",
    ])
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["e5-base", "frida"])
def test_cached_encoder_pins_revision_and_uses_retrieval_prompts(name, monkeypatch):
    loading, encoding = [], []

    class Model:
        max_seq_length = 512
        prompts = {"search_query": "search_query: ",
                   "search_document": "search_document: "}

        def __init__(self, source, **options):
            loading.append((source, options))

        def encode(self, texts, **options):
            encoding.append((texts, options))
            return np.array([[1., 0.]] * len(texts))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: None,
    ))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=Model,
    ))
    encoder, contract, _ = comparison.load_encoder(name, "unused", 2)
    encoder(["тест"], True)
    encoder(["ответ"], False)
    assert loading[0][1]["revision"] == comparison.REVISIONS[name]
    assert loading[0][1]["local_files_only"] is True
    if name == "e5-base":
        assert [call[0] for call in encoding] == [["query: тест"], ["passage: ответ"]]
    else:
        assert [call[1]["prompt_name"] for call in encoding] == [
            "search_query", "search_document",
        ]
        assert contract["query_rule"] == "search_query: "
        assert contract["passage_rule"] == "search_document: "
