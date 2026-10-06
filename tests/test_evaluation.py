"""The offline report counts decisions at each stage without a model download."""

import sys

import numpy as np

from already_mentioned import evaluation, model_comparison
from already_mentioned.evaluation_cases import CASES, Case
from already_mentioned.services.embeddings import EmbeddingService


class FakeEmbeddings(EmbeddingService):
    async def embed_query(self, text: str) -> np.ndarray:
        return (
            np.array([1, 0], dtype=np.float32)
            if "вход" in text
            else np.array([0, 1], dtype=np.float32)
        )

    async def embed_passage(self, text: str) -> np.ndarray:
        return (
            np.array([1, 0], dtype=np.float32)
            if "вход" in text
            else np.array([0, 1], dtype=np.float32)
        )


def test_dataset_is_explicit_and_fictional() -> None:
    assert len(CASES) >= 12
    assert len({case.name for case in CASES}) == len(CASES)
    assert all(case.saved for case in CASES)
    assert all(
        case.expected is None or case.expected in {item[0] for item in case.saved}
        for case in CASES
    )


def test_counts_filter_ranking_and_wrong_answer() -> None:
    cases = (
        Case("верно", "Как войти в кабинет?", (("вход", "вход"),), "вход"),
        Case("фильтр", "Адрес кабинета известен", (("вход", "вход"),), "вход"),
        Case("порог", "Как открыть кабинет?", (("вход", "вход"),), "вход"),
        Case("ложная", "Как вернуть оплату?", (("вход", "вход"),), None),
        Case(
            "неверное",
            "Как сменить пароль?",
            (("вход", "вход"), ("пароль", "пароль")),
            "пароль",
        ),
        Case("молчание", "Спасибо за помощь", (("вход", "вход"),), None),
    )
    queries = {
        "Как войти в кабинет?": np.array([1.0, 0.0]),
        "Как открыть кабинет?": np.array([0.5, 0.866]),
        "Как вернуть оплату?": np.array([1.0, 0.0]),
        "Как сменить пароль?": np.array([1.0, 0.0]),
    }
    passages = {"вход": np.array([1.0, 0.0]), "пароль": np.array([0.0, 1.0])}
    result = evaluation.evaluate(cases, queries, passages, 0.88)
    assert (result.correct, result.wrong, result.missed, result.correct_silence) == (
        1,
        2,
        3,
        1,
    )
    assert (result.filter_missed, result.ranking_missed) == (1, 2)
    assert (result.wrong_best, result.below_threshold) == (1, 1)
    assert evaluation.evaluate(cases, queries, passages, 0.4).correct == 2


def test_cli_reports_multiple_thresholds_without_database_or_model(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(evaluation, "FridaEmbeddingService", FakeEmbeddings)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "--thresholds",
            "0.8",
            "0.9",
            "--sizes",
            "1",
            "4",
            "--repeats",
            "1",
        ],
    )
    evaluation.main()
    output = capsys.readouterr().out
    assert "Порог 0.80:" in output and "Порог 0.90:" in output
    assert "ошибочных подсказок" in output
    assert "фильтр" in output and "ранжирование/порог" in output
    assert "неверный лидер" in output and "верный лидер ниже порога" in output
    assert "1 решений:" in output and "4 решений:" in output


def test_benchmark_uses_requested_sizes() -> None:
    vector = np.array([1.0, 0.0])
    rows = evaluation.benchmark(vector, (vector,), (1, 7), 2)
    assert [size for size, _ in rows] == [1, 7]
    assert all(elapsed >= 0 for _, elapsed in rows)
    assert evaluation.benchmark_filter(("Как войти?", "Спасибо"), 2) >= 0


def test_model_comparison_cli_uses_same_cases_without_download(
    monkeypatch, capsys
) -> None:
    class FakeAlternative(FakeEmbeddings):
        def __init__(self, _name: str) -> None:
            pass

    monkeypatch.setattr(model_comparison, "FridaEmbeddingService", FakeEmbeddings)
    monkeypatch.setattr(model_comparison, "PlainTextFastEmbed", FakeAlternative)
    monkeypatch.setattr(
        sys, "argv", ["model_comparison", "--models", "current", "minilm", "--details"]
    )
    model_comparison.main()
    output = capsys.readouterr().out
    assert "current: 2 измерений" in output
    assert "minilm: 2 измерений" in output
    assert "подобранный на этом наборе" in output
    assert "фильтр-без-маркера" in output
