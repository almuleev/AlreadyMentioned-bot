"""Compare isolated historical banks without models or private messages."""

import csv
import json
import sys
from contextlib import nullcontext
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
    calls, encoded = [], []

    def fake_loader(name, cache, threads):
        calls.append((name, threads))

        def encoder(texts, query):
            encoded.append((len(texts), query))
            return np.array([[1., 0.]] * len(texts))

        return encoder, {"model": "fake", "normalization": "L2"}, 0.

    memory = SimpleNamespace(rss=100, peak_wset=200)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda: SimpleNamespace(memory_info=lambda: memory),
    ))
    monkeypatch.setattr(comparison, "load_encoder", fake_loader)
    output = tmp_path / "reports"
    monkeypatch.setattr(sys, "argv", [
        "comparison", "--model", "current", "--datasets",
        *map(str, folders), "--output", str(output), "--passage-batch-size", "1",
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
    assert report["contract"]["encoding_batch_sizes"] == {"query": 1, "passage": 1}
    assert all(size == 1 for size, _ in encoded)
    assert report["process_memory"]["phases"]["after_load_rss_mib"] == 100 / 2**20
    assert "load_test" not in report


@pytest.mark.parametrize("option,value", [
    ("--threads", "0"), ("--load-cycles", "13"), ("--passage-batch-size", "0"),
])
def test_cli_invalid_limits_fail_before_model_loading(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", [
        "comparison", "--model", "current", "--datasets", "missing",
        option, value,
    ])
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["e5-base", "frida", "frida-int8"])
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
    quantized = []

    def fake_quantizer(model):
        quantized.append(model)
        return model, {"method": "fake_int8"}

    monkeypatch.setattr(comparison, "quantize_linears", fake_quantizer)
    encoder, contract, _ = comparison.load_encoder(name, "unused", 2)
    encoder(["тест"], True)
    encoder(["ответ"], False)
    source = "frida" if name == "frida-int8" else name
    assert loading[0][1]["revision"] == comparison.REVISIONS[source]
    assert loading[0][1]["local_files_only"] is True
    assert bool(quantized) == (name == "frida-int8")
    if quantized:
        assert contract["quantization"] == {"method": "fake_int8"}
    if name == "e5-base":
        assert [call[0] for call in encoding] == [["query: тест"], ["passage: ответ"]]
    else:
        assert [call[1]["prompt_name"] for call in encoding] == [
            "search_query", "search_document",
        ]
        assert contract["query_rule"] == "search_query: "
        assert contract["passage_rule"] == "search_document: "


def test_quantization_validates_engine_and_converts_all_linears_in_memory(monkeypatch):
    class Linear:
        pass

    class QuantizedLinear:
        pass

    class Parameter:
        def __init__(self):
            self.data = np.array([1., 2.], dtype=np.float32)

        def numel(self):
            return self.data.size

        def element_size(self):
            return self.data.itemsize

        def detach(self):
            return self

        def clone(self):
            return self.data.copy()

    class Model:
        def __init__(self):
            self.layers = [Linear(), Linear()]
            self.evaluated = False
            self.parameter = Parameter()

        def modules(self):
            return self.layers

        def eval(self):
            self.evaluated = True

        def parameters(self):
            return [self.parameter]

    calls = []

    def convert(model, spec, *, dtype, inplace):
        calls.append((spec, dtype, inplace, model.evaluated))
        model.layers = [QuantizedLinear(), QuantizedLinear()]
        return model

    backend = SimpleNamespace(supported_engines=["onednn"])
    torch = SimpleNamespace(
        __version__="test", nn=SimpleNamespace(Linear=Linear), qint8="qint8",
        no_grad=nullcontext,
        backends=SimpleNamespace(quantized=backend),
        ao=SimpleNamespace(
            quantization=SimpleNamespace(quantize_dynamic=convert),
            nn=SimpleNamespace(quantized=SimpleNamespace(
                dynamic=SimpleNamespace(Linear=QuantizedLinear),
            )),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    model = Model()
    original_storage = model.parameter.data
    result, contract = comparison.quantize_linears(model)
    assert result is model
    assert calls == [({Linear}, "qint8", True, True)]
    assert contract["engine"] == "onednn"
    assert contract["converted_linears"] == 2
    assert contract["checkpoint_modified"] is False
    assert contract["other_modules"] == "unchanged_fp32"
    assert contract["fp32_storage_copied_bytes"] == 8
    assert model.parameter.data is not original_storage
    np.testing.assert_array_equal(model.parameter.data, original_storage)
    backend.supported_engines = ["none"]
    with pytest.raises(ValueError, match="backend"):
        comparison.quantize_linears(Model())
    backend.supported_engines = ["onednn"]
    empty = Model()
    empty.layers = []
    with pytest.raises(ValueError, match="линейных"):
        comparison.quantize_linears(empty)
    torch.ao.quantization.quantize_dynamic = lambda model, *args, **kwargs: model
    with pytest.raises(ValueError, match="Не все"):
        comparison.quantize_linears(Model())


def test_load_measurement_includes_backlog_in_response_time():
    value = [0.]

    def operation():
        value[0] += 2

    def sleep(seconds):
        value[0] += seconds

    result = comparison.measure_serial_load(
        [operation] * 5, 2, clock=lambda: value[0], sleep=sleep,
    )
    assert result["requests"] == 10
    assert result["first_cycle_initial_wait_ms"] == 0
    assert result["last_cycle_initial_wait_ms"] == 5000
    assert result["max_response_ms"] == 15000
    assert result["max_queue_wait_ms"] == 13000
    assert result["duration_seconds"] == 20


def test_load_measurement_waits_for_arrivals_when_worker_keeps_up():
    value = [0.]

    def operation():
        value[0] += 0.1

    def sleep(seconds):
        value[0] += seconds

    result = comparison.measure_serial_load(
        [operation] * 5, 3, clock=lambda: value[0], sleep=sleep,
    )
    assert result["last_cycle_initial_wait_ms"] == 0
    assert result["max_response_ms"] == pytest.approx(500)
    assert result["duration_seconds"] == pytest.approx(10.5)
