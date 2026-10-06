"""Offline, chronological comparison of cached encoders on labelled CSV banks."""

import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from time import sleep as time_sleep

import numpy as np

from already_mentioned.candidate_model_comparison import (
    MODELS,
    REMOTE_CODE,
    REVISIONS,
    encode,
)
from already_mentioned.historical_evaluation import (
    Outcome,
    calibrate,
    load_dataset,
    metrics,
    score_dataset,
    split_cases,
)
from already_mentioned.model_comparison import ALTERNATIVES
from already_mentioned.services.embeddings import (
    MODEL_NAME,
    FastEmbedEmbeddingService,
    normalize_vector,
)
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.similarity import cosine_similarity

FAST = {"e5-small": MODEL_NAME, **ALTERNATIVES}


def quantize_linears(model):
    """Convert only this in-memory model; cached checkpoint remains untouched."""
    import torch

    supported = torch.backends.quantized.supported_engines
    engine = next((e for e in ("fbgemm", "onednn") if e in supported), None)
    if engine is None:
        raise ValueError("Нет поддерживаемого CPU backend для INT8")
    torch.backends.quantized.engine = engine
    original = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    if not original:
        raise ValueError("В модели нет линейных слоёв для квантования")
    start = perf_counter()
    model.eval()
    result = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
    )
    converted = sum(
        isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in result.modules()
    )
    if converted != original:
        raise ValueError("Не все линейные слои были квантованы")
    # Remaining embeddings can keep the complete safetensors mapping resident.
    # Give the FP32 parameters their own storage without changing their values.
    copied_bytes = 0
    with torch.no_grad():
        for parameter in result.parameters():
            copied_bytes += parameter.numel() * parameter.element_size()
            parameter.data = parameter.detach().clone()
    gc.collect()
    return result, {
        "method": "torch_dynamic_linear_int8", "engine": engine,
        "torch_version": torch.__version__, "converted_linears": converted,
        "other_modules": "unchanged_fp32", "seconds": perf_counter() - start,
        "fp32_storage_copied_bytes": copied_bytes,
        "checkpoint_modified": False,
    }


def load_encoder(name, cache, threads):
    """Use only existing cache; one model per process, no production vectors."""
    started = perf_counter()
    name = "frida" if name == "current" else name
    int8 = name == "frida-int8"
    name = "frida" if int8 else name
    contract = {
        "model": FAST.get(name, MODELS.get(name)),
        "normalization": "L2",
        "dtype": "float32",
        "device": "cpu",
        "threads": threads,
        "query_rule": "plain",
        "passage_rule": "plain",
    }
    if name in FAST:
        from fastembed import TextEmbedding

        if name == "e5-small":
            service = FastEmbedEmbeddingService(cache, threads=threads)
            service._encode_sync("query: пробный вопрос")
            model = service._model
            contract.update(query_rule="query: ", passage_rule="passage: ")
        else:
            model = TextEmbedding(
                model_name=FAST[name], cache_dir=str(cache), threads=threads,
                local_files_only=True, providers=["CPUExecutionProvider"],
            )
        contract["backend"] = "fastembed"
        backend = model.model
        contract["artifact"] = str(backend._model_dir)
        contract["artifact_source"] = backend.model_description.sources.hf

        def run(texts, query):
            rule = contract["query_rule" if query else "passage_rule"]
            return np.asarray(
                list(model.embed([rule + t for t in texts], batch_size=8)),
                dtype=np.float32,
            )

    else:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        model = SentenceTransformer(
            MODELS[name], cache_folder=str(cache), revision=REVISIONS[name],
            trust_remote_code=name in REMOTE_CODE, device="cpu",
            local_files_only=True,
            config_kwargs={"use_memory_efficient_attention": False}
            if name == "arctic-m" else None,
        )
        contract.update(
            backend="sentence_transformers", revision=REVISIONS[name],
            max_sequence_length=model.max_seq_length,
        )
        if int8:
            model, contract["quantization"] = quantize_linears(model)
        if name in {"e5-base", "e5-large"}:
            contract.update(query_rule="query: ", passage_rule="passage: ")
        elif name == "rosberta":
            contract.update(
                query_rule="classification: ", passage_rule="classification: ",
            )
        elif name == "frida":
            contract.update(
                query_rule=model.prompts["search_query"],
                passage_rule=model.prompts["search_document"],
            )
        elif name in {"arctic-m", "qwen-0.6b"}:
            contract["query_rule"] = model.prompts["query"]

        def run(texts, query):
            if name == "frida":
                # The model card specifies these prompts for answer retrieval.
                return np.asarray(model.encode(
                    texts, prompt_name="search_query" if query else "search_document",
                    batch_size=16, normalize_embeddings=True,
                    show_progress_bar=False,
                ), dtype=np.float32)
            return encode(model, name, texts, query=query)

    return run, contract, perf_counter() - started


def evaluate_vectors(cases, solutions, queries, passages):
    # Rejected positive cases are encoded to distinguish the filter from ranking.
    # Rejected negatives cannot produce a bot reply and need no model inference.
    selected = tuple(
        c for c in cases if is_question_candidate(c.message) or c.expected
    )
    scored = score_dataset(selected, solutions, queries, passages)["both"]
    by_id = {(o.case.chat, o.case.message_id): o for o in scored}
    values = tuple(
        by_id.get((c.chat, c.message_id), Outcome(c, False, None, -2.0, None))
        for c in cases
    )
    validation_ids, test_ids = split_cases(cases, 0.6)
    validation = tuple(
        o for o in values if (o.case.chat, o.case.message_id) in validation_ids
    )
    test = tuple(o for o in values if (o.case.chat, o.case.message_id) in test_ids)
    threshold, _ = calibrate(validation, (0.0,))
    return {
        "threshold": threshold,
        "validation": metrics(validation, threshold),
        "test": metrics(test, threshold),
        "all": metrics(values, threshold),
        "fixed_0_88": {
            "test": metrics(test, 0.88), "all": metrics(values, 0.88),
        },
        "test_by_chat": {
            chat: metrics(tuple(o for o in test if o.case.chat == chat), threshold)
            for chat in sorted({c.chat for c in cases})
        },
        "details": [
            {
                "chat": o.case.chat, "message_id": o.case.message_id,
                "split": "validation"
                if (o.case.chat, o.case.message_id) in validation_ids else "test",
                "filter_passed": o.passed, "expected": sorted(o.case.expected),
                "best": o.best_id, "score": o.score, "gap": o.gap,
            }
            for o in values
        ],
    }


def ranking_timing(query, passage_vectors):
    # Repeat real vectors for size measurements, not quality evaluation.
    vectors = list(passage_vectors.values())
    result = []
    for size in (15, 50, 100, 200):
        pairs = [(vectors[i % len(vectors)], vectors[(i + 1) % len(vectors)])
                 for i in range(size)]
        durations = []
        for _ in range(10):
            start = perf_counter()
            max(max(cosine_similarity(query, q), cosine_similarity(query, a))
                for q, a in pairs)
            durations.append((perf_counter() - start) * 1000)
        result.append({
            "solutions": size, "mean_ms": float(np.mean(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
        })
    return result


def measure_serial_load(work, cycles, *, clock=perf_counter, sleep=time_sleep):
    """Five requests per five seconds through one serial model worker."""
    start = clock()
    waits, latencies, cycle_waits = [], [], []
    for cycle in range(cycles):
        arrival = start + cycle * 5
        if clock() < arrival:
            sleep(arrival - clock())
        cycle_waits.append(max(0., clock() - arrival) * 1000)
        for operation in work:
            waits.append(max(0., clock() - arrival) * 1000)
            operation()
            latencies.append((clock() - arrival) * 1000)
    return {
        "chats": len(work), "cycles": cycles, "requests": len(latencies),
        "interval_seconds_per_chat": 5, "all_messages_pass_filter": True,
        "mean_response_ms": float(np.mean(latencies)),
        "p95_response_ms": float(np.percentile(latencies, 95)),
        "max_response_ms": max(latencies), "max_queue_wait_ms": max(waits),
        "first_cycle_initial_wait_ms": cycle_waits[0],
        "last_cycle_initial_wait_ms": cycle_waits[-1],
        "duration_seconds": clock() - start,
        "excludes_telegram_sqlite_os_limits": True,
    }


def run(args):
    import psutil

    datasets = []
    for folder in args.datasets:
        paths = [folder / "reviewed_cases_full.csv", *sorted(
            folder.glob("chat*_inferred_solution_bank.csv")
        )]
        cases, solutions, excluded = load_dataset(paths[0], paths[1:])
        datasets.append((folder, paths, cases, solutions, excluded))
    load_cases = {}
    if args.load_cycles:
        for case in datasets[0][2]:
            available = any(s.chat == case.chat
                            and s.answer_message_id < case.message_id
                            for s in datasets[0][3])
            if available and is_question_candidate(case.message) and (
                case.chat not in load_cases
                or len(case.message) > len(load_cases[case.chat].message)
            ):
                load_cases[case.chat] = case
        if len(load_cases) < 5:
            raise ValueError("Для проверки нагрузки нужны вопросы пяти чатов")
    query_texts = sorted({
        c.message for _, _, cases, _, _ in datasets for c in cases
        if is_question_candidate(c.message) or c.expected
    })
    passage_texts = sorted({
        text for _, _, _, bank, _ in datasets for s in bank
        for text in (s.question, s.answer)
    })
    encoder, contract, load_seconds = load_encoder(
        args.model, args.cache_dir, args.threads,
    )
    process = psutil.Process()
    memory_phases = {"after_load_rss_mib": process.memory_info().rss / 2**20}
    encoder(query_texts[:1], True)
    queries, durations = {}, []
    max_query_rss = 0.0
    # Single calls measure the bot's online workload, without batch acceleration.
    for number, text in enumerate(query_texts, 1):
        start = perf_counter()
        queries[text] = normalize_vector(encoder([text], True)[0])
        durations.append((perf_counter() - start) * 1000)
        max_query_rss = max(max_query_rss, process.memory_info().rss / 2**20)
        if number % 200 == 0:
            print(f"{args.model}: запросы {number}/{len(query_texts)}", flush=True)
    memory_phases.update(
        after_queries_rss_mib=process.memory_info().rss / 2**20,
        max_rss_after_single_query_mib=max_query_rss,
    )
    start = perf_counter()
    passages = {}
    for offset in range(0, len(passage_texts), args.passage_batch_size):
        texts = passage_texts[offset:offset + args.passage_batch_size]
        passages.update({text: normalize_vector(vector)
                         for text, vector in zip(texts, encoder(texts, False),
                                                 strict=True)})
    passage_seconds = perf_counter() - start
    memory_phases["after_passages_rss_mib"] = process.memory_info().rss / 2**20
    contract["dimension"] = len(next(iter(queries.values())))
    contract["encoding_batch_sizes"] = {
        "query": 1, "passage": args.passage_batch_size,
    }
    if any(len(v) != contract["dimension"] for v in passages.values()):
        raise ValueError("Размерности запросов и решений не совпали")
    report = {
        "format_version": 1, "name": args.model, "contract": contract,
        "annotation_status": "provisional_not_owner_confirmed",
        "ranking": "max_question_answer_cosine_same_chat_past_answers_only",
        "calibration": "full_F1_earlier_60_percent_per_chat_no_margin",
        "host": {"platform": platform.platform(), "cpu": platform.processor()},
        "timing": {
            "load_seconds": load_seconds, "unique_queries": len(queries),
            "unique_passages": len(passages),
            "single_query_mean_ms": float(np.mean(durations)),
            "single_query_p95_ms": float(np.percentile(durations, 95)),
            "passage_batch_mean_ms": passage_seconds * 1000 / len(passages),
            "passage_batch_size": args.passage_batch_size,
            "ranking": ranking_timing(next(iter(queries.values())), passages),
            "excludes_sqlite_telegram": True,
        },
        "scenarios": [],
    }
    if args.load_cycles:
        work, payloads = [], []
        for chat in sorted(load_cases)[:5]:
            case = load_cases[chat]
            bank = [s for s in datasets[0][3] if s.chat == chat
                    and s.answer_message_id < case.message_id]
            if not bank:
                raise ValueError("Нет предшествующих решений для проверки нагрузки")
            pairs = [(passages[s.question], passages[s.answer]) for s in bank]
            pairs = [pairs[i % len(pairs)] for i in range(200)]

            def operation(text=case.message, pairs=pairs):
                query = normalize_vector(encoder([text], True)[0])
                max(max(cosine_similarity(query, q), cosine_similarity(query, a))
                    for q, a in pairs)

            work.append(operation)
            payloads.append({"chars": len(case.message), "solutions": 200,
                             "sha256": hashlib.sha256(
                                 case.message.encode("utf-8")).hexdigest()})
        print("Проверка очереди: пять чатов, сообщения каждые пять секунд...",
              flush=True)
        report["load_test"] = measure_serial_load(work, args.load_cycles)
        report["load_test"]["payloads"] = payloads
        memory_phases["after_load_test_rss_mib"] = process.memory_info().rss / 2**20
    for folder, paths, cases, solutions, excluded in datasets:
        result = evaluate_vectors(cases, solutions, queries, passages)
        report["scenarios"].append({
            "name": folder.name,
            "dataset": {"cases": len(cases), "solutions": len(solutions),
                        "positives": sum(bool(c.expected) for c in cases),
                        "excluded": excluded},
            "inputs": [{"name": p.name,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in paths],
            **result,
        })
        m = result["test"]
        print(f"{args.model} / {folder.name}: поздние — верных {m['correct']}, "
              f"ошибочных {m['wrong']}, пропусков {m['missed']}", flush=True)
    memory = psutil.Process().memory_info()
    report["process_memory"] = {
        "rss_mib": memory.rss / 2**20,
        "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
        "includes_python_libraries_vectors": True,
        "phases": memory_phases,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{args.model}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"Отчёт: {target}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Сравнить локальные модели на исторических CSV без Telegram и БД"
    )
    parser.add_argument(
        "--model", required=True, choices=("current", *FAST, *MODELS, "frida-int8"),
    )
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("training_runs/bank_v2_model_comparison"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--load-cycles", type=int, default=0)
    parser.add_argument("--passage-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Число потоков должно быть положительным")
    if not 0 <= args.load_cycles <= 12:
        parser.error("Число циклов нагрузки должно быть от 0 до 12")
    if args.passage_batch_size < 1:
        parser.error("Размер пакета сохранённых текстов должен быть положительным")
    if (args.output / f"{args.model}.json").resolve() in {
        (folder / "reviewed_cases_full.csv").resolve() for folder in args.datasets
    }:
        parser.error("Отчёт не должен заменять входной файл")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run(args)


if __name__ == "__main__":
    main()
