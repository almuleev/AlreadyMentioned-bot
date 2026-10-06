"""Read-only production FRIDA calibration on explicitly supplied CSV scenarios."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from already_mentioned.historical_evaluation import (
    HistoricalCase,
    Outcome,
    load_dataset,
    metrics,
    read_rows,
    score_dataset,
    split_cases,
)
from already_mentioned.services.embeddings import (
    FRIDA_CONTRACT,
    EmbeddingService,
    FridaEmbeddingService,
)
from already_mentioned.services.questions import is_question_candidate


def select_threshold(outcomes: tuple[Outcome, ...], beta: float = 1.1) -> float:
    """Select on calibration cases only; beta > 1 gives recall slightly more weight."""
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Beta должна быть положительным конечным числом")
    if not any(o.case.expected for o in outcomes):
        raise ValueError("В ранней части нужны положительные случаи")
    best_key, best = None, 1.0
    for step in range(1001):
        threshold = step / 1000
        result = metrics(outcomes, threshold)
        tp, fp, fn = result["correct"], result["wrong"], result["missed"]
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp
        score = (1 + beta**2) * tp / denominator if denominator else 0
        key = (score, -fp, tp, threshold)
        if best_key is None or key > best_key:
            best_key, best = key, threshold
    return best


def file_hash(path: Path) -> dict:
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


async def run(args: argparse.Namespace, embeddings: EmbeddingService) -> dict:
    datasets = []
    for folder in args.datasets:
        paths = [
            folder / "reviewed_cases_full.csv",
            *sorted(folder.glob("chat*_inferred_solution_bank.csv")),
        ]
        cases, bank, excluded = load_dataset(paths[0], paths[1:])
        datasets.append((paths, cases, bank, excluded))
    # Freeze the split before the newly unresolved labels were excluded.
    source = read_rows(args.split_cases, {"chat", "message_id"})
    split_source = tuple(
        HistoricalCase(r["chat"], int(r["message_id"]), "", frozenset()) for r in source
    )
    source_ids = {(c.chat, c.message_id) for c in split_source}
    if len(source_ids) != len(split_source):
        raise ValueError("Повторные ID в источнике разбиения")
    validation_ids, test_ids = split_cases(split_source, 0.6)
    for paths, _, _, _ in datasets:
        rows = read_rows(paths[0], {"chat", "message_id"})
        if {(r["chat"], int(r["message_id"])) for r in rows} != source_ids:
            raise ValueError("Источник разбиения должен содержать те же ID сообщений")
    inputs = [[file_hash(p) for p in paths] for paths, _, _, _ in datasets]
    split_hash = file_hash(args.split_cases)
    selected_texts = sorted(
        {
            c.message
            for _, cases, _, _ in datasets
            for c in cases
            if is_question_candidate(c.message) or c.expected
        }
    )
    saved_texts = sorted(
        {t for _, _, bank, _ in datasets for s in bank for t in (s.question, s.answer)}
    )
    started = perf_counter()
    await embeddings.embed_query("Проверка загрузки")
    load_seconds = perf_counter() - started
    queries, passages, durations = {}, {}, []
    for number, text in enumerate(selected_texts, 1):
        started = perf_counter()
        queries[text] = await embeddings.embed_query(text)
        durations.append((perf_counter() - started) * 1000)
        if number % 100 == 0:
            print(f"Запросы: {number}/{len(selected_texts)}", flush=True)
    started = perf_counter()
    for number, text in enumerate(saved_texts, 1):
        passages[text] = await embeddings.embed_passage(text)
        if number % 100 == 0:
            print(f"Сохранённые тексты: {number}/{len(saved_texts)}", flush=True)
    passage_seconds = perf_counter() - started
    scenarios, threshold = [], None
    for index, (_, cases, bank, excluded) in enumerate(datasets):
        selected = tuple(
            c for c in cases if is_question_candidate(c.message) or c.expected
        )
        scored = score_dataset(selected, bank, queries, passages)["both"]
        by_id = {(o.case.chat, o.case.message_id): o for o in scored}
        outcomes = tuple(
            by_id.get((c.chat, c.message_id), Outcome(c, False, None, -2, None))
            for c in cases
        )
        validation = tuple(
            o for o in outcomes if (o.case.chat, o.case.message_id) in validation_ids
        )
        test = tuple(
            o for o in outcomes if (o.case.chat, o.case.message_id) in test_ids
        )
        if index == 0:
            threshold = select_threshold(validation, args.beta)
        scenarios.append(
            {
                "scenario": "main" if index == 0 else "with_inferred",
                "inputs": inputs[index],
                "cases": len(cases),
                "solutions": len(bank),
                "excluded": excluded,
                "validation_cases": len(validation),
                "test_cases": len(test),
                "before": {
                    "validation": metrics(validation, 0.88),
                    "test": metrics(test, 0.88),
                },
                "after": {
                    "validation": metrics(validation, threshold),
                    "test": metrics(test, threshold),
                },
                "test_by_chat": {
                    chat: metrics(
                        tuple(o for o in test if o.case.chat == chat), threshold
                    )
                    for chat in sorted({c.chat for c in cases})
                },
                "sensitivity_validation": [
                    {"threshold": t, **metrics(validation, t)}
                    for t in sorted(
                        {
                            max(0, round(threshold - 0.02, 3)),
                            threshold,
                            min(1, round(threshold + 0.02, 3)),
                            0.88,
                        }
                    )
                ],
                "details": [
                    {
                        "chat": o.case.chat,
                        "message_id": o.case.message_id,
                        "split": "validation"
                        if (o.case.chat, o.case.message_id) in validation_ids
                        else "test",
                        "expected": sorted(o.case.expected),
                        "best": o.best_id,
                        "score": o.score,
                        "filter_passed": o.passed,
                    }
                    for o in outcomes
                ],
            }
        )
    # Detect modified inputs before publishing results.
    if inputs != [
        [file_hash(p) for p in paths] for paths, _, _, _ in datasets
    ] or split_hash != file_hash(args.split_cases):
        raise ValueError("Входы изменились во время проверки")
    return {
        "format_version": 1,
        "model_contract": FRIDA_CONTRACT,
        "annotation_status": "provisional_not_owner_confirmed",
        "selection": {
            "dataset": "main_validation_only",
            "objective": "F_beta",
            "beta": args.beta,
            "grid_step": 0.001,
            "margin": 0,
            "threshold": threshold,
            "baseline_threshold": 0.88,
        },
        "split": {
            "source": split_hash,
            "fraction": 0.6,
            "method": "frozen_earlier_message_ids_before_unknown_exclusion",
            "validation_ids_sha256": hashlib.sha256(
                json.dumps(sorted(validation_ids)).encode()
            ).hexdigest(),
            "test_ids_sha256": hashlib.sha256(
                json.dumps(sorted(test_ids)).encode()
            ).hexdigest(),
            "later_cases_previously_examined": True,
        },
        "filter_sha256": hashlib.sha256(
            Path(__file__).with_name("services").joinpath("questions.py").read_bytes()
        ).hexdigest(),
        "timing": {
            "load_seconds": load_seconds,
            "unique_queries": len(queries),
            "unique_passages": len(passages),
            "query_mean_ms": float(np.mean(durations)),
            "query_p95_ms": float(np.percentile(durations, 95)),
            "passage_mean_ms": passage_seconds / len(passages) * 1000,
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Калибровка порога FRIDA без Telegram и SQLite"
    )
    parser.add_argument(
        "--datasets",
        required=True,
        nargs=2,
        type=Path,
        help="Основной и дополнительный CSV-сценарии, в этом порядке",
    )
    parser.add_argument("--split-cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--beta", type=float, default=1.1)
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix != ".json":
        parser.error("Укажите новый отдельный JSON-файл")
    if not np.isfinite(args.beta) or args.beta <= 0:
        parser.error("Beta должна быть положительным конечным числом")
    report = asyncio.run(run(args, FridaEmbeddingService()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Выбран порог {report['selection']['threshold']:.3f}; отчёт: {args.output}")


if __name__ == "__main__":
    main()
