"""Evaluate top-k pair reranking without altering production embeddings or search."""

import argparse
import asyncio
import csv
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from time import perf_counter

import numpy as np

from already_mentioned.historical_evaluation import (
    HistoricalCase,
    HistoricalSolution,
    Outcome,
    calibrate,
    chosen_id,
    encode_dataset,
    load_dataset,
    metrics,
    split_cases,
)
from already_mentioned.services.embeddings import (
    MODEL_DIMENSION,
    MODEL_NAME,
    EmbeddingService,
    FastEmbedEmbeddingService,
)
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reranking import (
    RERANKER_FILE,
    RERANKER_MODEL,
    RERANKER_REVISION,
    OnnxReranker,
    PairScorer,
)
from already_mentioned.services.similarity import cosine_similarity

INPUTS = ("answer", "question_answer")


def shortlist(
    case: HistoricalCase,
    solutions: tuple[HistoricalSolution, ...],
    queries: dict[str, np.ndarray],
    passages: dict[str, np.ndarray],
) -> tuple[tuple[HistoricalSolution, float], ...]:
    candidates = []
    for solution in solutions:
        if solution.chat != case.chat or solution.answer_message_id >= case.message_id:
            continue
        score = max(
            cosine_similarity(queries[case.message], passages[solution.question]),
            cosine_similarity(queries[case.message], passages[solution.answer]),
        )
        candidates.append((solution, score))
    return tuple(sorted(candidates, key=lambda item: -item[1]))


def rerank(
    case: HistoricalCase,
    candidates: tuple[tuple[HistoricalSolution, float], ...],
    scorer: PairScorer,
    mode: str,
) -> Outcome:
    passed = is_question_candidate(case.message)
    if not passed or not candidates:
        return Outcome(case, passed, None, -2.0, None)
    texts = [
        s.answer if mode == "answer" else f"Вопрос: {s.question}\nОтвет: {s.answer}"
        for s, _ in candidates
    ]
    scores = scorer.predict(case.message, texts)
    if len(scores) != len(candidates) or not all(
        np.isfinite(s) and 0 <= s <= 1 for s in scores
    ):
        raise ValueError("Нужна одна конечная оценка от 0 до 1 на кандидата")
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: -item[1])
    (solution, _), score = ranked[0]
    gap = score - ranked[1][1] if len(ranked) > 1 else None
    return Outcome(
        case, passed, solution.id, float(score), float(gap) if gap is not None else None
    )


def memory_snapshot() -> dict | None:
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    return {
        "rss_mb": info.rss / 1024**2,
        "peak_rss_mb": getattr(info, "peak_wset", info.rss) / 1024**2,
    }


def report_experiment(
    values: tuple[Outcome, ...],
    validation_ids: frozenset[tuple[str, int]],
    threshold: float | None = None,
) -> dict:
    validation = tuple(
        o for o in values if (o.case.chat, o.case.message_id) in validation_ids
    )
    test = tuple(
        o for o in values if (o.case.chat, o.case.message_id) not in validation_ids
    )
    if threshold is None:
        threshold, _ = calibrate(validation, (0.0,))
    return {
        "threshold": threshold,
        "validation": metrics(validation, threshold),
        "test": metrics(test, threshold),
        "all": metrics(values, threshold),
        "test_by_chat": {
            chat: metrics(tuple(o for o in test if o.case.chat == chat), threshold)
            for chat in sorted({o.case.chat for o in values})
        },
    }


def write_review(
    path: Path,
    baseline: tuple[Outcome, ...],
    reranked: tuple[Outcome, ...] | None,
    solutions: tuple[HistoricalSolution, ...],
    threshold: float,
    limit: int,
) -> int:
    bank = {(s.chat, s.id): s for s in solutions}
    pairs = [
        (old, new)
        for old, new in zip(baseline, reranked or baseline, strict=True)
        if old.case.expected
        or (
            chosen_id(old, 0.88, 0) is not None
            if reranked is None
            else chosen_id(old, 0.88, 0) != chosen_id(new, threshold, 0)
        )
    ]
    pairs.sort(
        key=lambda pair: (
            not bool(pair[0].case.expected),
            pair[0].case.chat,
            pair[0].case.message_id,
        )
    )
    positive = [pair for pair in pairs if pair[0].case.expected]
    negative = [pair for pair in pairs if not pair[0].case.expected]
    groups = [
        sorted(
            [pair for pair in negative if pair[0].case.chat == chat],
            key=lambda pair: (-pair[0].score, pair[0].case.message_id),
        )
        for chat in sorted({pair[0].case.chat for pair in negative})
    ]
    pairs = positive + [
        pair for group in zip_longest(*groups) for pair in group if pair is not None
    ]
    rows = []
    for old, new in pairs[:limit]:
        row = {
            "chat": old.case.chat,
            "message_id": old.case.message_id,
            "new_message": old.case.message,
            "expected_solution_id": "|".join(sorted(old.case.expected)) or "SILENCE",
            "owner_decision": "",
            "owner_comment": "",
            "review_reason": "positive_label"
            if old.case.expected
            else "bot_prompt_on_silence"
            if reranked is None
            else "methods_disagree",
        }
        comparisons = [("e5", old, 0.88)]
        if reranked is not None:
            comparisons.append(("reranker", new, threshold))
        for label, outcome, cutoff in comparisons:
            solution = bank.get((old.case.chat, outcome.best_id))
            row.update(
                {
                    f"{label}_decision": chosen_id(outcome, cutoff, 0) or "SILENCE",
                    f"{label}_candidate": outcome.best_id or "",
                    f"{label}_question": solution.question if solution else "",
                    f"{label}_answer": solution.answer if solution else "",
                    f"{label}_score": outcome.score,
                }
            )
        rows.append(row)
    fields = [
        "chat",
        "message_id",
        "new_message",
        "expected_solution_id",
        "owner_decision",
        "owner_comment",
        "review_reason",
    ]
    fields.extend(
        f"{label}_{field}"
        for label in ("e5", "reranker")
        for field in ("decision", "candidate", "question", "answer", "score")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_baseline(args: argparse.Namespace, source: Path) -> tuple:
    """Reuse only a matching historical report; never fabricate new retrieval."""
    cases, solutions, excluded = load_dataset(args.cases, args.banks)
    saved = json.loads(source.read_text(encoding="utf-8"))
    inputs = [
        {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (args.cases, *args.banks)
    ]
    contract = {
        "model": MODEL_NAME,
        "dimension": MODEL_DIMENSION,
        "query_prefix": "query:",
        "saved_prefix": "passage:",
        "normalization": "L2",
        "dtype": "float32",
    }
    if (
        saved.get("format_version") != 1
        or saved.get("inputs") != inputs
        or saved.get("model_contract") != contract
    ):
        raise ValueError("Прежний отчёт не соответствует CSV или контракту E5")
    rows = [d for d in saved.get("details", []) if d.get("strategy") == "both"]
    by_id = {(d["chat"], d["message_id"]): d for d in rows}
    if len(rows) != len(by_id) or set(by_id) != {(c.chat, c.message_id) for c in cases}:
        raise ValueError("Прежний отчёт не содержит ровно один результат на случай")
    baseline = []
    bank = {(s.chat, s.id): s for s in solutions}
    for case in cases:
        row = by_id[(case.chat, case.message_id)]
        best, score, gap = row["best"], row["score"], row["gap"]
        solution = bank.get((case.chat, best))
        if (
            row["expected"] != sorted(case.expected)
            or row["filter_passed"] != is_question_candidate(case.message)
            or not np.isfinite(score)
            or (gap is not None and not np.isfinite(gap))
            or (
                best is not None
                and (solution is None or solution.answer_message_id >= case.message_id)
            )
        ):
            raise ValueError("Прежние результаты противоречат проверенному набору")
        baseline.append(Outcome(case, row["filter_passed"], best, score, gap))
    return cases, solutions, excluded, inputs, tuple(baseline)


def prepare_review(args: argparse.Namespace) -> dict:
    _, solutions, excluded, inputs, baseline = load_baseline(
        args, args.prepare_review_from
    )
    count = write_review(
        args.review_output, baseline, None, solutions, 0.88, args.review_limit
    )
    report = {
        "format_version": 1,
        "status": "review_only_no_new_inference",
        "annotation_status": "provisional_not_owner_confirmed",
        "inputs": inputs,
        "source_report_sha256": hashlib.sha256(
            args.prepare_review_from.read_bytes()
        ).hexdigest(),
        "excluded": excluded,
        "baseline_reused": metrics(baseline, 0.88),
        "review_rows": count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


async def run(
    args: argparse.Namespace, embeddings: EmbeddingService | None, scorer: PairScorer
) -> dict:
    if args.candidates_from:
        cases, solutions, excluded, _, baseline = load_baseline(
            args, args.candidates_from
        )
        bank = {(s.chat, s.id): s for s in solutions}
        candidates = [
            ((bank[(o.case.chat, o.best_id)], o.score),) if o.best_id else ()
            for o in baseline
        ]
        encoding_time = None
        print("Используются прежние кандидаты top-1; E5 не загружается.", flush=True)
    else:
        cases, solutions, excluded = load_dataset(args.cases, args.banks)
        print("Кодирование текущей E5 small...", flush=True)
        queries, passages, encoding_time = await encode_dataset(
            cases, solutions, embeddings
        )
        candidates = [shortlist(case, solutions, queries, passages) for case in cases]
        baseline = tuple(
            Outcome(
                case,
                is_question_candidate(case.message),
                found[0][0].id if found else None,
                found[0][1] if found else -2.0,
                found[0][1] - found[1][1] if len(found) > 1 else None,
            )
            for case, found in zip(cases, candidates, strict=True)
        )
    validation_ids, test_ids = split_cases(cases, args.validation_fraction)
    memory = {"with_embeddings": None if args.candidates_from else memory_snapshot()}
    print("Загрузка и прогрев локального ONNX-оценщика...", flush=True)
    start = perf_counter()
    scorer.predict("Как войти в кабинет?", ["Откройте форму входа."])
    load_seconds = perf_counter() - start
    memory["with_reranker_only" if args.candidates_from else "with_both_models"] = (
        memory_snapshot()
    )
    experiments, details = [], []
    reference = None
    reference_threshold = None
    for mode in INPUTS:
        for k in args.top_k:
            values, durations = [], []
            for index, (case, found) in enumerate(zip(cases, candidates, strict=True)):
                start = perf_counter()
                outcome = rerank(case, found[:k], scorer, mode)
                elapsed = (perf_counter() - start) * 1000
                if outcome.passed and found:
                    durations.append(elapsed)
                values.append(outcome)
                if (index + 1) % 200 == 0:
                    print(f"{mode}, top-{k}: {index + 1}/{len(cases)}", flush=True)
            values = tuple(values)
            result = report_experiment(values, validation_ids)
            coverage = {}
            for name, ids in (("all", None), ("test", test_ids)):
                positive = [
                    (c, f)
                    for c, f in zip(cases, candidates, strict=True)
                    if c.expected and (ids is None or (c.chat, c.message_id) in ids)
                ]
                coverage[name] = {
                    "positives": len(positive),
                    "retrieved": sum(
                        any(s.id in c.expected for s, _ in f[:k]) for c, f in positive
                    ),
                }
            experiments.append(
                {
                    "input": mode,
                    "top_k": k,
                    **result,
                    "candidate_coverage": coverage,
                    "warm_query_mean_ms": float(np.mean(durations))
                    if durations
                    else 0.0,
                    "warm_query_p95_ms": float(np.percentile(durations, 95))
                    if durations
                    else 0.0,
                }
            )
            details.extend(
                {
                    "chat": o.case.chat,
                    "message_id": o.case.message_id,
                    "input": mode,
                    "top_k": k,
                    "best": o.best_id,
                    "score": o.score,
                    "expected": sorted(o.case.expected),
                    "decision": chosen_id(o, result["threshold"], 0),
                }
                for o in values
            )
            # Preregistered review reference, never choose a variant on test metrics.
            if mode == "answer" and k == max(args.top_k):
                reference, reference_threshold = values, result["threshold"]
            snapshot = memory_snapshot()
            if snapshot:
                memory["last"] = snapshot
    review_rows = write_review(
        args.review_output,
        baseline,
        reference,
        solutions,
        reference_threshold,
        args.review_limit,
    )
    report = {
        "format_version": 1,
        "annotation_status": "provisional_not_owner_confirmed",
        "embedding_model": MODEL_NAME,
        "embedding_threads": None if args.candidates_from else args.threads,
        "retrieval_source": "verified_historical_top1"
        if args.candidates_from
        else "live_e5",
        "models_coexist": not bool(args.candidates_from),
        "source_report_sha256": hashlib.sha256(
            args.candidates_from.read_bytes()
        ).hexdigest()
        if args.candidates_from
        else None,
        "reranker": {
            "model": RERANKER_MODEL,
            "revision": RERANKER_REVISION,
            "file": RERANKER_FILE,
            "score_transform": "sigmoid_uncalibrated",
            "threads": args.threads,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "load_and_download_seconds": load_seconds,
            "total_pairs": getattr(scorer, "total_pairs", None),
            "truncated_pairs": getattr(scorer, "truncated_pairs", None),
        },
        "inputs": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (args.cases, *args.banks)
        ],
        "dataset": {
            "cases": len(cases),
            "positives": sum(bool(c.expected) for c in cases),
            "excluded": excluded,
            "solutions": len(solutions),
        },
        "split": {"validation": len(validation_ids), "test": len(test_ids)},
        "encoding_timing": encoding_time,
        "process_memory": memory,
        "baseline_fixed": report_experiment(baseline, validation_ids, 0.88),
        "baseline_calibrated": report_experiment(baseline, validation_ids),
        "experiments": experiments,
        "details": details,
        "review": {"input": "answer", "top_k": max(args.top_k), "rows": review_rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверить локальное повторное ранжирование без Telegram и SQLite"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--banks", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("training_runs/reranking/report.json")
    )
    parser.add_argument(
        "--review-output", type=Path, default=Path("training_runs/reranking/review.csv")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--review-limit", type=int, default=60)
    parser.add_argument("--validation-fraction", type=float, default=0.6)
    parser.add_argument(
        "--prepare-review-from",
        type=Path,
        help="Подготовить проверку из совпадающего отчёта истории без загрузки моделей",
    )
    parser.add_argument(
        "--candidates-from",
        type=Path,
        help="Переоценить top-1 из совпадающего отчёта истории без загрузки E5",
    )
    args = parser.parse_args()
    if (
        min(*args.top_k, args.threads, args.batch_size, args.review_limit) < 1
        or not 8 <= args.max_length <= 512
    ):
        parser.error("Числа должны быть положительными; длина пары от 8 до 512")
    if not 0 < args.validation_fraction < 1:
        parser.error("Доля ранних случаев должна быть между 0 и 1")
    paths = [
        p.resolve() for p in (args.cases, *args.banks, args.output, args.review_output)
    ]
    if args.prepare_review_from:
        paths.append(args.prepare_review_from.resolve())
    if args.candidates_from:
        paths.append(args.candidates_from.resolve())
    if (
        len(set(paths)) != len(paths)
        or args.output.suffix != ".json"
        or args.review_output.suffix != ".csv"
    ):
        parser.error("Входы и отчёты должны быть разными файлами; отчёты JSON и CSV")
    args.top_k = sorted(set(args.top_k))
    if args.candidates_from and (args.top_k != [1] or args.prepare_review_from):
        parser.error("Кэш прежнего отчёта поддерживает только --top-k 1 и новый прогон")
    if args.prepare_review_from:
        try:
            report = prepare_review(args)
        except (ValueError, OSError, KeyError, TypeError) as error:
            parser.exit(1, f"Не удалось подготовить проверку: {error}\n")
        print(
            f"Подготовлено {report['review_rows']} случаев без загрузки моделей. "
            f"Новых метрик ранжирования нет. Очередь: {args.review_output}"
        )
        return
    scorer = OnnxReranker(
        args.cache_dir,
        threads=args.threads,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_only=args.local_only,
    )
    try:
        report = asyncio.run(
            run(
                args,
                None
                if args.candidates_from
                else FastEmbedEmbeddingService(args.cache_dir, threads=args.threads),
                scorer,
            )
        )
    except (ValueError, OSError, MemoryError, KeyError, TypeError) as error:
        parser.exit(1, f"Не удалось выполнить проверку: {error}\n")
    except Exception as error:
        if type(error).__module__.startswith("onnxruntime."):
            parser.exit(
                1,
                "Ошибка ONNX Runtime. Если указана нехватка памяти, освободите "
                "её и повторите прогон. Для очереди без моделей используйте "
                f"--prepare-review-from. Детали: {error}\n",
            )
        raise
    for row in report["experiments"]:
        m = row["test"]
        print(
            f"{row['input']}, top-{row['top_k']}: поздняя часть "
            f"верно {m['correct']}, ошибочно {m['wrong']}, пропуски {m['missed']}; "
            f"CPU {row['warm_query_mean_ms']:.1f} мс/запрос"
        )
    print(f"Отчёт: {args.output}; очередь проверки: {args.review_output}")


if __name__ == "__main__":
    main()
