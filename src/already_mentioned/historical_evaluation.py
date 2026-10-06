"""Read-only search ablation on explicitly labelled historical CSV exports."""

import argparse
import asyncio
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from already_mentioned.services.embeddings import (
    FRIDA_CONTRACT,
    MODEL_DIMENSION,
    MODEL_NAME,
    EmbeddingService,
    FridaEmbeddingService,
    normalize_vector,
)
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.similarity import cosine_similarity

STRATEGIES = ("question", "answer", "both")
LEGACY_REPORT_CONTRACT = {
    "model": MODEL_NAME,
    "dimension": MODEL_DIMENSION,
    "query_prefix": "query:",
    "saved_prefix": "passage:",
    "normalization": "L2",
    "dtype": "float32",
}


@dataclass(frozen=True)
class HistoricalSolution:
    chat: str
    id: str
    answer_message_id: int
    question: str
    answer: str


@dataclass(frozen=True)
class HistoricalCase:
    chat: str
    message_id: int
    message: str
    expected: frozenset[str]


@dataclass(frozen=True)
class Outcome:
    case: HistoricalCase
    passed: bool
    best_id: str | None
    score: float
    # The runner-up must be a different solution, not its second text vector.
    gap: float | None


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"Отсутствуют обязательные столбцы CSV: {path.name}")
        return list(reader)


def load_dataset(
    cases_path: Path, bank_paths: list[Path]
) -> tuple[tuple[HistoricalCase, ...], tuple[HistoricalSolution, ...], int]:
    solutions = []
    keys = set()
    for path in bank_paths:
        for row in read_rows(
            path,
            {
                "chat",
                "solution_id",
                "answer_message_id",
                "saved_question",
                "saved_answer",
                "historical_proxy_status",
            },
        ):
            if row["historical_proxy_status"] != "usable_candidate":
                continue
            key = (row["chat"], row["solution_id"])
            if key in keys or not all(key):
                raise ValueError("Пустой или повторный ID решения в чате")
            keys.add(key)
            solutions.append(
                HistoricalSolution(
                    *key,
                    int(row["answer_message_id"]),
                    row["saved_question"],
                    row["saved_answer"],
                )
            )
    bank = {(s.chat, s.id): s for s in solutions}
    cases = []
    seen = set()
    excluded = 0
    for row in read_rows(
        cases_path,
        {
            "chat",
            "message_id",
            "new_message",
            "expected_solution_id",
        },
    ):
        key = (row["chat"], int(row["message_id"]))
        if key in seen or not key[0]:
            raise ValueError("Пустой чат или повторный ID нового сообщения")
        seen.add(key)
        label = row["expected_solution_id"].strip()
        if label == "UNRESOLVED":
            excluded += 1
            continue
        if not label:
            raise ValueError(
                "Пустая метка не означает молчание: используйте UNRESOLVED"
            )
        expected = frozenset() if label == "SILENCE" else frozenset(label.split("|"))
        for solution_id in expected:
            solution = bank.get((key[0], solution_id))
            if solution is None or solution.answer_message_id >= key[1]:
                raise ValueError(
                    "Эталон должен быть более ранним решением того же чата"
                )
        cases.append(HistoricalCase(*key, row["new_message"], expected))
    if not cases or not solutions:
        raise ValueError("Нужны размеченные случаи и хотя бы одно решение")
    return (
        tuple(sorted(cases, key=lambda c: (c.chat, c.message_id))),
        tuple(solutions),
        excluded,
    )


def split_cases(
    cases: tuple[HistoricalCase, ...], fraction: float
) -> tuple[frozenset[tuple[str, int]], frozenset[tuple[str, int]]]:
    """Select earlier cases per chat without consulting labels or scores."""
    totals = Counter(case.chat for case in cases)
    seen: Counter[str] = Counter()
    validation, test = set(), set()
    for case in sorted(cases, key=lambda c: (c.chat, c.message_id)):
        seen[case.chat] += 1
        target = (
            validation if seen[case.chat] <= int(totals[case.chat] * fraction) else test
        )
        target.add((case.chat, case.message_id))
    if not validation or not test:
        raise ValueError("Недостаточно случаев для двух частей истории")
    return frozenset(validation), frozenset(test)


async def encode_dataset(
    cases: tuple[HistoricalCase, ...],
    solutions: tuple[HistoricalSolution, ...],
    embeddings: EmbeddingService,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    messages = sorted({case.message for case in cases})
    passages = sorted({text for s in solutions for text in (s.question, s.answer)})
    # Exclude lazy loading from timings; call exactly the bot's service contract.
    await embeddings.embed_query(messages[0])
    query_vectors, passage_vectors = {}, {}
    durations = []
    for text in messages:
        start = perf_counter()
        query_vectors[text] = normalize_vector(await embeddings.embed_query(text))
        durations.append((perf_counter() - start) * 1000)
    start = perf_counter()
    for text in passages:
        passage_vectors[text] = normalize_vector(await embeddings.embed_passage(text))
    timing = {
        "unique_queries": len(messages),
        "unique_passages": len(passages),
        "single_query_mean_ms": float(np.mean(durations)),
        "single_query_p95_ms": float(np.percentile(durations, 95)),
        "passage_mean_ms": (perf_counter() - start) * 1000 / len(passages),
        "excludes_load_sqlite_telegram": True,
    }
    return query_vectors, passage_vectors, timing


def score_dataset(
    cases: tuple[HistoricalCase, ...],
    solutions: tuple[HistoricalSolution, ...],
    queries: dict[str, np.ndarray],
    passages: dict[str, np.ndarray],
) -> dict[str, tuple[Outcome, ...]]:
    result: dict[str, list[Outcome]] = {mode: [] for mode in STRATEGIES}
    for case in cases:
        scores: dict[str, list[tuple[str, float]]] = {mode: [] for mode in STRATEGIES}
        for solution in solutions:
            if (
                solution.chat != case.chat
                or solution.answer_message_id >= case.message_id
            ):
                continue
            question = cosine_similarity(
                queries[case.message], passages[solution.question]
            )
            answer = cosine_similarity(queries[case.message], passages[solution.answer])
            for mode, score in (
                ("question", question),
                ("answer", answer),
                ("both", max(question, answer)),
            ):
                scores[mode].append((solution.id, score))
        for mode in STRATEGIES:
            ranking = sorted(scores[mode], key=lambda item: -item[1])
            best_id, score = ranking[0] if ranking else (None, -2.0)
            gap = score - ranking[1][1] if len(ranking) >= 2 else None
            result[mode].append(
                Outcome(case, is_question_candidate(case.message), best_id, score, gap)
            )
    return {mode: tuple(values) for mode, values in result.items()}


def chosen_id(outcome: Outcome, threshold: float, margin: float) -> str | None:
    if not outcome.passed or outcome.score < threshold:
        return None
    if margin > 0 and (outcome.gap is None or outcome.gap < margin):
        return None
    return outcome.best_id


def metrics(outcomes: tuple[Outcome, ...], threshold: float, margin: float = 0) -> dict:
    counts = dict.fromkeys(
        (
            "correct",
            "wrong",
            "missed",
            "correct_silence",
            "false_prompt",
            "wrong_solution",
            "filter_missed",
            "wrong_leader",
            "below_threshold",
            "margin_missed",
            "correct_leader_without_filter",
            "positives",
        ),
        0,
    )
    for outcome in outcomes:
        expected = outcome.case.expected
        chosen = chosen_id(outcome, threshold, margin)
        if expected:
            counts["positives"] += 1
            counts["correct_leader_without_filter"] += outcome.best_id in expected
            if chosen in expected:
                counts["correct"] += 1
            else:
                counts["missed"] += 1
                if not outcome.passed:
                    counts["filter_missed"] += 1
                elif outcome.best_id not in expected:
                    counts["wrong_leader"] += 1
                elif outcome.score < threshold:
                    counts["below_threshold"] += 1
                else:
                    counts["margin_missed"] += 1
                if chosen is not None:
                    counts["wrong_solution"] += 1
                    counts["wrong"] += 1
        elif chosen is None:
            counts["correct_silence"] += 1
        else:
            counts["false_prompt"] += 1
            counts["wrong"] += 1
    tp, fp, fn = counts["correct"], counts["wrong"], counts["missed"]
    return {
        **counts,
        "total": len(outcomes),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def calibrate(
    outcomes: tuple[Outcome, ...], margins: tuple[float, ...]
) -> tuple[float, float]:
    """Maximize validation F1, counting wrong positive leaders as FP and FN."""
    thresholds = sorted(
        {o.score for o in outcomes if o.passed and o.best_id is not None}
    )
    # Include an explicit abstention option even if the maximum is exactly one.
    thresholds.append(1.000001)
    best_key = None
    best = (1.000001, 0.0)
    for margin in margins:
        for threshold in thresholds:
            value = metrics(outcomes, threshold, margin)
            key = (value["f1"], -value["wrong"], value["correct"], threshold, -margin)
            if best_key is None or key > best_key:
                best_key, best = key, (threshold, margin)
    return best


async def run(
    args: argparse.Namespace,
    embeddings: EmbeddingService,
    *,
    contract: dict = LEGACY_REPORT_CONTRACT,
) -> dict:
    cases, solutions, excluded = load_dataset(args.cases, args.banks)
    validation_ids, test_ids = split_cases(cases, args.validation_fraction)
    queries, passages, timing = await encode_dataset(cases, solutions, embeddings)
    outcomes = score_dataset(cases, solutions, queries, passages)
    reports = []
    details = []
    for mode, values in outcomes.items():
        validation = tuple(
            o for o in values if (o.case.chat, o.case.message_id) in validation_ids
        )
        test = tuple(o for o in values if (o.case.chat, o.case.message_id) in test_ids)
        for experiment, threshold, margin in (
            ("fixed", args.threshold, 0.0),
            ("threshold", *calibrate(validation, (0.0,))),
            ("threshold_and_margin", *calibrate(validation, tuple(args.margins))),
        ):
            reports.append(
                {
                    "strategy": mode,
                    "experiment": experiment,
                    "threshold": threshold,
                    "margin": margin,
                    "validation": metrics(validation, threshold, margin),
                    "test": metrics(test, threshold, margin),
                    "all": metrics(values, threshold, margin),
                    "test_by_chat": {
                        chat: metrics(
                            tuple(o for o in test if o.case.chat == chat),
                            threshold,
                            margin,
                        )
                        for chat in sorted({case.chat for case in cases})
                    },
                }
            )
        details.extend(
            {
                "chat": o.case.chat,
                "message_id": o.case.message_id,
                "split": "validation"
                if (o.case.chat, o.case.message_id) in validation_ids
                else "test",
                "strategy": mode,
                "filter_passed": o.passed,
                "expected": sorted(o.case.expected),
                "best": o.best_id,
                "score": o.score,
                "gap": o.gap,
            }
            for o in values
        )
    report = {
        "format_version": 1,
        "annotation_status": "provisional_not_owner_confirmed",
        "model_contract": contract,
        "inputs": [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (args.cases, *args.banks)
        ],
        "dataset": {
            "cases": len(cases),
            "positives": sum(bool(c.expected) for c in cases),
            "excluded": excluded,
            "solutions": len(solutions),
        },
        "split": {
            "validation": len(validation_ids),
            "test": len(test_ids),
            "validation_fraction": args.validation_fraction,
            "method": "earlier_message_ids_per_chat",
        },
        "timing": timing,
        "experiments": reports,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить поиск на размеченной истории без Telegram и SQLite"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--banks", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_runs/history_ablation/report.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--validation-fraction", type=float, default=0.6)
    parser.add_argument(
        "--margins", type=float, nargs="+", default=[0, 0.005, 0.01, 0.02, 0.04]
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 < args.validation_fraction < 1:
        parser.error("Порог должен быть от 0 до 1, доля ранних случаев — между 0 и 1")
    if any(not np.isfinite(m) or not 0 <= m <= 2 for m in args.margins):
        parser.error("Разница оценок должна быть от 0 до 2")
    if args.output.suffix != ".json" or args.output.resolve() in {
        p.resolve() for p in (args.cases, *args.banks)
    }:
        parser.error("Отчёт должен быть отдельным JSON-файлом")
    try:
        report = asyncio.run(
            run(
                args,
                FridaEmbeddingService(args.cache_dir),
                contract=FRIDA_CONTRACT,
            )
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Не удалось выполнить проверку: {error}\n")
    print(
        f"Предварительный набор: {report['dataset']['cases']} случаев; "
        f"отчёт: {args.output}"
    )
    for row in report["experiments"]:
        value = row["test"]
        print(
            f"{row['strategy']} / {row['experiment']}: "
            f"порог {row['threshold']:.4f}, разница {row['margin']:.4f}; "
            f"на поздней части верных {value['correct']}, ошибочных {value['wrong']}, "
            f"пропусков {value['missed']}"
        )


if __name__ == "__main__":
    main()
