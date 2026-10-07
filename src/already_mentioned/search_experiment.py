"""Compare FRIDA retrieval, contextual documents and top-five reranking offline."""

import argparse
import asyncio
import copy
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
    split_cases,
)
from already_mentioned.services.embeddings import FRIDA_CONTRACT, FridaEmbeddingService
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reranking import (
    RERANKER_FILE,
    RERANKER_MODEL,
    RERANKER_REVISION,
    OnnxReranker,
)
from already_mentioned.services.similarity import cosine_similarity
from already_mentioned.threshold_calibration import file_hash, select_threshold

STRATEGIES = ("baseline", "context_pair", "rerank_top5")


def pair_text(solution):
    return f"Вопрос: {solution.question}\nОтвет: {solution.answer}"


def retrieve(case, bank, queries, passages, pairs):
    """Only earlier same-chat solutions; stable ordering for tied scores."""
    rows = []
    for solution in bank:
        if solution.chat != case.chat or solution.answer_message_id >= case.message_id:
            continue
        query = queries[case.message]
        question = cosine_similarity(query, passages[solution.question])
        answer = cosine_similarity(query, passages[solution.answer])
        rows.append({
            "id": solution.id, "question_score": question, "answer_score": answer,
            "score": max(question, answer),
            "context_score": cosine_similarity(query, pairs[pair_text(solution)]),
        })
    return sorted(rows, key=lambda row: -row["score"])


def to_outcome(case, ranking, strategy):
    field = {"baseline": "score", "context_pair": "context_score",
             "rerank_top5": "rerank_score"}[strategy]
    ordered = sorted(ranking, key=lambda row: -row[field])
    score = ordered[0][field] if ordered else -2.0
    gap = score - ordered[1][field] if len(ordered) > 1 else None
    return Outcome(case, is_question_candidate(case.message),
                   ordered[0]["id"] if ordered else None, score, gap)


def freeze_split(datasets, source_path):
    source = read_rows(source_path, {"chat", "message_id"})
    cases = tuple(HistoricalCase(r["chat"], int(r["message_id"]), "", frozenset())
                  for r in source)
    ids = {(c.chat, c.message_id) for c in cases}
    if len(ids) != len(cases):
        raise ValueError("Повторные ID в источнике разбиения")
    for _, paths, _, _, _ in datasets:
        rows = read_rows(paths[0], {"chat", "message_id"})
        if {(r["chat"], int(r["message_id"])) for r in rows} != ids:
            raise ValueError("Источник разбиения должен содержать те же ID")
    return split_cases(cases, 0.6)


def recall_guard_report(baseline, reranked, beta=1.1):
    """Select only on main early cases, retaining the baseline's correct count."""
    result = copy.deepcopy(reranked)
    result["name"] = "frida-rerank_recall_guard"
    result["calibration"] += "_at_least_baseline_early_correct_count"

    def outcomes(rows):
        return tuple(Outcome(
            HistoricalCase(r["chat"], r["message_id"], "", frozenset(r["expected"])),
            r["filter_passed"], r["best"], r["score"], r["gap"],
        ) for r in rows)

    base_rows = baseline["scenarios"][0]["details"]
    candidate_rows = result["scenarios"][0]["details"]
    for report in (baseline, reranked):
        if report["contract"] != FRIDA_CONTRACT:
            raise ValueError("Нужен контракт полной FRIDA")
    if (baseline["split_source"] != reranked["split_source"]
            or baseline["filter_sha256"] != reranked["filter_sha256"]
            or len(baseline["scenarios"]) != len(reranked["scenarios"])):
        raise ValueError("Отчёты должны иметь одинаковые входы и разбиение")
    for base, candidate in zip(
        baseline["scenarios"], reranked["scenarios"], strict=True
    ):
        fields = ("chat", "message_id", "split", "filter_passed", "expected")
        if (base["inputs"] != candidate["inputs"] or base["name"] != candidate["name"]
                or [[r[f] for f in fields] for r in base["details"]]
                != [[r[f] for f in fields] for r in candidate["details"]]):
            raise ValueError("Отчёты должны описывать одни и те же случаи")
    base_early = outcomes([r for r in base_rows if r["split"] == "validation"])
    early = outcomes([r for r in candidate_rows if r["split"] == "validation"])
    minimum = metrics(base_early, baseline["scenarios"][0]["threshold"])["correct"]
    best, chosen = None, None
    for step in range(1001):
        threshold = step / 1000
        m = metrics(early, threshold)
        if m["correct"] < minimum:
            continue
        tp, fp, fn = m["correct"], m["wrong"], m["missed"]
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp
        score = (1 + beta**2) * tp / denominator if denominator else 0
        key = (score, -fp, tp, threshold)
        if best is None or key > best:
            best, chosen = key, threshold
    if chosen is None:
        raise ValueError("Оценщик не может сохранить число ранних правильных ответов")
    result["recall_guard"] = {
        "minimum_main_early_correct": minimum,
        "does_not_guarantee_same_correct_cases": True,
        "late_cases_not_used_for_threshold": True,
    }
    for scenario in result["scenarios"]:
        scenario["threshold"] = chosen
        rows = scenario["details"]
        for split, field in (("validation", "validation"), ("test", "test")):
            scenario[field] = metrics(
                outcomes([r for r in rows if r["split"] == split]), chosen
            )
        scenario["all"] = metrics(outcomes(rows), chosen)
    return result


async def run(args, embeddings, reranker):
    beta = getattr(args, "beta", 1.1)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Beta должна быть положительной")
    datasets = []
    input_hashes = {}
    for folder in args.datasets:
        paths = [folder / "reviewed_cases_full.csv",
                 *sorted(folder.glob("chat*_inferred_solution_bank.csv"))]
        cases, bank, excluded = load_dataset(paths[0], paths[1:])
        datasets.append((folder, paths, cases, bank, excluded))
        input_hashes.update({path: file_hash(path) for path in paths})
    split_hash = file_hash(args.split_cases)
    early_ids, _ = freeze_split(datasets, args.split_cases)
    messages = sorted({c.message for _, _, cases, _, _ in datasets for c in cases
                       if is_question_candidate(c.message) or c.expected})
    documents = sorted({text for _, _, _, bank, _ in datasets for s in bank
                        for text in (s.question, s.answer)})
    contexts = sorted({pair_text(s) for _, _, _, bank, _ in datasets for s in bank})
    queries, passages, pairs = {}, {}, {}
    query_times = []
    print("Загрузка полной FRIDA FP32", flush=True)
    await embeddings.embed_query("Проверка загрузки")
    for label, texts, target, encoder in (
        ("Запросы", messages, queries, embeddings.embed_query),
        ("Вопросы и ответы", documents, passages, embeddings.embed_passage),
        ("Контекстные пары", contexts, pairs, embeddings.embed_passage),
    ):
        for index, text in enumerate(texts, 1):
            started = perf_counter()
            target[text] = await encoder(text)
            if label == "Запросы":
                query_times.append((perf_counter() - started) * 1000)
            if index % 50 == 0 or index == len(texts):
                print(f"{label}: {index}/{len(texts)}", flush=True)
    reports = {strategy: {
        "format_version": 1, "name": f"frida-{strategy}",
        "contract": FRIDA_CONTRACT,
        "annotation_status": "provisional_not_owner_confirmed",
        "calibration": f"F_beta_{beta}_main_early_only_frozen_60_percent_per_chat",
        "beta": beta,
        "split_source": split_hash,
        "filter_sha256": file_hash(Path(__file__).parent / "services/questions.py"),
        "ranking": strategy, "scenarios": [],
        "timing": {"single_query_mean_ms": float(np.mean(query_times)),
                   "single_query_p95_ms": float(np.percentile(query_times, 95)),
                   "excludes_sqlite_telegram": True},
    } for strategy in STRATEGIES}
    reports["rerank_top5"]["reranker"] = {
        "model": RERANKER_MODEL, "revision": RERANKER_REVISION,
        "file": RERANKER_FILE, "top_k": 5,
        "pair_format": "question_and_answer", "score_is_probability": False,
    }
    thresholds, score_cache, rerank_times = {}, {}, []
    for folder, paths, cases, bank, excluded in datasets:
        bank_ids = {(s.chat, s.id): s for s in bank}
        outcomes = {s: [] for s in STRATEGIES}
        details = {s: [] for s in STRATEGIES}
        for index, case in enumerate(cases, 1):
            rankings = {s: [] for s in STRATEGIES}
            if is_question_candidate(case.message) or case.expected:
                base = retrieve(case, bank, queries, passages, pairs)
                rankings["baseline"] = base
                rankings["context_pair"] = sorted(
                    base, key=lambda r: -r["context_score"]
                )
                top = [dict(row) for row in base[:5]]
                keys = [(case.message, pair_text(bank_ids[(case.chat, r["id"])]))
                        for r in top]
                missing = list(dict.fromkeys(k for k in keys if k not in score_cache))
                if missing:
                    started = perf_counter()
                    scores = await asyncio.to_thread(
                        reranker.predict, case.message, [k[1] for k in missing]
                    )
                    if len(scores) != len(missing) or not np.all(np.isfinite(scores)):
                        raise ValueError("Оценщик вернул неверные scores")
                    rerank_times.append((perf_counter() - started) * 1000)
                    score_cache.update(zip(missing, scores, strict=True))
                for row, key in zip(top, keys, strict=True):
                    row["rerank_score"] = score_cache[key]
                rankings["rerank_top5"] = sorted(top, key=lambda r: -r["rerank_score"])
            for strategy in STRATEGIES:
                ranking = rankings[strategy]
                outcome = to_outcome(case, ranking, strategy)
                outcomes[strategy].append(outcome)
                details[strategy].append({
                    "chat": case.chat, "message_id": case.message_id,
                    "split": "validation" if (case.chat, case.message_id)
                    in early_ids else "test", "filter_passed": outcome.passed,
                    "expected": sorted(case.expected), "best": outcome.best_id,
                    "score": outcome.score, "gap": outcome.gap,
                    "candidates": ranking[:5],
                })
            if index % 100 == 0 or index == len(cases):
                print(f"{folder.name}: ранжирование {index}/{len(cases)}", flush=True)
        for strategy in STRATEGIES:
            values = tuple(outcomes[strategy])
            early = tuple(o for o in values if (o.case.chat, o.case.message_id)
                          in early_ids)
            late = tuple(o for o in values if (o.case.chat, o.case.message_id)
                         not in early_ids)
            if strategy not in thresholds:
                thresholds[strategy] = select_threshold(early, beta=beta)
            threshold = thresholds[strategy]
            coverage = {}
            for split in ("validation", "test"):
                positive = [r for r in details[strategy]
                            if r["split"] == split and r["expected"]]
                coverage[split] = {
                    "positives": len(positive),
                    "with_expected_top5": sum(any(c["id"] in r["expected"]
                                                  for c in r["candidates"])
                                             for r in positive),
                }
            scenario = {
                "name": folder.name, "inputs": [input_hashes[p] for p in paths],
                "dataset": {"cases": len(cases), "solutions": len(bank),
                            "excluded": excluded},
                "threshold": threshold, "validation": metrics(early, threshold),
                "test": metrics(late, threshold), "all": metrics(values, threshold),
                "candidate_coverage": coverage, "details": details[strategy],
            }
            if strategy == "baseline":
                scenario["fixed_0_458"] = metrics(late, 0.458)
            reports[strategy]["scenarios"].append(scenario)
            m = scenario["test"]
            print(f"{strategy} / {folder.name}: поздние — верных {m['correct']}, "
                  f"ошибок {m['wrong']}, пропусков {m['missed']}", flush=True)
    reports["rerank_top5"]["timing"].update({
        "rerank_call_mean_ms": float(np.mean(rerank_times)) if rerank_times else 0,
        "rerank_call_p95_ms": float(np.percentile(rerank_times, 95))
        if rerank_times else 0, "cached_pair_scores": len(score_cache),
        "includes_first_reranker_load": True,
        "warm_rerank_call_mean_ms": float(np.mean(rerank_times[1:]))
        if len(rerank_times) > 1 else None,
        "warm_rerank_call_p95_ms": float(np.percentile(rerank_times[1:], 95))
        if len(rerank_times) > 1 else None,
    })
    if (file_hash(args.split_cases) != split_hash
            or any(file_hash(path) != value for path, value in input_hashes.items())):
        raise ValueError("Исходные CSV изменились во время эксперимента")
    try:
        reports["rerank_recall_guard"] = recall_guard_report(
            reports["baseline"], reports["rerank_top5"], beta
        )
    except ValueError as error:
        reports["rerank_top5"]["recall_guard_unavailable"] = str(error)
    args.output.mkdir(parents=True, exist_ok=True)
    for strategy, report in reports.items():
        path = args.output / f"{strategy}.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Улучшения поиска с полной FRIDA офлайн"
    )
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--split-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Укажите новую папку: предыдущие эксперименты сохраняются")
    try:
        asyncio.run(run(args, FridaEmbeddingService(), OnnxReranker(local_only=True)))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
