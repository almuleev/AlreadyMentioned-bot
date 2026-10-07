"""Offline input/score ablation on hash-matched FRIDA top-five candidates."""

import argparse
import copy
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from already_mentioned.historical_evaluation import HistoricalCase, Outcome, metrics
from already_mentioned.model_results_viewer import build_data
from already_mentioned.services.embeddings import FRIDA_CONTRACT
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reranking import (
    RERANKER_FILE,
    RERANKER_MODEL,
    RERANKER_REVISION,
    GteOnnxReranker,
    OnnxReranker,
)
from already_mentioned.threshold_calibration import file_hash, select_threshold

STRATEGIES = (
    "rerank_pair_single", "rerank_plain_pair", "rerank_answer_single",
    "balanced_top5", "hybrid_top5",
)
FORMATS = ("labelled_pair", "plain_pair", "answer")

TRANSPORT_ISSUE = re.compile(
    r"\b(?:не\s+успева(?:ю|ем|ет|ют)|не\s+успе(?:ю|ем|ет|ют)|"
    r"задерж(?:али|ивается|иваются))\b", re.IGNORECASE,
)
TRANSPORT_SUBJECT = re.compile(
    r"\b(?:рейс\w*|самол[её]т\w*|автобус\w*|шаттл\w*|трансфер\w*)\b",
    re.IGNORECASE,
)


def experimental_transport_filter(text):
    """Offline hypothesis for implicit help requests; never called by the bot."""
    stripped = text.strip()
    return is_question_candidate(text) or bool(
        not stripped.startswith("/") and TRANSPORT_ISSUE.search(stripped)
        and TRANSPORT_SUBJECT.search(stripped)
    )


def passage(solution, mode):
    if mode == "labelled_pair":
        return (f"Вопрос: {solution['saved_question']}\n"
                f"Ответ: {solution['saved_answer']}")
    if mode == "plain_pair":
        return f"{solution['saved_question']}\n{solution['saved_answer']}"
    if mode == "answer":
        return solution["saved_answer"]
    raise ValueError("Неизвестный формат пары")


def variants(candidates, scores, strategies=STRATEGIES):
    """Fixed weights, selected before evaluation; no label-dependent rules."""
    result = {}
    for strategy in strategies:
        rows = []
        for index, candidate in enumerate(candidates):
            row = dict(candidate)
            row.update({f"{mode}_rerank_score": values[index]
                        for mode, values in scores.items()})
            if strategy == "balanced_top5":
                value = (row["question_score"] + row["answer_score"]) / 2
            elif strategy == "hybrid_top5":
                value = 0.75 * row["score"] + 0.25 * scores["labelled_pair"][index]
            else:
                mode = {"rerank_pair_single": "labelled_pair",
                        "rerank_plain_pair": "plain_pair",
                        "rerank_answer_single": "answer"}[strategy]
                value = scores[mode][index]
            row["experiment_score"] = value
            rows.append(row)
        result[strategy] = sorted(rows, key=lambda r: -r["experiment_score"])
    return result


def outcomes(rows):
    return tuple(Outcome(
        HistoricalCase(r["chat"], r["message_id"], "", frozenset(r["expected"])),
        r["filter_passed"], r["best"], r["score"], r["gap"],
    ) for r in rows)


def run(args, scorer):
    beta = getattr(args, "beta", 1.1)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Beta должна быть положительной")
    strategies = tuple(getattr(args, "strategies", None) or STRATEGIES)
    if not strategies or len(set(strategies)) != len(strategies) or (
        not set(strategies) <= set(STRATEGIES)
    ):
        raise ValueError("Некорректный набор стратегий")
    modes = {"rerank_pair_single": "labelled_pair",
             "rerank_plain_pair": "plain_pair", "rerank_answer_single": "answer",
             "hybrid_top5": "labelled_pair"}
    formats = tuple(mode for mode in FORMATS if mode in
                    {modes[s] for s in strategies if s in modes})
    if args.output.exists():
        raise ValueError("Укажите новую папку результатов")
    report_bytes = args.candidates_from.read_bytes()
    source = json.loads(report_bytes)
    if source["contract"] != FRIDA_CONTRACT or source.get("ranking") != "baseline":
        raise ValueError("Нужен исходный отчёт поиска полной FRIDA")
    if source["split_source"] != file_hash(args.split_cases):
        raise ValueError("Не совпадает источник разбиения")
    filter_path = Path(__file__).parent / "services/questions.py"
    if source["filter_sha256"] != file_hash(filter_path):
        raise ValueError("Фильтр изменился: повторите исходный эксперимент")
    data = build_data([args.candidates_from], args.datasets)
    if not data["scenarios"]:
        raise ValueError("Нет сценариев")
    # Single-pair inference prevents dependence on neighbouring candidates/padding.
    if scorer.batch_size != 1:
        raise ValueError("Для сравнения нужен batch_size=1")
    reports = {}
    for strategy in strategies:
        report = copy.deepcopy(source)
        prefix = getattr(scorer, "experiment_prefix", "frida")
        report.update(name=f"{prefix}-{strategy}", ranking=strategy, scenarios=[],
                      calibration=f"F_beta_{beta}_main_early_only_reused_frozen_split",
                      beta=beta,
                      reused_candidates=file_hash(args.candidates_from),
                      timing={"frida_loaded": False, "query_encoding_measured": False,
                              "excludes_sqlite_telegram": True},
                      candidate_scope="original_baseline_top5_only")
        if strategy != "balanced_top5":
            report["reranker"] = {
                "model": getattr(scorer, "model", RERANKER_MODEL),
                "revision": getattr(scorer, "revision", RERANKER_REVISION),
                "file": getattr(scorer, "file", RERANKER_FILE),
                "top_k": 5, "batch_size": 1,
                "max_length": scorer.max_length, "score_is_probability": False,
                "pair_format": {"rerank_pair_single": "labelled_pair",
                                "rerank_plain_pair": "plain_pair",
                                "rerank_answer_single": "answer",
                                "hybrid_top5": "labelled_pair"}[strategy],
            }
        if strategy == "hybrid_top5":
            report["weights"] = {"frida_max": 0.75, "labelled_pair_rerank": 0.25}
        elif strategy == "balanced_top5":
            report["weights"] = {"question_similarity": 0.5, "answer_similarity": 0.5}
        reports[strategy] = report
    cache, durations, thresholds = {}, {mode: [] for mode in formats}, {}
    for scenario_source in source["scenarios"]:
        sn = scenario_source["name"]
        scenario = data["scenarios"][sn]
        model = scenario["models"][0]
        all_details = {strategy: [] for strategy in strategies}
        for index, (key, case) in enumerate(scenario["cases"].items(), 1):
            original = model["details"][key]
            candidates = original.get("candidates", [])
            if len(candidates) > 5 or (
                original["best"] is not None
                and (not candidates or candidates[0]["id"] != original["best"])
            ):
                raise ValueError("Нужны сохранённые первые пять исходных кандидатов")
            if any(c["score"] != max(c["question_score"], c["answer_score"])
                   for c in candidates):
                raise ValueError("Неверные исходные оценки FRIDA")
            scores = {mode: [] for mode in formats}
            for mode in formats:
                for candidate in candidates:
                    solution = scenario["bank"][case["chat"]][candidate["id"]]
                    text = passage(solution, mode)
                    pair_key = (case["new_message"], text)
                    if pair_key not in cache:
                        started = perf_counter()
                        value = scorer.predict(case["new_message"], [text])
                        if (len(value) != 1 or not np.isfinite(value[0])
                                or not 0 <= value[0] <= 1):
                            raise ValueError("Оценщик вернул неверную оценку")
                        durations[mode].append((perf_counter() - started) * 1000)
                        cache[pair_key] = float(value[0])
                    scores[mode].append(cache[pair_key])
            for strategy, ranking in variants(candidates, scores, strategies).items():
                detail = {field: original[field] for field in
                          ("chat", "message_id", "split", "filter_passed", "expected")}
                value = ranking[0]["experiment_score"] if ranking else -2.0
                detail.update(best=ranking[0]["id"] if ranking else None, score=value,
                              gap=value - ranking[1]["experiment_score"]
                              if len(ranking) > 1 else None, candidates=ranking)
                all_details[strategy].append(detail)
            if index % 100 == 0:
                print(f"{sn}: {index}/{len(scenario['cases'])}", flush=True)
        for strategy, details in all_details.items():
            early = outcomes([r for r in details if r["split"] == "validation"])
            late = outcomes([r for r in details if r["split"] == "test"])
            if strategy not in thresholds:
                thresholds[strategy] = select_threshold(early, beta=beta)
            threshold = thresholds[strategy]
            output = copy.deepcopy(scenario_source)
            output.pop("fixed_0_458", None)
            output.update(details=details, threshold=threshold,
                          validation=metrics(early, threshold),
                          test=metrics(late, threshold),
                          all=metrics(outcomes(details), threshold))
            reports[strategy]["scenarios"].append(output)
            print(f"{strategy}: порог {threshold}; ранняя {output['validation']}; "
                  f"поздняя {output['test']}", flush=True)
    # Revalidate inputs before saving, so the reports cannot join modified CSVs.
    if args.candidates_from.read_bytes() != report_bytes:
        raise ValueError("Отчёт кандидатов изменился")
    if source["split_source"] != file_hash(args.split_cases):
        raise ValueError("Источник разбиения изменился")
    if source["filter_sha256"] != file_hash(filter_path):
        raise ValueError("Фильтр изменился во время эксперимента")
    build_data([args.candidates_from], args.datasets)
    for report in reports.values():
        report["timing"]["ablation_inference"] = {
            mode: {"unique_calls": len(times), "mean_ms": float(np.mean(times))
                   if times else None, "p95_ms": float(np.percentile(times, 95))
                   if times else None} for mode, times in durations.items()
        }
        report["timing"].update(total_unique_pairs=len(cache),
                                includes_first_reranker_load=True,
                                shared_experiment_cost_not_strategy_latency=True)
        report["tokenization"] = {"total_pairs": scorer.total_pairs,
                                  "truncated_pairs": scorer.truncated_pairs}
    args.output.mkdir(parents=True)
    for strategy, report in reports.items():
        with (args.output / f"{strategy}.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser(description="Форматы оценщика на кандидатах FRIDA")
    parser.add_argument("--candidates-from", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--split-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.1)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    parser.add_argument("--reranker", choices=("minilm-int8", "gte-fp32"),
                        default="minilm-int8")
    parser.add_argument("--download-model", action="store_true",
                        help="Разрешить загрузку закреплённых файлов модели")
    args = parser.parse_args()
    try:
        scorer_class = GteOnnxReranker if args.reranker == "gte-fp32" else OnnxReranker
        run(args, scorer_class(local_only=not args.download_model, batch_size=1))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
