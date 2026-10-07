"""Token-free replay of the production search over an explicit historical export."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from already_mentioned.historical_evaluation import (
    HistoricalCase,
    Outcome,
    load_dataset,
    read_rows,
    split_cases,
)
from already_mentioned.historical_evaluation import (
    metrics as historical_metrics,
)
from already_mentioned.models.entities import Chat, Solution
from already_mentioned.services.embeddings import FRIDA_CONTRACT, FridaEmbeddingService
from already_mentioned.services.questions import is_question_candidate
from already_mentioned.services.reranking import (
    RERANKER_FILE,
    RERANKER_MODEL,
    RERANKER_REVISION,
    OnnxReranker,
)
from already_mentioned.services.search import SearchService
from already_mentioned.threshold_calibration import file_hash


def metrics(outcomes, threshold):
    """Classify emitted decisions; abstained leaders are not retained here."""
    result = historical_metrics(outcomes, threshold)
    for key in ("wrong_leader", "below_threshold", "correct_leader_without_filter"):
        result.pop(key)
    return result


async def run(args):
    if args.output.exists():
        raise ValueError("Нужен новый файл результатов")
    folder = args.dataset
    paths = [
        folder / "reviewed_cases_full.csv",
        *sorted(folder.glob("chat*_inferred_solution_bank.csv")),
    ]
    labelled, bank, excluded = load_dataset(paths[0], paths[1:])
    raw = read_rows(paths[0], {"chat", "message_id", "new_message"})
    hashes = [file_hash(p) for p in paths]
    early, _ = split_cases(
        tuple(
            HistoricalCase(r["chat"], int(r["message_id"]), "", frozenset())
            for r in raw
        ),
        0.6,
    )
    known = {(c.chat, c.message_id): c for c in labelled}
    chats = {
        name: index + 1 for index, name in enumerate(sorted({r["chat"] for r in raw}))
    }
    encoder = FridaEmbeddingService()
    started = perf_counter()
    await encoder.embed_query("Проверка загрузки")
    reranker = (
        OnnxReranker(local_only=True, batch_size=1) if args.mode == "hybrid" else None
    )
    if reranker:
        await asyncio.to_thread(reranker.predict, "Проверка загрузки", ["Проверка"])
    load_seconds = perf_counter() - started
    process = psutil.Process()
    rss_after_load = process.memory_info().rss
    documents = {}
    for index, text in enumerate(
        sorted({t for s in bank for t in (s.question, s.answer)}), 1
    ):
        documents[text] = (await encoder.embed_passage(text)).tobytes()
        if index % 50 == 0:
            print(f"Документы: {index}", flush=True)
    by_chat = {}
    original_ids = {}
    for index, item in enumerate(bank, 1):
        original_ids[index] = item.id
        by_chat.setdefault(chats[item.chat], []).append(
            Solution(
                index,
                chats[item.chat],
                0,
                item.answer_message_id,
                item.question,
                item.answer,
                documents[item.question],
                "offline-question",
                "offline-answer",
                documents[item.answer],
            )
        )

    class Chats:
        async def get_chat(self, chat_id):
            return Chat(
                chat_id, "Офлайн", args.baseline_threshold, args.hybrid_threshold
            )

    class Solutions:
        message_id = 0

        async def set_embedding(self, *args):
            raise RuntimeError("Офлайн-векторы должны быть полными")

        async def set_answer_embedding(self, *args):
            raise RuntimeError("Офлайн-векторы должны быть полными")

        async def list_for_chat(self, chat_id):
            return [
                s
                for s in by_chat.get(chat_id, [])
                if s.answer_message_id < self.message_id
            ]

    solutions = Solutions()
    search = SearchService(
        encoder, Chats(), solutions, mode=args.mode, reranker=reranker
    )
    outcomes, details, times = [], [], []
    filtered_requests = 0
    sampled_rss = [rss_after_load, process.memory_info().rss]
    for index, row in enumerate(
        sorted(raw, key=lambda r: (r["chat"], int(r["message_id"]))), 1
    ):
        message = int(row["message_id"])
        solutions.message_id = message
        passed = is_question_candidate(row["new_message"])
        searchable = passed and bool(await solutions.list_for_chat(chats[row["chat"]]))
        begin = perf_counter()
        match = (
            await search.find_best(chats[row["chat"]], row["new_message"])
            if passed
            else None
        )
        duration = (perf_counter() - begin) * 1000
        if passed:
            filtered_requests += 1
        if searchable:
            times.append(duration)
        best = original_ids[match.solution.id] if match else None
        case = known.get((row["chat"], message))
        if case:
            outcomes.append(
                Outcome(case, passed, best, match.similarity if match else -2.0, None)
            )
        details.append(
            {
                "chat": row["chat"],
                "message_id": message,
                "filter_passed": passed,
                "best": best,
                "score": match.similarity if match else None,
                "labelled": case is not None,
                "elapsed_ms": duration,
            }
        )
        sampled_rss.append(process.memory_info().rss)
        if index % 100 == 0:
            print(f"Рабочий поиск: {index}/{len(raw)}", flush=True)
    if hashes != [file_hash(p) for p in paths]:
        raise ValueError("Источники изменились")
    threshold = (
        args.hybrid_threshold if args.mode == "hybrid" else args.baseline_threshold
    )
    result = {
        "date": "2026-10-08",
        "production_search_replayed": True,
        "telegram_used": False,
        "sqlite_used": False,
        "contract": FRIDA_CONTRACT,
        "mode": args.mode,
        "threshold": threshold,
        "threshold_calibrated_here": False,
        "candidate_leaders_for_abstentions_recorded": False,
        "inputs": hashes,
        "history_rows": len(raw),
        "labelled": len(labelled),
        "excluded_unknown": excluded,
        "validation": metrics(
            tuple(o for o in outcomes if (o.case.chat, o.case.message_id) in early),
            threshold,
        ),
        "test": metrics(
            tuple(o for o in outcomes if (o.case.chat, o.case.message_id) not in early),
            threshold,
        ),
        "all": metrics(tuple(outcomes), threshold),
        "details": details,
        "timing": {
            "load_seconds": load_seconds,
            "filtered_requests": filtered_requests,
            "searchable_requests": len(times),
            "warm_mean_ms": float(np.mean(times)) if times else None,
            "warm_p95_ms": float(np.percentile(times, 95)) if times else None,
            "includes_real_query_encoding_and_top5_reranking": True,
            "excludes_telegram_sqlite_queue": True,
        },
        "memory": {
            "rss_after_joint_load_bytes": rss_after_load,
            "max_post_request_rss_bytes": max(sampled_rss),
        },
        "reranker": {
            "model": RERANKER_MODEL,
            "revision": RERANKER_REVISION,
            "file": RERANKER_FILE,
            "batch_size": 1,
            "max_length": 512,
        }
        if reranker
        else None,
    }
    try:
        import resource

        result["memory"]["whole_process_peak_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
    except ImportError:
        result["memory"]["whole_process_peak_bytes"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"details", "inputs"}},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Офлайн-повтор рабочего поиска без Telegram"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "hybrid"), required=True)
    parser.add_argument("--baseline-threshold", type=float, required=True)
    parser.add_argument("--hybrid-threshold", type=float, required=True)
    args = parser.parse_args()
    if any(
        not np.isfinite(t) or not 0 <= t <= 1
        for t in (args.baseline_threshold, args.hybrid_threshold)
    ):
        parser.error("Пороги должны быть от 0 до 1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
