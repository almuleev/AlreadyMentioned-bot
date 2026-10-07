"""Build a private offline HTML viewer from hash-matched historical reports."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def build_data(reports: list[Path], datasets: list[Path]) -> dict:
    result = {"scenarios": {}}
    for path in reports:
        report_bytes = path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
        report_hash = hashlib.sha256(report_bytes).hexdigest()
        if report.get("format_version") != 1 or not report.get("scenarios"):
            raise ValueError(f"Нужен полный historical_model_comparison отчёт: {path}")
        for scenario in report["scenarios"]:
            inputs = scenario["inputs"]
            if not inputs or any(Path(i["name"]).name != i["name"] for i in inputs):
                raise ValueError("Недопустимые имена входных CSV")
            matching = [folder for folder in datasets if all(
                (folder / i["name"]).is_file()
                and hashlib.sha256((folder / i["name"]).read_bytes()).hexdigest()
                == i["sha256"] for i in inputs
            )]
            if not matching:
                raise ValueError(f"Нет CSV с совпадающими хэшами: {path.name} / "
                                 f"{scenario['name']}")
            folder = matching[0]
            signature = sorted((i["name"], i["sha256"]) for i in inputs)
            key = scenario["name"]
            if key not in result["scenarios"]:
                cases = read_csv(folder / "reviewed_cases_full.csv")
                bank = {}
                for item in inputs:
                    if item["name"] == "reviewed_cases_full.csv":
                        continue
                    for row in read_csv(folder / item["name"]):
                        if row["historical_proxy_status"] == "usable_candidate":
                            bank.setdefault(row["chat"], {})[row["solution_id"]] = row
                result["scenarios"][key] = {
                    "signature": signature, "bank": bank, "models": [],
                    "cases": {f"{r['chat']}:{int(r['message_id'])}": r for r in cases
                              if r["expected_solution_id"] != "UNRESOLVED"},
                }
            target = result["scenarios"][key]
            if any(m["report_sha256"] == report_hash for m in target["models"]):
                raise ValueError("Один отчёт указан повторно")
            if target["signature"] != signature:
                raise ValueError(f"Разные версии данных в сценарии {key}")
            details = {}
            for row in scenario["details"]:
                case_key = f"{row['chat']}:{row['message_id']}"
                case = target["cases"].get(case_key)
                expected = (
                    case["expected_solution_id"].split("|")
                    if case and case["expected_solution_id"] != "SILENCE" else []
                )
                if case is None or sorted(expected) != sorted(row["expected"]):
                    raise ValueError("Случай или метка не совпадает с CSV")
                if case_key in details or row["split"] not in {"validation", "test"}:
                    raise ValueError("Повторный случай или неверная часть истории")
                if not math.isfinite(row["score"]):
                    raise ValueError("Некорректная оценка сходства")
                best = row["best"]
                if best is not None:
                    solution = target["bank"].get(row["chat"], {}).get(best)
                    if (solution is None or int(solution["answer_message_id"])
                            >= row["message_id"]):
                        raise ValueError(
                            "Кандидат должен предшествовать сообщению в чате"
                        )
                candidate_ids = set()
                for candidate in row.get("candidates", []):
                    candidate_id = candidate["id"]
                    solution = target["bank"].get(row["chat"], {}).get(candidate_id)
                    if (solution is None or candidate_id in candidate_ids
                            or int(solution["answer_message_id"]) >= row["message_id"]):
                        raise ValueError("Неверный или повторный кандидат")
                    if any(not math.isfinite(value) for name, value in candidate.items()
                           if name.endswith("score")):
                        raise ValueError("Некорректная оценка кандидата")
                    candidate_ids.add(candidate_id)
                details[case_key] = row
            if details.keys() != target["cases"].keys():
                raise ValueError("Отчёт содержит неполный набор сообщений")
            if target["models"] and any(
                row["split"] != target["models"][0]["details"][case_key]["split"]
                or row["filter_passed"]
                != target["models"][0]["details"][case_key]["filter_passed"]
                for case_key, row in details.items()
            ):
                raise ValueError("Разбиение или фильтр различаются между моделями")
            model_name = report["name"]
            model_name = {
                "frida-baseline": "FRIDA · исходный поиск",
                "frida-context_pair": "FRIDA · вопрос и ответ вместе",
                "frida-rerank_top5": "FRIDA · повторное ранжирование пяти",
                "frida-rerank_recall_guard": "FRIDA · ограничение потерь полноты",
                "frida-rerank_pair_single": "FRIDA · оценщик по одному кандидату",
                "frida-rerank_plain_pair": "FRIDA · оценщик без заголовков",
                "frida-rerank_answer_single": "FRIDA · оценщик только ответа",
                "frida-balanced_top5": "FRIDA · среднее вопроса и ответа",
                "frida-hybrid_top5": "FRIDA · 75% сходства и 25% оценщика",
                "frida-gte-fp32-rerank_pair_single": "FRIDA · оценщик GTE FP32",
                "frida-gte-fp32-hybrid_top5": "FRIDA · 75% сходства и 25% GTE FP32",
                "frida-baseline_transport_filter": "FRIDA · фильтр транспорта",
                "frida-hybrid_transport_filter": "FRIDA · смесь и фильтр транспорта",
            }.get(model_name, model_name)
            if model_name == "current":
                model_name = f"{report['contract']['model']} (current)"
            label = model_name
            if any(m["name"] == label for m in target["models"]):
                label = f"{model_name} · {path.parent.name}"
            if any(m["name"] == label for m in target["models"]):
                raise ValueError("Повторная модель: задайте отдельные папки прогонов")
            target["models"].append({
                "name": label, "report": str(path.resolve()),
                "report_sha256": report_hash,
                "threshold": scenario["threshold"], "details": details,
                "contract": report["contract"], "timing": report.get("timing", {}),
                "memory": report.get("process_memory", {}),
                "reranker": report.get("reranker"),
            })
    return result


def write_viewer(data: dict, output: Path):
    template = Path(__file__).with_name("model_results_viewer.html").read_text(
        encoding="utf-8"
    )
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__DATA__", payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Локальный просмотр реакций моделей")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("training_runs/model_viewer/index.html"))
    args = parser.parse_args()
    if args.output.suffix != ".html" or args.output.resolve() in {
        p.resolve() for p in args.reports
    }:
        parser.error("Выход должен быть отдельным HTML-файлом")
    try:
        data = build_data(args.reports, args.datasets)
        write_viewer(data, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Просмотр готов: {args.output.resolve()}")


if __name__ == "__main__":
    main()
