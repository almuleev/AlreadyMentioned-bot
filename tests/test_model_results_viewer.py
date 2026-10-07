import csv
import hashlib
import json

import pytest

from already_mentioned.model_results_viewer import build_data, write_viewer


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    records = {
        "reviewed_cases_full.csv": [{
            "chat": "A", "message_id": "10", "new_message": "</script><b>Вход?</b>",
            "expected_solution_id": "s1",
        }],
        "bank.csv": [{
            "chat": "A", "solution_id": "s1", "answer_message_id": "2",
            "saved_question": "Где вход?", "saved_answer": "В меню",
            "answer_speaker": "Администратор", "answer_date": "2026-01-01",
            "historical_proxy_status": "usable_candidate",
        }],
    }
    for name, rows in records.items():
        with (folder / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
    report = {"format_version": 1, "name": "frida", "contract": {"model": "FRIDA"},
              "scenarios": [{
                  "name": "main", "threshold": 0.5,
                  "inputs": [{"name": name, "sha256": hashlib.sha256(
                      (folder / name).read_bytes()).hexdigest()} for name in records],
                  "details": [{"chat": "A", "message_id": 10, "split": "test",
                               "filter_passed": True, "expected": ["s1"],
                               "best": "s1", "score": 0.7, "gap": None}],
              }]}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return folder, path, report


def test_join_keeps_source_and_escapes_script(inputs, tmp_path):
    folder, path, _ = inputs
    data = build_data([path], [folder])
    scenario = data["scenarios"]["main"]
    assert scenario["bank"]["A"]["s1"]["answer_speaker"] == "Администратор"
    assert scenario["models"][0]["details"]["A:10"]["score"] == 0.7
    output = tmp_path / "viewer.html"
    write_viewer(data, output)
    html = output.read_text(encoding="utf-8")
    assert "</script><b>" not in html
    assert "\\u003c/script>" in html
    assert "__DATA__" not in html


def test_rejects_changed_csv(inputs):
    folder, path, _ = inputs
    with (folder / "bank.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="хэшами"):
        build_data([path], [folder])


@pytest.mark.parametrize("change", ["missing", "label", "duplicate", "candidate"])
def test_rejects_invalid_report_cases(inputs, change):
    folder, path, report = inputs
    row = report["scenarios"][0]["details"][0]
    if change == "missing":
        report["scenarios"][0]["details"] = []
    elif change == "label":
        row["expected"] = []
    elif change == "candidate":
        row["best"] = "absent"
    else:
        report["scenarios"][0]["details"].append(row.copy())
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        build_data([path], [folder])


def test_rejects_mixed_splits(inputs, tmp_path):
    folder, path, report = inputs
    report["name"] = "other"
    report["scenarios"][0]["details"][0]["split"] = "validation"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Разбиение"):
        build_data([path, other], [folder])


def test_viewer_keeps_component_scores_and_rejects_duplicate_candidates(inputs):
    folder, path, report = inputs
    row = report["scenarios"][0]["details"][0]
    candidate = {"id": "s1", "question_score": 0.6, "answer_score": 0.7,
                 "score": 0.7, "context_score": 0.8, "rerank_score": 0.9}
    row["candidates"] = [candidate]
    path.write_text(json.dumps(report), encoding="utf-8")
    data = build_data([path], [folder])
    scores = data["scenarios"]["main"]["models"][0]["details"]["A:10"]["candidates"]
    assert scores == [candidate]
    row["candidates"].append(candidate.copy())
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="повторный кандидат"):
        build_data([path], [folder])


def test_current_model_label_uses_actual_contract(inputs):
    folder, path, report = inputs
    report["name"] = "current"
    path.write_text(json.dumps(report), encoding="utf-8")
    data = build_data([path], [folder])
    assert data["scenarios"]["main"]["models"][0]["name"].startswith("FRIDA (current)")
