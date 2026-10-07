import pytest

from already_mentioned.partial_label_review import build_cases


def test_partial_is_acceptable_unrelated_is_not_and_unknown_stays_unknown():
    history = [{"chat": "A", "message_id": str(i), "message_text": "Вопрос?",
                "date": "2026-01-01", "speaker_alias": "Участник",
                "expected_solution_id": label}
               for i, label in [(10, "SILENCE"), (11, "one"), (12, "UNRESOLVED")]]
    bank = [{"chat": "A", "solution_id": "one", "answer_message_id": "2"}]
    annotations = {("A", 10, "one"): ("P", "Отвечает на часть вопроса"),
                   ("A", 11, "one"): ("N", "Другой вопрос")}
    result = build_cases(history, [], bank, annotations)
    assert [r["expected_solution_id"] for r in result] == [
        "one", "SILENCE", "UNRESOLVED"
    ]
    assert result[0]["speaker_alias"] == "Участник"
    assert result[0]["date"] == "2026-01-01"
    assert "P:" in result[0]["partial_review_evidence"]
    # A solution from another bank must not silently enter this scenario.
    assert (build_cases(history, [], [], annotations)[0]["expected_solution_id"]
            == "SILENCE")
    unconfirmed = [{**bank[0], "historical_proxy_status": "needs_review"}]
    rejected = build_cases(history, [], unconfirmed, annotations)
    assert rejected[0]["expected_solution_id"] == "SILENCE"
    with pytest.raises(ValueError, match="предшествовать"):
        build_cases(history, [], [{**bank[0], "answer_message_id": "15"}], annotations)
