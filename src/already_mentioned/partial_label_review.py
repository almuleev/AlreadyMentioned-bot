"""Versioned manual-label policy: substantive partial answers are acceptable."""


def build_cases(history, previous, bank, annotations):
    """Merge explicit reviews only; never infer correctness from model scores.

    annotations: {(chat, message_id, solution_id): (F/P/N/U, reason)}.
    Source IDs, dates, aliases, context and author provenance are retained.
    """
    prior = {(r["chat"], int(r["message_id"])): r for r in previous}
    all_solutions = {(r["chat"], r["solution_id"]): r for r in bank}
    solutions = {
        key: row
        for key, row in all_solutions.items()
        if row.get("historical_proxy_status", "usable_candidate") == "usable_candidate"
    }
    if len(prior) != len(previous) or len(all_solutions) != len(bank):
        raise ValueError("Повторные ID источников")
    result, seen = [], set()
    for original in history:
        key = original["chat"], int(original["message_id"])
        if key in seen:
            raise ValueError("Повторный ID сообщения")
        seen.add(key)
        row = {
            **original,
            **prior.get(key, {}),
            "new_message": original["message_text"],
        }
        label = row.get("expected_solution_id", "").strip() or "UNRESOLVED"
        expected = (
            set() if label in {"SILENCE", "UNRESOLVED"} else set(label.split("|"))
        )
        reasons = []
        for (chat, message, solution_id), (rating, reason) in annotations.items():
            if (chat, message) != key:
                continue
            if rating not in {"F", "P", "N", "U"}:
                raise ValueError("Неизвестная ручная оценка")
            solution = solutions.get((chat, solution_id))
            # Inferred-organizer solutions belong only to their separate scenario.
            if solution is None:
                continue
            if int(solution["answer_message_id"]) >= message:
                raise ValueError("Решение должно предшествовать вопросу")
            if rating in {"F", "P"}:
                expected.add(solution_id)
                reasons.append(f"{solution_id}: {rating}: {reason}")
            elif rating == "N":
                expected.discard(solution_id)
        row["expected_solution_id"] = (
            "|".join(sorted(expected))
            if expected
            else "UNRESOLVED"
            if label == "UNRESOLVED"
            else "SILENCE"
        )
        row["label_policy"] = "substantive_full_or_partial_v1"
        if reasons:
            row["partial_review_evidence"] = " | ".join(reasons)
        result.append(row)
    if not prior.keys() <= seen:
        raise ValueError("История не содержит все прежние случаи")
    return result
