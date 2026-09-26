"""Conservative rule for finding likely Russian-language questions."""

import re

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
QUESTION_START = re.compile(
    r"^(?:как|где|когда|почему|зачем|что|кто|куда|откуда|сколько|"
    r"можно ли|нужно ли|есть ли|подскажите|помогите|"
    r"не могу|не вижу|не получается|не работает|не открывается|"
    r"не приходит|ошибка|проблема)\b",
    re.IGNORECASE,
)


def is_question_candidate(text: str) -> bool:
    """Avoid embeddings for commands, ordinary chat, and non-Russian text."""
    stripped = text.strip()
    if len(stripped) < 7 or stripped.startswith("/") or not CYRILLIC.search(stripped):
        return False
    return "?" in stripped or "؟" in stripped or bool(QUESTION_START.search(stripped))
