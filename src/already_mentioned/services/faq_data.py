"""Bounded UTF-8 CSV parsing and literal FAQ duplicate review."""

import csv
import hashlib
import io
import json
from dataclasses import dataclass

MAX_FILE = 512 * 1024
MAX_ROWS = 50
MAX_TEXT = 3800  # UTF-16 units, reserving space for Telegram headings.
TEMPLATE = (
    'вопрос;ответ\n"Где найти материалы?";'
    '"Укажите здесь настоящую ссылку или инструкцию."\n'
).encode("utf-8-sig")


@dataclass(frozen=True, slots=True)
class FaqPair:
    row: int
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class FaqReview:
    pairs: tuple[FaqPair, ...]
    issues: tuple[str, ...] = ()
    duplicates: int = 0


def valid_text(text):
    return (
        bool(text.strip())
        and "\x00" not in text
        and len(text.encode("utf-16-le")) // 2 <= MAX_TEXT
    )


def pair_key(question, answer):
    normalized = [" ".join(text.split()).casefold() for text in (question, answer)]
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False).encode()
    ).hexdigest()


def review(pairs, existing, issues=()):
    known = {pair_key(s.question_text, s.answer_text) for s in existing}
    result, duplicates = [], 0
    for pair in pairs:
        key = pair_key(pair.question, pair.answer)
        if key in known:
            duplicates += 1
        else:
            known.add(key)
            result.append(pair)
    return FaqReview(tuple(result), tuple(issues), duplicates)


def parse_csv(data: bytes):
    if len(data) > MAX_FILE:
        raise ValueError("Файл больше 512 КиБ.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Сохраните CSV в UTF-8.") from error
    first = text.splitlines()[0] if text else ""
    delimiter = next(
        (
            d
            for d in (";", ",")
            if [c.strip().casefold() for c in next(csv.reader([first], delimiter=d))]
            == ["вопрос", "ответ"]
        ),
        None,
    )
    if delimiter is None:
        raise ValueError(
            "Нужны ровно две колонки: вопрос и ответ; разделитель ; или ,."
        )
    reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
    next(reader)
    pairs, issues, count = [], [], 0
    try:
        while True:
            line = reader.line_num + 1
            row = next(reader, None)
            if row is None:
                break
            if not row or all(not cell.strip() for cell in row):
                continue
            count += 1
            if count > MAX_ROWS:
                raise ValueError("В одной загрузке допускается до 50 пар.")
            if len(row) != 2 or not all(valid_text(cell) for cell in row):
                issues.append(
                    f"Строка {line}: нужны два непустых текста до 3800 знаков UTF-16."
                )
            else:
                pairs.append(FaqPair(line, row[0].strip(), row[1].strip()))
    except csv.Error as error:
        raise ValueError("Некорректные кавычки или структура CSV.") from error
    return tuple(pairs), tuple(issues)


class LimitedBuffer(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_FILE:
            raise ValueError("Файл больше 512 КиБ.")
        return super().write(data)
