"""Data returned from the SQLite repositories."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chat:
    telegram_chat_id: int
    title: str
    similarity_threshold: float


@dataclass(frozen=True, slots=True)
class Solution:
    id: int
    chat_id: int
    question_message_id: int
    answer_message_id: int
    question_text: str
    answer_text: str
    question_embedding: bytes
    question_link: str
    answer_link: str


@dataclass(frozen=True, slots=True)
class Feedback:
    id: int
    solution_id: int
    query_message_id: int
    user_id: int
    vote: str
