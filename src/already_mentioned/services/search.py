"""Find the single strongest confirmed answer within one chat."""

import asyncio
import math
from dataclasses import dataclass, replace

import numpy as np

from already_mentioned.config import DEFAULT_HYBRID_THRESHOLD
from already_mentioned.models.entities import Solution
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.embeddings import EmbeddingService, normalize_vector
from already_mentioned.services.reranking import PairScorer
from already_mentioned.services.similarity import cosine_similarity


@dataclass(frozen=True, slots=True)
class SearchMatch:
    solution: Solution
    similarity: float
    question_similarity: float | None = None
    answer_similarity: float | None = None


class SearchService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        chats: ChatRepository,
        solutions: SolutionRepository,
        *,
        mode: str = "baseline",
        reranker: PairScorer | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.chats = chats
        self.solutions = solutions
        if mode not in {"baseline", "hybrid"} or (
            mode == "hybrid" and reranker is None
        ):
            raise ValueError("Для режима hybrid нужен оценщик MiniLM")
        self.mode = mode
        self.reranker = reranker
        self._rerank_lock = asyncio.Lock()

    def threshold_for(self, chat) -> float:
        if self.mode == "baseline":
            return chat.similarity_threshold
        return (
            chat.hybrid_threshold
            if chat.hybrid_threshold is not None
            else DEFAULT_HYBRID_THRESHOLD
        )

    @property
    def mode_label(self) -> str:
        return "FRIDA + MiniLM" if self.mode == "hybrid" else "FRIDA"

    async def find_best(self, chat_id: int, question: str) -> SearchMatch | None:
        chat = await self.chats.get_chat(chat_id)
        if chat is None:
            return None
        ranked = await self.find_candidates(
            chat_id, question, limit=5 if self.mode == "hybrid" else 1
        )
        ranked = await self.rerank_candidates(question, ranked)
        if not ranked or ranked[0].similarity < self.threshold_for(chat):
            return None
        return ranked[0]

    async def rerank_candidates(self, question, ranked):
        """Apply the active decision score to the raw FRIDA top five."""
        if self.mode == "hybrid" and ranked:
            ranked = ranked[:5]
            async with self._rerank_lock:
                scores = await asyncio.to_thread(
                    self.reranker.predict,
                    question,
                    [
                        f"Вопрос: {m.solution.question_text}\n"
                        f"Ответ: {m.solution.answer_text}"
                        for m in ranked
                    ],
                )
            if len(scores) != len(ranked) or any(
                not math.isfinite(s) or not 0 <= s <= 1 for s in scores
            ):
                raise ValueError("Некорректные оценки MiniLM")
            ranked = sorted(
                (
                    replace(m, similarity=0.75 * m.similarity + 0.25 * score)
                    for m, score in zip(ranked, scores, strict=True)
                ),
                key=lambda m: -m.similarity,
            )
        return ranked

    async def find_candidates(
        self, chat_id: int, question: str, *, limit: int = 5, backfill: bool = True
    ) -> list[SearchMatch]:
        """Rank distinct same-chat solutions before threshold; encode the query once.

        Diagnostics disable backfill to preserve their read-only database access.
        """
        if not 1 <= limit <= 100:
            raise ValueError("Число кандидатов должно быть от 1 до 100")
        candidates = await self.solutions.list_for_chat(chat_id)
        if not candidates:
            return []

        query = normalize_vector(await self.embeddings.embed_query(question))
        ranked = []
        for solution in candidates:
            scores = []
            for text, data, save_embedding in (
                (
                    solution.question_text,
                    solution.question_embedding,
                    self.solutions.set_embedding,
                ),
                (
                    solution.answer_text,
                    solution.answer_embedding,
                    self.solutions.set_answer_embedding,
                ),
            ):
                if not data:
                    passage = normalize_vector(
                        await self.embeddings.embed_passage(text)
                    )
                    data = passage.tobytes()
                    if backfill:
                        await save_embedding(solution.id, data)
                try:
                    passage = np.frombuffer(data, dtype=np.float32)
                    similarity = cosine_similarity(query, passage)
                except ValueError:
                    similarity = None
                scores.append(similarity)
            valid = [score for score in scores if score is not None]
            if valid:
                ranked.append(SearchMatch(solution, max(valid), *scores))
        # Python's stable sort preserves the original first-leader tie behavior.
        return sorted(ranked, key=lambda match: -match.similarity)[:limit]
