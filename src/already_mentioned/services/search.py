"""Find the single strongest confirmed answer within one chat."""

from dataclasses import dataclass

import numpy as np

from already_mentioned.models.entities import Solution
from already_mentioned.repositories.chats import ChatRepository
from already_mentioned.repositories.solutions import SolutionRepository
from already_mentioned.services.embeddings import EmbeddingService, normalize_vector
from already_mentioned.services.similarity import cosine_similarity


@dataclass(frozen=True, slots=True)
class SearchMatch:
    solution: Solution
    similarity: float


class SearchService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        chats: ChatRepository,
        solutions: SolutionRepository,
    ) -> None:
        self.embeddings = embeddings
        self.chats = chats
        self.solutions = solutions

    async def find_best(self, chat_id: int, question: str) -> SearchMatch | None:
        chat = await self.chats.get_chat(chat_id)
        if chat is None:
            return None
        candidates = await self.solutions.list_for_chat(chat_id)
        if not candidates:
            return None

        query = normalize_vector(await self.embeddings.embed_query(question))
        best: SearchMatch | None = None
        for solution in candidates:
            data = solution.question_embedding
            if not data:
                passage = normalize_vector(
                    await self.embeddings.embed_passage(solution.question_text)
                )
                data = passage.tobytes()
                await self.solutions.set_embedding(solution.id, data)
            try:
                passage = np.frombuffer(data, dtype=np.float32)
                similarity = cosine_similarity(query, passage)
            except ValueError:
                continue
            if best is None or similarity > best.similarity:
                best = SearchMatch(solution, similarity)
        if best is None or best.similarity < chat.similarity_threshold:
            return None
        return best
