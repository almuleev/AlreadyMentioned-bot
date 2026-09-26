import threading

import numpy as np
import pytest

from already_mentioned.services.embeddings import FastEmbedEmbeddingService


@pytest.mark.asyncio
async def test_lazy_service_uses_prefixes_and_worker_thread() -> None:
    seen: list[tuple[str, int]] = []

    class FakeModel:
        def embed(self, documents: list[str]):
            seen.append((documents[0], threading.get_ident()))
            yield np.array([3.0, 4.0], dtype=np.float32)

    service = FastEmbedEmbeddingService()
    assert service._model is None
    service._model = FakeModel()
    main_thread = threading.get_ident()
    query = await service.embed_query("Как войти?")
    passage = await service.embed_passage("Как зайти?")
    assert [item[0] for item in seen] == [
        "query: Как войти?",
        "passage: Как зайти?",
    ]
    assert all(item[1] != main_thread for item in seen)
    np.testing.assert_allclose(query, [0.6, 0.8])
    np.testing.assert_allclose(passage, [0.6, 0.8])
