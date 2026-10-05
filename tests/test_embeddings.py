import threading

import numpy as np
import pytest

from already_mentioned.services.embeddings import MODEL_NAME, FastEmbedEmbeddingService


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


@pytest.mark.asyncio
@pytest.mark.parametrize("threads", [None, 2])
async def test_cpu_threads_are_forwarded_without_changing_vector_contract(
    monkeypatch, threads
):
    import fastembed

    received = []

    class FakeModel:
        @staticmethod
        def list_supported_models():
            return [{"model": MODEL_NAME}]

        def __init__(self, **kwargs):
            received.append(kwargs)

        def embed(self, documents):
            assert documents == ["query: Как войти?"]
            yield np.array([3.0, 4.0], dtype=np.float32)

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeModel)
    vector = await FastEmbedEmbeddingService(threads=threads).embed_query("Как войти?")
    assert received == [
        {"model_name": MODEL_NAME, "cache_dir": "models", "threads": threads}
    ]
    np.testing.assert_allclose(vector, [0.6, 0.8])


def test_invalid_thread_limit_is_rejected():
    with pytest.raises(ValueError):
        FastEmbedEmbeddingService(threads=0)
