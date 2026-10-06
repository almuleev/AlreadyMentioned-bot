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


@pytest.mark.asyncio
async def test_frida_pinned_load_and_query_document_contract(monkeypatch):
    import sys
    from types import SimpleNamespace

    from already_mentioned.services.embeddings import (
        FRIDA_MODEL_NAME,
        FRIDA_REVISION,
        FridaEmbeddingService,
    )

    loads, calls = [], []
    main_thread = threading.get_ident()

    class FakeFrida:
        prompts = {
            "search_query": "search_query: ",
            "search_document": "search_document: ",
        }

        def __init__(self, *args, **kwargs):
            loads.append((args, kwargs))

        def float(self):
            pass

        def eval(self):
            pass

        def __getitem__(self, index):
            assert index == 1
            return SimpleNamespace(pooling_mode="cls", include_prompt=True)

        def get_embedding_dimension(self):
            return 1536

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs, threading.get_ident()))
            result = np.zeros((1, 1536), dtype=np.float32)
            result[0, 0] = 3
            result[0, 1] = 4
            return result

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeFrida),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None)
    )
    service = FridaEmbeddingService()
    assert service._model is None
    q = await service.embed_query("Как войти?")
    a = await service.embed_passage("Откройте сайт.")
    assert len(loads) == 1
    assert loads[0] == (
        (FRIDA_MODEL_NAME,),
        {
            "revision": FRIDA_REVISION,
            "cache_folder": "models",
            "device": "cpu",
            "local_files_only": True,
            "trust_remote_code": False,
        },
    )
    assert [c[1]["prompt_name"] for c in calls] == ["search_query", "search_document"]
    assert all(c[1]["batch_size"] == 1 and c[2] != main_thread for c in calls)
    assert service._model.max_seq_length == 512
    np.testing.assert_allclose(q[:2], [0.6, 0.8])
    np.testing.assert_allclose(q, a)
