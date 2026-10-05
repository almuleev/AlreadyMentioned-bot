"""Lazy, event-loop-safe embedding service for multilingual E5 small."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_DIMENSION = 384


class EmbeddingService(ABC):
    """Encode questions with E5 prefixes without blocking the event loop."""

    @abstractmethod
    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """Encode a new question using the ``query:`` prefix."""

    @abstractmethod
    async def embed_passage(self, text: str) -> NDArray[np.float32]:
        """Encode a confirmed question using the ``passage:`` prefix."""


def normalize_vector(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.asarray(vector, dtype=np.float32)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("Embedding must be a finite, one-dimensional vector")
    length = float(np.linalg.norm(result))
    if length == 0:
        raise ValueError("Embedding must not be a zero vector")
    return result / length


class FastEmbedEmbeddingService(EmbeddingService):
    """Load the local ONNX model only on first use and run it in a worker thread."""

    def __init__(
        self, cache_dir: str | Path = "models", *, threads: int | None = None
    ) -> None:
        if threads is not None and threads < 1:
            raise ValueError("Число потоков должно быть положительным")
        self.cache_dir = str(cache_dir)
        self.threads = threads
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        return await self._encode("query", text)

    async def embed_passage(self, text: str) -> NDArray[np.float32]:
        return await self._encode("passage", text)

    async def _encode(self, prefix: str, text: str) -> NDArray[np.float32]:
        async with self._lock:
            return await asyncio.to_thread(self._encode_sync, f"{prefix}: {text}")

    def _encode_sync(self, text: str) -> NDArray[np.float32]:
        if self._model is None:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType

            supported = TextEmbedding.list_supported_models()
            if not any(item["model"] == MODEL_NAME for item in supported):
                TextEmbedding.add_custom_model(
                    model=MODEL_NAME,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=MODEL_NAME),
                    dim=MODEL_DIMENSION,
                    model_file="onnx/model.onnx",
                )
            self._model = TextEmbedding(
                model_name=MODEL_NAME, cache_dir=self.cache_dir, threads=self.threads
            )
        vector = next(iter(self._model.embed([text])))
        return normalize_vector(vector)
