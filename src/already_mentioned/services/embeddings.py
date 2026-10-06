"""CPU embedding services; FRIDA is the production model, E5 is experimental."""

import argparse
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Legacy experiment constants also identify adopted pre-version-3 SQLite vectors.
# Do not retarget these names when changing the production encoder.
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_DIMENSION = 384
FRIDA_MODEL_NAME = "ai-forever/FRIDA"
FRIDA_REVISION = "850455b605544a944739b25f81ddf812b6e3d0d5"
FRIDA_DIMENSION = 1536
FRIDA_CONTRACT = {
    "model": FRIDA_MODEL_NAME,
    "revision": FRIDA_REVISION,
    "dimension": FRIDA_DIMENSION,
    "query_prefix": "search_query:",
    "saved_prefix": "search_document:",
    "pooling": "CLS",
    "max_length": 512,
    "normalization": "L2",
    "dtype": "float32-le",
    "precision": "fp32",
}
E5_CONTRACT = {
    "model": MODEL_NAME,
    "dimension": MODEL_DIMENSION,
    "query_prefix": "query:",
    "saved_prefix": "passage:",
    "pooling": "MEAN",
    "normalization": "L2",
    "dtype": "float32-le",
}


class EmbeddingService(ABC):
    """Encode query and saved text without blocking the event loop."""

    @abstractmethod
    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """Encode a new question under the service's query contract."""

    @abstractmethod
    async def embed_passage(self, text: str) -> NDArray[np.float32]:
        """Encode a saved question or answer under the document contract."""


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


class FridaEmbeddingService(EmbeddingService):
    """Pinned full FP32 FRIDA, serialized CPU inference and single-text batches."""

    def __init__(
        self,
        cache_dir: str | Path = "models",
        *,
        threads: int = 2,
        local_files_only: bool = True,
    ) -> None:
        if threads < 1:
            raise ValueError("Число потоков должно быть положительным")
        self.cache_dir = str(cache_dir)
        self.threads = threads
        self.local_files_only = local_files_only
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        return await self._encode("search_query", text)

    async def embed_passage(self, text: str) -> NDArray[np.float32]:
        return await self._encode("search_document", text)

    async def _encode(self, prompt: str, text: str) -> NDArray[np.float32]:
        async with self._lock:
            return await asyncio.to_thread(self._encode_sync, prompt, text)

    def _encode_sync(self, prompt: str, text: str) -> NDArray[np.float32]:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.threads)
            model = SentenceTransformer(
                FRIDA_MODEL_NAME,
                revision=FRIDA_REVISION,
                cache_folder=self.cache_dir,
                device="cpu",
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
            model.float()
            model.eval()
            if (
                model.get_embedding_dimension() != FRIDA_DIMENSION
                or model[1].pooling_mode != "cls"
                or not model[1].include_prompt
                or model.prompts.get("search_query") != "search_query: "
                or model.prompts.get("search_document") != "search_document: "
            ):
                raise RuntimeError("Загруженная FRIDA не соответствует контракту")
            model.max_seq_length = 512
            self._model = model
        vector = normalize_vector(
            self._model.encode(
                [text],
                prompt_name=prompt,
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
        )
        if vector.size != FRIDA_DIMENSION:
            raise ValueError("Неверная размерность FRIDA")
        return vector.astype("<f4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить локальную модель FRIDA")
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    service = FridaEmbeddingService(
        args.cache_dir,
        local_files_only=not args.download_model,
    )
    asyncio.run(service.embed_query("Проверка загрузки"))
    print(f"FRIDA готова: {FRIDA_DIMENSION} измерений, ревизия {FRIDA_REVISION}")


if __name__ == "__main__":
    main()
