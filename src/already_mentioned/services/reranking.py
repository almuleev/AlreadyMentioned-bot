"""Pinned local pair scoring for production MiniLM and offline comparisons."""

from pathlib import Path
from typing import Any, Protocol

import numpy as np

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
RERANKER_FILE = "onnx/model_quint8_avx2.onnx"


class PairScorer(Protocol):
    def predict(self, query: str, passages: list[str]) -> list[float]:
        """Return one score per passage, preserving input order."""


class OnnxReranker:
    """Pinned multilingual MiniLM cross-encoder with bounded CPU concurrency."""

    model = RERANKER_MODEL
    revision = RERANKER_REVISION
    file = RERANKER_FILE
    experiment_prefix = "frida"

    def __init__(
        self,
        cache_dir: Path = Path("models"),
        *,
        threads: int = 2,
        batch_size: int = 4,
        max_length: int = 512,
        local_only: bool = False,
    ) -> None:
        if threads < 1 or batch_size < 1 or not 8 <= max_length <= 512:
            raise ValueError("Некорректные параметры CPU, пакета или длины пары")
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_only = local_only
        self._session: Any = None
        self._tokenizer: Any = None
        self.total_pairs = 0
        self.truncated_pairs = 0

    def _load(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        def download(filename: str) -> str:
            return hf_hub_download(
                self.model,
                filename,
                revision=self.revision,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_only,
                token=False,
            )

        tokenizer = Tokenizer.from_file(download("tokenizer.json"))
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("В токенизаторе не найден токен дополнения")
        tokenizer.enable_truncation(
            max_length=self.max_length, strategy="longest_first"
        )
        tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            download(self.file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = tokenizer

    def predict(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._session is None:
            self._load()
        result = []
        for offset in range(0, len(passages), self.batch_size):
            batch = passages[offset : offset + self.batch_size]
            encoded = self._tokenizer.encode_batch([(query, text) for text in batch])
            arrays = {
                "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.asarray(
                    [e.attention_mask for e in encoded], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [e.type_ids for e in encoded], dtype=np.int64
                ),
            }
            names = [item.name for item in self._session.get_inputs()]
            if not set(names) <= arrays.keys():
                raise ValueError("Неподдерживаемый контракт входов ONNX")
            logits = np.asarray(
                self._session.run(None, {name: arrays[name] for name in names})[0]
            )
            if logits.shape not in {(len(batch),), (len(batch), 1)} or not np.all(
                np.isfinite(logits)
            ):
                raise ValueError("Оценщик вернул некорректные оценки")
            # Monotonic scale for threshold selection, not calibrated probability.
            scores = 1 / (
                1 + np.exp(-np.clip(logits.reshape(-1).astype(np.float64), -60, 60))
            )
            result.extend(float(score) for score in scores)
            self.total_pairs += len(batch)
            self.truncated_pairs += sum(
                bool(getattr(e, "overflowing", [])) for e in encoded
            )
        return result


class GteOnnxReranker(OnnxReranker):
    """Offline FP32 community ONNX export; no remote Python code is loaded."""

    model = "onnx-community/gte-multilingual-reranker-base"
    revision = "ee64367e35a2db0da46bb6497e13a18f8bd585cb"
    file = "onnx/model.onnx"
    experiment_prefix = "frida-gte-fp32"
