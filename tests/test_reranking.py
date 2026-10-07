"""Pair inference preserves order/masks and uses no real model in tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from already_mentioned.services.reranking import GteOnnxReranker, OnnxReranker


def test_onnx_pair_batching_masks_and_monotonic_scores():
    pairs, feeds = [], []

    class Tokenizer:
        def encode_batch(self, batch):
            pairs.extend(batch)
            return [
                SimpleNamespace(
                    ids=[1, 2, 0],
                    attention_mask=[1, 1, 0],
                    type_ids=[0, 1, 0],
                    overflowing=[],
                )
                for _ in batch
            ]

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name=n) for n in ("input_ids", "attention_mask")]

        def run(self, _, feed):
            feeds.append(feed)
            return [np.asarray([[0.0]] * len(feed["input_ids"]))]

    scorer = OnnxReranker(batch_size=2)
    scorer._tokenizer, scorer._session = Tokenizer(), Session()
    assert (
        scorer.predict("Новый вопрос", ["Первый ответ", "Второй ответ", "Третий ответ"])
        == [0.5] * 3
    )
    assert pairs == [
        ("Новый вопрос", text)
        for text in ("Первый ответ", "Второй ответ", "Третий ответ")
    ]
    assert [f["input_ids"].shape[0] for f in feeds] == [2, 1]
    assert all(set(feed) == {"input_ids", "attention_mask"} for feed in feeds)
    assert feeds[0]["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 0]]
    assert feeds[0]["input_ids"].dtype == np.int64
    assert scorer.total_pairs == 3 and scorer.truncated_pairs == 0


@pytest.mark.parametrize("logits", [np.array([[np.nan]]), np.array([[1, 2]])])
def test_bad_onnx_outputs_are_rejected(logits):
    scorer = OnnxReranker()
    scorer._tokenizer = SimpleNamespace(
        encode_batch=lambda _: [
            SimpleNamespace(ids=[1], attention_mask=[1], type_ids=[0])
        ]
    )
    scorer._session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input_ids")], run=lambda *_: [logits]
    )
    with pytest.raises(ValueError, match="некорректные оценки"):
        scorer.predict("Вопрос", ["Ответ"])


def test_empty_input_does_not_load_model():
    scorer = OnnxReranker()
    assert scorer.predict("Вопрос", []) == [] and scorer._session is None


@pytest.mark.parametrize(
    "settings", [{"threads": 0}, {"batch_size": 0}, {"max_length": 513}]
)
def test_limits_rejected_before_model_load(settings):
    with pytest.raises(ValueError):
        OnnxReranker(**settings)


@pytest.mark.parametrize("scorer_class", [OnnxReranker, GteOnnxReranker])
def test_model_loading_uses_own_pinned_identity_and_cache_policy(
    monkeypatch, scorer_class
):
    import huggingface_hub
    import onnxruntime
    import tokenizers

    downloads = []
    tokenizer = SimpleNamespace(
        token_to_id=lambda _: 1, enable_truncation=lambda **_: None,
        enable_padding=lambda **_: None,
    )

    def download(model, filename, **kwargs):
        downloads.append((model, filename, kwargs))
        return filename

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", lambda _: tokenizer)
    sessions = []
    monkeypatch.setattr(onnxruntime, "InferenceSession",
                        lambda path, **kwargs: sessions.append((path, kwargs)))
    scorer = scorer_class(local_only=True, batch_size=1)
    scorer._load()
    assert [name for _, name, _ in downloads] == ["tokenizer.json", scorer.file]
    assert all(model == scorer.model and kwargs["revision"] == scorer.revision
               and kwargs["local_files_only"] and kwargs["token"] is False
               for model, _, kwargs in downloads)
    assert sessions[0][0] == scorer.file
    assert sessions[0][1]["providers"] == ["CPUExecutionProvider"]
