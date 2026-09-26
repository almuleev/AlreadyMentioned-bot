import numpy as np
import pytest

from already_mentioned.services.similarity import cosine_similarity


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 0], [1, 0], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 0], [-1, 0], -1.0),
        ([1, 1], [2, 2], 1.0),
    ],
)
def test_cosine_similarity(left: list[int], right: list[int], expected: float) -> None:
    assert cosine_similarity(np.array(left), np.array(right)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1, 2], [1]),
        ([], []),
        ([[1, 2]], [[1, 2]]),
        ([0, 0], [1, 0]),
        ([1, float("nan")], [1, 2]),
    ],
)
def test_invalid_vectors_raise_value_error(left: list, right: list) -> None:
    with pytest.raises(ValueError):
        cosine_similarity(left, right)
