"""Pure vector similarity helpers."""

import numpy as np
from numpy.typing import ArrayLike


def cosine_similarity(first: ArrayLike, second: ArrayLike) -> float:
    """Return cosine similarity for two nonzero, one-dimensional vectors.

    Raise ValueError for empty, differently sized, non-finite or zero vectors.
    """
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.size == 0 or left.shape != right.shape:
        raise ValueError(
            "Vectors must be non-empty, one-dimensional, and equal in size"
        )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Vectors must contain only finite numbers")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
