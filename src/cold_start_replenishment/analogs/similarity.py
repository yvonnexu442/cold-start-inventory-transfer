from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def euclidean_similarity(
    target: Sequence[float], candidates: NDArray[np.float64]
) -> NDArray[np.float64]:
    distances = np.linalg.norm(candidates - np.asarray(target, dtype=float), axis=1)
    return np.asarray(1.0 / (1.0 + distances), dtype=np.float64)


def normalized_weights(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(scores, dtype=float), 0, None)
    total = clipped.sum()
    if total <= 0:
        return np.full(len(clipped), 1.0 / len(clipped), dtype=np.float64)
    return np.asarray(clipped / total, dtype=np.float64)
