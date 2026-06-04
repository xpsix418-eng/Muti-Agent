from __future__ import annotations

import numpy as np


def graph_attention_sparsity(attention: np.ndarray, threshold: float = 0.05) -> float:
    if attention.size == 0:
        return 0.0
    return float(np.mean(attention < threshold))


def assignment_entropy(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    clipped = np.clip(weights, 1e-8, 1.0)
    return float(np.mean(-np.sum(clipped * np.log(clipped), axis=-1)))


def mean_interception_time_advantage(ita: np.ndarray) -> float:
    if ita.size == 0:
        return 0.0
    return float(np.mean(ita))
