from __future__ import annotations

import numpy as np


def radius_graph(positions: np.ndarray, radius: float, self_loops: bool = True) -> np.ndarray:
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adjacency = (distances <= radius).astype(np.float32)
    if not self_loops:
        np.fill_diagonal(adjacency, 0.0)
    return adjacency
