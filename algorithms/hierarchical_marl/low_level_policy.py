from __future__ import annotations

import numpy as np


class LowLevelPolicy:
    def act(self, option: str, defenders: np.ndarray, intruders: np.ndarray) -> np.ndarray:
        del option
        directions = intruders.mean(axis=0, keepdims=True) - defenders
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        return directions / np.maximum(norms, 1e-6)
