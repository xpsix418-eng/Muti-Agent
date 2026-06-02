from __future__ import annotations

import numpy as np


class NearestIntruderPolicy:
    def __init__(self, speed: float):
        self.speed = speed

    def act(self, defenders: np.ndarray, intruders: np.ndarray) -> np.ndarray:
        deltas = intruders[None, :, :] - defenders[:, None, :]
        distances = np.linalg.norm(deltas, axis=-1)
        nearest = np.argmin(distances, axis=1)
        directions = intruders[nearest] - defenders
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        return directions / np.maximum(norms, 1e-6) * self.speed
