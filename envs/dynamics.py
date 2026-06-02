from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointMassConfig:
    dt: float
    max_speed: float
    world_size: tuple[float, ...]


def clip_positions(positions: np.ndarray, world_size: tuple[float, ...]) -> np.ndarray:
    bounds = np.asarray(world_size, dtype=np.float32)
    return np.clip(positions, 0.0, bounds)


def integrate_velocity(
    positions: np.ndarray,
    velocity_actions: np.ndarray,
    config: PointMassConfig,
) -> np.ndarray:
    actions = np.asarray(velocity_actions, dtype=np.float32)
    norms = np.linalg.norm(actions, axis=-1, keepdims=True)
    scale = np.minimum(1.0, config.max_speed / np.maximum(norms, 1e-6))
    clipped_velocity = actions * scale
    next_positions = positions + clipped_velocity * config.dt
    return clip_positions(next_positions, config.world_size)
