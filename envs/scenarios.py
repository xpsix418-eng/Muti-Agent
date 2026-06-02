from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioConfig:
    dim: int
    world_size: tuple[float, ...]
    num_defenders: int
    num_intruders: int
    defender_spawn_radius: float
    intruder_spawn_min_radius: float
    intruder_spawn_max_radius: float


def random_2d(config: ScenarioConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(config.world_size, dtype=np.float32) / 2.0
    defender_angles = rng.uniform(0.0, 2.0 * np.pi, size=config.num_defenders)
    defender_radii = rng.uniform(0.0, config.defender_spawn_radius, size=config.num_defenders)
    defenders = center + np.stack(
        [np.cos(defender_angles) * defender_radii, np.sin(defender_angles) * defender_radii],
        axis=-1,
    )

    intruder_angles = rng.uniform(0.0, 2.0 * np.pi, size=config.num_intruders)
    intruder_radii = rng.uniform(
        config.intruder_spawn_min_radius,
        config.intruder_spawn_max_radius,
        size=config.num_intruders,
    )
    intruders = center + np.stack(
        [np.cos(intruder_angles) * intruder_radii, np.sin(intruder_angles) * intruder_radii],
        axis=-1,
    )
    return defenders.astype(np.float32), intruders.astype(np.float32), center.astype(np.float32)
