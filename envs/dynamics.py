from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
IntruderBehavior = Literal["straight_attack", "random_maneuver", "evasive_intruder"]


@dataclass(frozen=True)
class PointMassConfig:
    dt: float
    max_speed: float
    world_size: tuple[float, ...] | float


@dataclass(frozen=True)
class DynamicsConfig:
    dt: float
    world_size: float
    defender_max_speed: float
    intruder_max_speed: float
    defender_acceleration_scale: float = 1.0
    random_maneuver_scale: float = 0.35
    evasive_distance: float = 100.0
    evasive_strength: float = 0.8


def clip_positions(positions: Array, world_size: tuple[float, ...] | float) -> Array:
    bounds = _world_bounds(world_size, positions.shape[-1])
    return np.clip(positions, 0.0, bounds).astype(np.float32)


def limit_speed(velocities: Array, max_speed: float) -> Array:
    norms = np.linalg.norm(velocities, axis=-1, keepdims=True)
    scales = np.minimum(1.0, max_speed / np.maximum(norms, 1e-6))
    return (velocities * scales).astype(np.float32)


def update_defender_dynamics(
    positions: Array,
    velocities: Array,
    actions: Array,
    config: DynamicsConfig,
) -> tuple[Array, Array]:
    acceleration = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
    acceleration = acceleration * config.defender_acceleration_scale
    next_velocities = limit_speed(velocities + acceleration * config.dt, config.defender_max_speed)
    next_positions = clip_positions(positions + next_velocities * config.dt, config.world_size)
    return next_positions, next_velocities


def update_intruder_dynamics(
    positions: Array,
    velocities: Array,
    protected_asset_position: Array,
    defender_positions: Array,
    config: DynamicsConfig,
    behavior: IntruderBehavior = "straight_attack",
    rng: np.random.Generator | None = None,
    active_mask: Array | None = None,
) -> tuple[Array, Array]:
    rng = rng or np.random.default_rng()
    active = np.ones(len(positions), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    desired = _unit_vectors(protected_asset_position[None, :] - positions)

    if behavior == "random_maneuver":
        desired = desired + rng.normal(0.0, config.random_maneuver_scale, size=desired.shape).astype(np.float32)
        desired = _unit_vectors(desired)
    elif behavior == "evasive_intruder":
        desired = _evasive_directions(positions, defender_positions, desired, config)
    elif behavior != "straight_attack":
        raise ValueError(f"Unsupported intruder behavior: {behavior}")

    next_velocities = desired * config.intruder_max_speed
    next_velocities[~active] = 0.0
    next_velocities = limit_speed(next_velocities, config.intruder_max_speed)
    next_positions = clip_positions(positions + next_velocities * config.dt * active[:, None], config.world_size)
    return next_positions, next_velocities.astype(np.float32)


def integrate_velocity(positions: Array, velocity_actions: Array, config: PointMassConfig) -> Array:
    clipped_velocity = limit_speed(np.asarray(velocity_actions, dtype=np.float32), config.max_speed)
    next_positions = positions + clipped_velocity * config.dt
    return clip_positions(next_positions, config.world_size)


def _evasive_directions(positions: Array, defender_positions: Array, desired: Array, config: DynamicsConfig) -> Array:
    if len(defender_positions) == 0:
        return desired
    deltas = positions[:, None, :] - defender_positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    nearest = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(positions)), nearest]
    away = _unit_vectors(deltas[np.arange(len(positions)), nearest])
    weights = np.clip(1.0 - nearest_distances / max(config.evasive_distance, 1e-6), 0.0, 1.0)[:, None]
    blended = desired + away * weights * config.evasive_strength
    return _unit_vectors(blended)


def _unit_vectors(vectors: Array) -> Array:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-6)).astype(np.float32)


def _world_bounds(world_size: tuple[float, ...] | float, dim: int) -> Array:
    if isinstance(world_size, tuple):
        return np.asarray(world_size, dtype=np.float32)
    return np.full(dim, float(world_size), dtype=np.float32)
