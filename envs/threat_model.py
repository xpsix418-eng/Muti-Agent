from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ThreatAssessment:
    nearest_distances: Array
    zone_distances: Array
    captured: Array
    breached: Array


@dataclass(frozen=True)
class ThreatWeights:
    w_distance: float = 0.30
    w_velocity: float = 0.20
    w_heading: float = 0.20
    w_density: float = 0.15
    w_asset: float = 0.15


class ThreatModel:
    def __init__(
        self,
        world_size: float,
        protected_radius: float,
        intruder_max_speed: float,
        density_radius: float = 120.0,
        asset_value_threat: float = 1.0,
        weights: ThreatWeights | None = None,
    ):
        self.world_size = float(world_size)
        self.protected_radius = float(protected_radius)
        self.intruder_max_speed = float(intruder_max_speed)
        self.density_radius = float(density_radius)
        self.asset_value_threat = float(asset_value_threat)
        self.weights = weights or ThreatWeights()

    def score(self, intruder_positions: Array, intruder_velocities: Array, protected_asset_position: Array) -> Array:
        if len(intruder_positions) == 0:
            return np.asarray([], dtype=np.float32)

        distance_to_asset = np.linalg.norm(intruder_positions - protected_asset_position[None, :], axis=1)
        distance_threat = 1.0 - distance_to_asset / max(self.world_size, 1e-6)
        distance_threat = np.clip(distance_threat, 0.0, 1.0)

        speeds = np.linalg.norm(intruder_velocities, axis=1)
        velocity_threat = np.clip(speeds / max(self.intruder_max_speed, 1e-6), 0.0, 1.0)

        to_asset = _unit_vectors(protected_asset_position[None, :] - intruder_positions)
        velocity_dirs = _unit_vectors(intruder_velocities)
        heading_threat = np.clip(np.sum(to_asset * velocity_dirs, axis=1), 0.0, 1.0)

        pairwise = np.linalg.norm(intruder_positions[:, None, :] - intruder_positions[None, :, :], axis=-1)
        nearby_counts = np.sum((pairwise <= self.density_radius) & (pairwise > 0.0), axis=1)
        density_threat = np.clip(nearby_counts / max(len(intruder_positions) - 1, 1), 0.0, 1.0)

        asset_value = np.full(len(intruder_positions), np.clip(self.asset_value_threat, 0.0, 1.0), dtype=np.float32)
        weighted = (
            self.weights.w_distance * distance_threat
            + self.weights.w_velocity * velocity_threat
            + self.weights.w_heading * heading_threat
            + self.weights.w_density * density_threat
            + self.weights.w_asset * asset_value
        )
        normalizer = (
            self.weights.w_distance
            + self.weights.w_velocity
            + self.weights.w_heading
            + self.weights.w_density
            + self.weights.w_asset
        )
        return np.clip(weighted / max(normalizer, 1e-6), 0.0, 1.0).astype(np.float32)


def assess_threats(
    defender_positions: Array,
    intruder_positions: Array,
    protected_center: Array,
    capture_radius: float,
    protected_zone_radius: float,
) -> ThreatAssessment:
    deltas = defender_positions[:, None, :] - intruder_positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    nearest_distances = distances.min(axis=0) if distances.size else np.array([], dtype=np.float32)
    zone_distances = np.linalg.norm(intruder_positions - protected_center, axis=-1)
    return ThreatAssessment(
        nearest_distances=nearest_distances.astype(np.float32),
        zone_distances=zone_distances.astype(np.float32),
        captured=nearest_distances <= capture_radius,
        breached=zone_distances <= protected_zone_radius,
    )


def _unit_vectors(vectors: Array) -> Array:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-6)).astype(np.float32)
