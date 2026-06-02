from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RuleBasedPolicy:
    lambda_threat: float = 1.0
    lambda_distance: float = 0.6
    visibility_radius: float = 450.0
    patrol_gain: float = 0.25

    def act(
        self,
        defender_positions: Array,
        defender_velocities: Array,
        intruder_positions: Array,
        intruder_active: Array,
        threat_scores: Array,
        protected_asset_position: Array,
        world_size: float,
    ) -> Array:
        actions = np.zeros((len(defender_positions), 2), dtype=np.float32)
        for idx, defender_position in enumerate(defender_positions):
            visible = self._visible_intruders(defender_position, intruder_positions, intruder_active)
            if np.any(visible):
                target_idx = self._select_target(defender_position, intruder_positions, visible, threat_scores, world_size)
                direction = intruder_positions[target_idx] - defender_position
            else:
                direction = self._patrol_direction(defender_position, protected_asset_position, idx, len(defender_positions))
            actions[idx] = self._direction_to_action(direction, defender_velocities[idx])
        return actions

    def _visible_intruders(self, defender_position: Array, intruder_positions: Array, intruder_active: Array) -> Array:
        distances = np.linalg.norm(intruder_positions - defender_position[None, :], axis=1)
        return (distances <= self.visibility_radius) & intruder_active

    def _select_target(
        self,
        defender_position: Array,
        intruder_positions: Array,
        visible: Array,
        threat_scores: Array,
        world_size: float,
    ) -> int:
        distances = np.linalg.norm(intruder_positions - defender_position[None, :], axis=1)
        normalized_distance = distances / max(world_size, 1e-6)
        scores = self.lambda_threat * threat_scores - self.lambda_distance * normalized_distance
        scores = np.where(visible, scores, -np.inf)
        return int(np.argmax(scores))

    def _patrol_direction(
        self,
        defender_position: Array,
        protected_asset_position: Array,
        defender_idx: int,
        num_defenders: int,
    ) -> Array:
        angle = 2.0 * np.pi * defender_idx / max(num_defenders, 1)
        patrol_offset = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32) * 80.0
        patrol_point = protected_asset_position + patrol_offset
        return (patrol_point - defender_position) * self.patrol_gain

    def _direction_to_action(self, direction: Array, velocity: Array) -> Array:
        desired = direction / max(float(np.linalg.norm(direction)), 1e-6)
        damping = 0.05 * velocity
        return np.clip(desired - damping, -1.0, 1.0).astype(np.float32)


class NearestIntruderPolicy:
    def __init__(self, speed: float):
        self.speed = speed

    def act(self, defenders: Array, intruders: Array) -> Array:
        del self
        deltas = intruders[None, :, :] - defenders[:, None, :]
        distances = np.linalg.norm(deltas, axis=-1)
        nearest = np.argmin(distances, axis=1)
        directions = intruders[nearest] - defenders
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        return np.clip(directions / np.maximum(norms, 1e-6), -1.0, 1.0).astype(np.float32)
