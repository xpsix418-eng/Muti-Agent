from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


Array = np.ndarray


@dataclass(frozen=True)
class HungarianAssignmentPolicy:
    distance_weight: float = 1.0
    threat_weight: float = 0.8
    heading_weight: float = 0.4

    def act(
        self,
        defender_positions: Array,
        defender_velocities: Array,
        intruder_positions: Array,
        intruder_velocities: Array,
        intruder_active: Array,
        threat_scores: Array,
        protected_asset_position: Array,
        world_size: float,
    ) -> Array:
        active_indices = np.flatnonzero(intruder_active)
        if len(active_indices) == 0:
            return np.zeros((len(defender_positions), 2), dtype=np.float32)

        active_positions = intruder_positions[active_indices]
        active_velocities = intruder_velocities[active_indices]
        active_threats = threat_scores[active_indices]
        cost = self._cost_matrix(
            defender_positions,
            active_positions,
            active_velocities,
            active_threats,
            protected_asset_position,
            world_size,
        )
        rows, cols = linear_sum_assignment(cost)
        actions = np.zeros((len(defender_positions), 2), dtype=np.float32)
        for row, col in zip(rows.tolist(), cols.tolist()):
            target_position = active_positions[col]
            direction = target_position - defender_positions[row]
            actions[row] = self._direction_to_action(direction, defender_velocities[row])
        return actions

    def _cost_matrix(
        self,
        defender_positions: Array,
        intruder_positions: Array,
        intruder_velocities: Array,
        threat_scores: Array,
        protected_asset_position: Array,
        world_size: float,
    ) -> Array:
        distance_cost = np.linalg.norm(defender_positions[:, None, :] - intruder_positions[None, :, :], axis=-1)
        distance_cost = distance_cost / max(world_size, 1e-6)
        threat_priority_cost = 1.0 - threat_scores[None, :]
        heading_cost = 1.0 - self._heading_alignment(intruder_positions, intruder_velocities, protected_asset_position)
        return (
            self.distance_weight * distance_cost
            + self.threat_weight * threat_priority_cost
            + self.heading_weight * heading_cost[None, :]
        ).astype(np.float32)

    def _heading_alignment(self, intruder_positions: Array, intruder_velocities: Array, protected_asset_position: Array) -> Array:
        to_asset = protected_asset_position[None, :] - intruder_positions
        to_asset = to_asset / np.maximum(np.linalg.norm(to_asset, axis=1, keepdims=True), 1e-6)
        velocity_dirs = intruder_velocities / np.maximum(np.linalg.norm(intruder_velocities, axis=1, keepdims=True), 1e-6)
        return np.clip(np.sum(to_asset * velocity_dirs, axis=1), 0.0, 1.0)

    def _direction_to_action(self, direction: Array, velocity: Array) -> Array:
        desired = direction / max(float(np.linalg.norm(direction)), 1e-6)
        damping = 0.05 * velocity
        return np.clip(desired - damping, -1.0, 1.0).astype(np.float32)


def assign_defenders_to_intruders(defenders: Array, intruders: Array) -> list[tuple[int, int]]:
    cost = np.linalg.norm(defenders[:, None, :] - intruders[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))
