from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InterceptionGraph:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    defender_indices: np.ndarray
    intruder_indices: np.ndarray
    asset_index: int
    interception_point_indices: np.ndarray
    pair_edge_features: np.ndarray
    heuristic_assignments: np.ndarray
    predicted_intercept_points: np.ndarray
    interception_time_advantage: np.ndarray


class InterceptionGraphBuilder:
    """Builds an interception prediction graph from simulated environment state."""

    node_feature_dim = 11
    edge_feature_dim = 7

    def __init__(
        self,
        world_size: float,
        defender_max_speed: float,
        intruder_max_speed: float,
        prediction_horizon: float = 5.0,
    ):
        self.world_size = float(world_size)
        self.defender_max_speed = float(defender_max_speed)
        self.intruder_max_speed = float(intruder_max_speed)
        self.prediction_horizon = float(prediction_horizon)

    def build(self, info: dict[str, Any]) -> InterceptionGraph:
        defenders = np.asarray(info["defender_positions"], dtype=np.float32)
        defender_velocities = np.asarray(info["defender_velocities"], dtype=np.float32)
        intruders = np.asarray(info["intruder_positions"], dtype=np.float32)
        intruder_velocities = np.asarray(info["intruder_velocities"], dtype=np.float32)
        asset = np.asarray(info.get("protected_asset_position", [500.0, 500.0]), dtype=np.float32)
        if "protected_asset_position" not in info:
            asset = self._infer_asset_from_state(info)
        threat_scores = np.asarray(info["threat_scores"], dtype=np.float32)
        intercepted = np.asarray(info.get("intercepted", np.zeros(len(intruders))), dtype=bool)
        breached = np.asarray(info.get("breached", np.zeros(len(intruders))), dtype=bool)
        active = ~(intercepted | breached)
        topology = np.asarray(info.get("communication_topology", np.eye(len(defenders))), dtype=np.float32)

        predicted_points = np.clip(
            intruders + intruder_velocities * self.prediction_horizon,
            0.0,
            self.world_size,
        ).astype(np.float32)
        node_features = self._node_features(
            defenders,
            defender_velocities,
            intruders,
            intruder_velocities,
            asset,
            predicted_points,
            threat_scores,
            active,
            topology,
        )
        num_defenders = len(defenders)
        num_intruders = len(intruders)
        defender_indices = np.arange(num_defenders, dtype=np.int64)
        intruder_indices = np.arange(num_defenders, num_defenders + num_intruders, dtype=np.int64)
        asset_index = num_defenders + num_intruders
        point_indices = np.arange(asset_index + 1, asset_index + 1 + num_intruders, dtype=np.int64)

        edges: list[tuple[int, int]] = []
        features: list[np.ndarray] = []
        pair_edge_features = np.zeros((num_defenders, num_intruders, self.edge_feature_dim), dtype=np.float32)
        ita = np.zeros((num_defenders, num_intruders), dtype=np.float32)

        for row in range(num_defenders):
            for col in range(num_defenders):
                if row == col:
                    continue
                edge = self._edge_feature(
                    defenders[row],
                    defenders[col],
                    np.zeros(2, dtype=np.float32),
                    asset,
                    0.0,
                    topology[row, col],
                    defenders[row],
                )
                edges.append((row, col))
                features.append(edge)

        for defender_idx in range(num_defenders):
            for intruder_idx in range(num_intruders):
                comm = float(np.any(topology[defender_idx] > 0.0))
                df_pos = defenders[defender_idx]
                it_pos = intruders[intruder_idx]
                ip_pos = predicted_points[intruder_idx]
                edge = self._edge_feature(
                    df_pos,
                    it_pos,
                    intruder_velocities[intruder_idx],
                    asset,
                    threat_scores[intruder_idx],
                    comm,
                    df_pos,
                    intercept_point=ip_pos,
                )
                edges.append((defender_indices[defender_idx], intruder_indices[intruder_idx]))
                features.append(edge)
                pair_edge_features[defender_idx, intruder_idx] = edge
                ita[defender_idx, intruder_idx] = edge[5]

                ip_edge = self._edge_feature(
                    df_pos,
                    ip_pos,
                    np.zeros(2, dtype=np.float32),
                    asset,
                    threat_scores[intruder_idx],
                    comm,
                    df_pos,
                    intruder_pos=it_pos,
                    intruder_vel=intruder_velocities[intruder_idx],
                )
                edges.append((defender_indices[defender_idx], point_indices[intruder_idx]))
                features.append(ip_edge)

        for intruder_idx in range(num_intruders):
            edges.append((intruder_indices[intruder_idx], asset_index))
            features.append(
                self._edge_feature(
                    intruders[intruder_idx],
                    asset,
                    intruder_velocities[intruder_idx],
                    asset,
                    threat_scores[intruder_idx],
                    1.0,
                    intruders[intruder_idx],
                )
            )
            edges.append((intruder_indices[intruder_idx], point_indices[intruder_idx]))
            features.append(
                self._edge_feature(
                    intruders[intruder_idx],
                    predicted_points[intruder_idx],
                    intruder_velocities[intruder_idx],
                    asset,
                    threat_scores[intruder_idx],
                    1.0,
                    intruders[intruder_idx],
                )
            )

        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_features = np.stack(features).astype(np.float32)
        assignments = self._heuristic_assignments(pair_edge_features, active)
        return InterceptionGraph(
            node_features=node_features.astype(np.float32),
            edge_index=edge_index,
            edge_features=edge_features,
            defender_indices=defender_indices,
            intruder_indices=intruder_indices,
            asset_index=asset_index,
            interception_point_indices=point_indices,
            pair_edge_features=pair_edge_features,
            heuristic_assignments=assignments,
            predicted_intercept_points=predicted_points,
            interception_time_advantage=ita,
        )

    def _node_features(
        self,
        defenders: np.ndarray,
        defender_velocities: np.ndarray,
        intruders: np.ndarray,
        intruder_velocities: np.ndarray,
        asset: np.ndarray,
        predicted_points: np.ndarray,
        threat_scores: np.ndarray,
        active: np.ndarray,
        topology: np.ndarray,
    ) -> np.ndarray:
        rows = []
        for idx, pos in enumerate(defenders):
            rows.append(
                self._node_row(
                    node_type=0,
                    position=pos,
                    velocity=defender_velocities[idx],
                    threat=0.0,
                    active=1.0,
                    comm=float(np.any(topology[idx] > 0.0)),
                )
            )
        for idx, pos in enumerate(intruders):
            rows.append(
                self._node_row(
                    node_type=1,
                    position=pos,
                    velocity=intruder_velocities[idx],
                    threat=float(threat_scores[idx]),
                    active=float(active[idx]),
                    comm=1.0,
                )
            )
        rows.append(self._node_row(2, asset, np.zeros(2, dtype=np.float32), 1.0, 1.0, 1.0))
        for idx, pos in enumerate(predicted_points):
            rows.append(
                self._node_row(
                    node_type=3,
                    position=pos,
                    velocity=np.zeros(2, dtype=np.float32),
                    threat=float(threat_scores[idx]),
                    active=float(active[idx]),
                    comm=1.0,
                )
            )
        return np.stack(rows).astype(np.float32)

    def _node_row(
        self,
        node_type: int,
        position: np.ndarray,
        velocity: np.ndarray,
        threat: float,
        active: float,
        comm: float,
    ) -> np.ndarray:
        type_one_hot = np.zeros(4, dtype=np.float32)
        type_one_hot[node_type] = 1.0
        velocity_scale = self.defender_max_speed if node_type == 0 else self.intruder_max_speed
        return np.concatenate(
            [
                type_one_hot,
                position.astype(np.float32) / max(self.world_size, 1e-6),
                velocity.astype(np.float32) / max(velocity_scale, 1e-6),
                np.asarray([threat, active, comm], dtype=np.float32),
            ]
        )

    def _edge_feature(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        dst_velocity: np.ndarray,
        asset: np.ndarray,
        threat_score: float,
        communication_available: float,
        defender_pos: np.ndarray,
        intercept_point: np.ndarray | None = None,
        intruder_pos: np.ndarray | None = None,
        intruder_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        delta = dst - src
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-6)
        target_pos = dst if intruder_pos is None else intruder_pos
        target_vel = dst_velocity if intruder_vel is None else intruder_vel
        to_asset = asset - target_pos
        time_to_asset = float(np.linalg.norm(to_asset) / max(np.linalg.norm(target_vel), 1e-6))
        ip = dst if intercept_point is None else intercept_point
        time_to_intercept = float(np.linalg.norm(ip - defender_pos) / max(self.defender_max_speed, 1e-6))
        ita = time_to_asset - time_to_intercept
        return np.asarray(
            [
                distance / max(self.world_size, 1e-6),
                direction[0],
                direction[1],
                threat_score,
                time_to_asset / 100.0,
                ita / 100.0,
                communication_available,
            ],
            dtype=np.float32,
        )

    def _heuristic_assignments(self, pair_edge_features: np.ndarray, active: np.ndarray) -> np.ndarray:
        num_defenders, num_intruders, _ = pair_edge_features.shape
        assignments = np.full(num_defenders, -1, dtype=np.int64)
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            return assignments
        distance_cost = pair_edge_features[:, active_indices, 0]
        threat = pair_edge_features[:, active_indices, 3]
        ita = pair_edge_features[:, active_indices, 5]
        cost = 0.45 * distance_cost - 0.20 * threat - 0.10 * ita
        use_counts = np.zeros(len(active_indices), dtype=np.int64)
        for defender_idx in range(num_defenders):
            ordered = np.argsort(cost[defender_idx])
            chosen = ordered[0]
            for candidate in ordered:
                if use_counts[candidate] == 0:
                    chosen = candidate
                    break
            assignments[defender_idx] = int(active_indices[chosen])
            use_counts[chosen] += 1
        return assignments

    def _infer_asset_from_state(self, info: dict[str, Any]) -> np.ndarray:
        state = np.asarray(info.get("global_state", []), dtype=np.float32)
        defenders = np.asarray(info.get("defender_positions", []), dtype=np.float32)
        topology_width = defenders.shape[0] * defenders.shape[0] if defenders.ndim == 2 else 0
        asset_start = state.size - topology_width - 1 - 2
        asset_end = asset_start + 2
        if 0 <= asset_start and asset_end <= state.size:
            return state[asset_start:asset_end].astype(np.float32)
        return np.asarray([self.world_size * 0.5, self.world_size * 0.5], dtype=np.float32)
