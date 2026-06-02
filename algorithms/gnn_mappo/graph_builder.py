from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class GraphData:
    node_features: Array
    edge_index: Array
    edge_features: Array
    adjacency: Array
    defender_node_indices: Array
    intruder_node_indices: Array
    asset_node_index: int


def build_dynamic_graph(info: dict[str, Any], protected_asset_position: Array, world_size: float) -> GraphData:
    defender_positions = info["defender_positions"]
    defender_velocities = info["defender_velocities"]
    intruder_positions = info["intruder_positions"]
    intruder_velocities = info["intruder_velocities"]
    threat_scores = info["threat_scores"]
    intercepted = info["intercepted"].astype(np.float32)
    comm_adj = info.get("comm_adj", info.get("communication_topology"))

    node_features = []
    for idx in range(len(defender_positions)):
        comm_reachable = float(np.any(comm_adj[idx]))
        node_features.append(_node_feature(0, defender_positions[idx], defender_velocities[idx], 0.0, 0.0, comm_reachable, world_size))
    for idx in range(len(intruder_positions)):
        node_features.append(_node_feature(1, intruder_positions[idx], intruder_velocities[idx], threat_scores[idx], intercepted[idx], 0.0, world_size))
    node_features.append(_node_feature(2, protected_asset_position, np.zeros(2, dtype=np.float32), 1.0, 0.0, 1.0, world_size))

    defender_indices = np.arange(len(defender_positions), dtype=np.int64)
    intruder_indices = np.arange(len(intruder_positions), dtype=np.int64) + len(defender_positions)
    asset_idx = len(defender_positions) + len(intruder_positions)
    edges: list[tuple[int, int]] = []
    edge_features: list[Array] = []

    for i in range(len(defender_positions)):
        for j in range(len(defender_positions)):
            if i != j and comm_adj[i, j] > 0.0:
                _append_edge(edges, edge_features, i, j, defender_positions, defender_positions, world_size, 1.0, 0.0)

    for d_idx, d_pos in enumerate(defender_positions):
        for local_i, i_pos in enumerate(intruder_positions):
            strength = float(threat_scores[local_i])
            _append_edge(
                edges,
                edge_features,
                d_idx,
                int(intruder_indices[local_i]),
                np.asarray([d_pos]),
                np.asarray([i_pos]),
                world_size,
                0.0,
                strength,
                source_local_index=0,
                target_local_index=0,
            )

    for local_i, i_pos in enumerate(intruder_positions):
        strength = float(threat_scores[local_i])
        _append_single_edge(edges, edge_features, int(intruder_indices[local_i]), asset_idx, i_pos, protected_asset_position, world_size, 0.0, strength)

    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_features_arr = np.asarray(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 5), dtype=np.float32)
    adjacency = np.zeros((len(node_features), len(node_features)), dtype=np.float32)
    if edge_index.size:
        adjacency[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(adjacency, 1.0)
    return GraphData(
        node_features=np.asarray(node_features, dtype=np.float32),
        edge_index=edge_index,
        edge_features=edge_features_arr,
        adjacency=adjacency,
        defender_node_indices=defender_indices,
        intruder_node_indices=intruder_indices,
        asset_node_index=asset_idx,
    )


def radius_graph(positions: Array, radius: float, self_loops: bool = True) -> Array:
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adjacency = (distances <= radius).astype(np.float32)
    if not self_loops:
        np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _node_feature(node_type: int, position: Array, velocity: Array, threat: float, intercepted: float, comm: float, world_size: float) -> Array:
    type_one_hot = np.zeros(3, dtype=np.float32)
    type_one_hot[node_type] = 1.0
    return np.concatenate(
        [
            type_one_hot,
            np.asarray(position, dtype=np.float32) / max(world_size, 1e-6),
            np.asarray(velocity, dtype=np.float32) / 20.0,
            np.asarray([threat, intercepted, comm], dtype=np.float32),
        ]
    )


def _append_edge(
    edges: list[tuple[int, int]],
    edge_features: list[Array],
    source: int,
    target: int,
    source_positions: Array,
    target_positions: Array,
    world_size: float,
    comm: float,
    threat_strength: float,
    source_local_index: int | None = None,
    target_local_index: int | None = None,
) -> None:
    s_idx = source if source_local_index is None else source_local_index
    t_idx = target if target_local_index is None else target_local_index
    _append_single_edge(edges, edge_features, source, target, source_positions[s_idx], target_positions[t_idx], world_size, comm, threat_strength)


def _append_single_edge(edges: list[tuple[int, int]], edge_features: list[Array], source: int, target: int, source_pos: Array, target_pos: Array, world_size: float, comm: float, threat_strength: float) -> None:
    delta = target_pos - source_pos
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, 1e-6)
    edges.append((source, target))
    edge_features.append(np.asarray([distance / max(world_size, 1e-6), direction[0], direction[1], comm, threat_strength], dtype=np.float32))
