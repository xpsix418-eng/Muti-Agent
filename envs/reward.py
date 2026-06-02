from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from envs.threat_model import ThreatAssessment


Array = np.ndarray
RewardMode = Literal["individual_reward", "team_reward"]


@dataclass(frozen=True)
class RewardConfig:
    capture: float = 10.0
    protected_zone_breach: float = -20.0
    distance_shaping: float = 0.0
    energy_penalty: float = 0.01
    step_penalty: float = -0.01
    alpha_intercept: float = 1.0
    alpha_high_threat: float = 1.0
    alpha_protect: float = 0.2
    beta_breach: float = 1.0
    beta_collision: float = 1.0
    beta_energy: float = 1.0
    beta_comm: float = 1.0
    beta_time: float = 1.0
    high_threat_threshold: float = 0.7
    intercept_reward: float = 10.0
    high_threat_intercept_reward: float = 5.0
    asset_protection_reward: float = 0.1
    breach_penalty: float = 25.0
    collision_penalty: float = 2.0
    communication_penalty: float = 0.02
    time_penalty: float = 0.01


@dataclass(frozen=True)
class RewardEvents:
    intercepted: Array
    breached: Array
    defender_collision_pairs: list[tuple[int, int]]
    communication_links: int


def predict_intercept_points(
    intruder_positions: Array,
    intruder_velocities: Array,
    prediction_horizon: float,
    world_size: float | None = None,
) -> Array:
    """Predict future points on synthetic intruder trajectories."""
    points = np.asarray(intruder_positions, dtype=np.float32) + np.asarray(intruder_velocities, dtype=np.float32) * float(
        prediction_horizon
    )
    if world_size is not None:
        points = np.clip(points, 0.0, float(world_size))
    return points.astype(np.float32)


def intercept_point_approach_reward(
    previous_defender_positions: Array,
    current_defender_positions: Array,
    previous_intercept_points: Array,
    current_intercept_points: Array,
    assigned_targets: Array,
) -> Array:
    assigned_targets = np.asarray(assigned_targets, dtype=np.int64)
    rewards = np.zeros(len(assigned_targets), dtype=np.float32)
    valid = assigned_targets >= 0
    if not np.any(valid):
        return rewards
    defender_indices = np.where(valid)[0]
    target_indices = assigned_targets[valid]
    previous_distances = np.linalg.norm(
        previous_defender_positions[defender_indices] - previous_intercept_points[target_indices],
        axis=1,
    )
    current_distances = np.linalg.norm(
        current_defender_positions[defender_indices] - current_intercept_points[target_indices],
        axis=1,
    )
    rewards[defender_indices] = (previous_distances - current_distances).astype(np.float32)
    return rewards


def blocking_position_reward(
    defender_positions: Array,
    intruder_positions: Array,
    protected_asset_position: Array,
    assigned_targets: Array,
    blocking_sigma: float,
) -> Array:
    assigned_targets = np.asarray(assigned_targets, dtype=np.int64)
    rewards = np.zeros(len(assigned_targets), dtype=np.float32)
    sigma = max(float(blocking_sigma), 1e-6)
    for defender_idx, target_idx in enumerate(assigned_targets.tolist()):
        if target_idx < 0:
            continue
        intruder_pos = intruder_positions[target_idx]
        attack_vector = protected_asset_position - intruder_pos
        attack_length = float(np.linalg.norm(attack_vector))
        if attack_length <= 1e-6:
            continue
        defender_vector = defender_positions[defender_idx] - intruder_pos
        projection = float(np.dot(defender_vector, attack_vector) / attack_length)
        if projection <= 0.0 or projection >= attack_length:
            continue
        cross = attack_vector[0] * defender_vector[1] - attack_vector[1] * defender_vector[0]
        perpendicular = float(abs(cross) / attack_length)
        rewards[defender_idx] = float(np.exp(-perpendicular / sigma))
    return rewards


def ttc_advantage_reward(
    defender_positions: Array,
    intruder_positions: Array,
    intruder_velocities: Array,
    protected_asset_position: Array,
    intercept_points: Array,
    assigned_targets: Array,
    defender_max_speed: float,
) -> Array:
    assigned_targets = np.asarray(assigned_targets, dtype=np.int64)
    rewards = np.zeros(len(assigned_targets), dtype=np.float32)
    defender_speed = max(float(defender_max_speed), 1e-6)
    for defender_idx, target_idx in enumerate(assigned_targets.tolist()):
        if target_idx < 0:
            continue
        intruder_speed = max(float(np.linalg.norm(intruder_velocities[target_idx])), 1e-6)
        time_to_asset = np.linalg.norm(intruder_positions[target_idx] - protected_asset_position) / intruder_speed
        time_to_intercept = np.linalg.norm(defender_positions[defender_idx] - intercept_points[target_idx]) / defender_speed
        if time_to_intercept < time_to_asset:
            rewards[defender_idx] = float((time_to_asset - time_to_intercept) / max(time_to_asset, 1e-6))
    return rewards


def intruder_progress_penalty(
    previous_intruder_positions: Array,
    current_intruder_positions: Array,
    protected_asset_position: Array,
    active_mask: Array,
) -> float:
    active = np.asarray(active_mask, dtype=bool)
    if not np.any(active):
        return 0.0
    previous_distances = np.linalg.norm(previous_intruder_positions[active] - protected_asset_position[None, :], axis=1)
    current_distances = np.linalg.norm(current_intruder_positions[active] - protected_asset_position[None, :], axis=1)
    progress = np.maximum(previous_distances - current_distances, 0.0)
    return float(np.mean(progress))


def compute_reward(
    actions: Array,
    threat_scores: Array,
    events: RewardEvents,
    config: RewardConfig,
    num_defenders: int,
    mode: RewardMode = "team_reward",
    interceptor_ids: Array | None = None,
) -> dict[str, float]:
    actions = np.asarray(actions, dtype=np.float32)
    intercepted = np.asarray(events.intercepted, dtype=bool)
    breached = np.asarray(events.breached, dtype=bool)
    threat_scores = np.asarray(threat_scores, dtype=np.float32)

    intercept_count = float(np.sum(intercepted))
    high_threat_count = float(np.sum(intercepted & (threat_scores >= config.high_threat_threshold)))
    breach_count = float(np.sum(breached))
    collision_count = float(len(events.defender_collision_pairs))
    energy_cost = float(np.mean(np.linalg.norm(actions, axis=1))) if len(actions) else 0.0
    comm_cost = float(events.communication_links)
    active_threat = float(np.mean(threat_scores[~intercepted])) if np.any(~intercepted) else 0.0

    shared = (
        config.alpha_intercept * config.intercept_reward * intercept_count
        + config.alpha_high_threat * config.high_threat_intercept_reward * high_threat_count
        + config.alpha_protect * config.asset_protection_reward * (1.0 - active_threat)
        - config.beta_breach * config.breach_penalty * breach_count
        - config.beta_collision * config.collision_penalty * collision_count
        - config.beta_energy * config.energy_penalty * energy_cost
        - config.beta_comm * config.communication_penalty * comm_cost
        - config.beta_time * config.time_penalty
    )

    if mode == "team_reward":
        return {f"defender_{idx}": float(shared) for idx in range(num_defenders)}
    if mode != "individual_reward":
        raise ValueError(f"Unsupported reward mode: {mode}")

    rewards = np.full(num_defenders, shared * 0.2, dtype=np.float32)
    rewards -= config.beta_energy * config.energy_penalty * np.linalg.norm(actions, axis=1)
    for idx, _ in events.defender_collision_pairs:
        rewards[idx] -= config.beta_collision * config.collision_penalty
    for _, idx in events.defender_collision_pairs:
        rewards[idx] -= config.beta_collision * config.collision_penalty
    if interceptor_ids is not None:
        for defender_idx in np.asarray(interceptor_ids, dtype=np.int64):
            if 0 <= defender_idx < num_defenders:
                rewards[defender_idx] += config.alpha_intercept * config.intercept_reward
    return {f"defender_{idx}": float(rewards[idx]) for idx in range(num_defenders)}


def detect_intercepts(defender_positions: Array, intruder_positions: Array, intercept_radius: float) -> tuple[Array, Array]:
    distances = np.linalg.norm(defender_positions[:, None, :] - intruder_positions[None, :, :], axis=-1)
    nearest_defenders = np.argmin(distances, axis=0)
    intercepted = distances.min(axis=0) <= intercept_radius
    return intercepted, nearest_defenders.astype(np.int64)


def detect_breaches(intruder_positions: Array, protected_asset_position: Array, protected_radius: float) -> Array:
    distances = np.linalg.norm(intruder_positions - protected_asset_position[None, :], axis=1)
    return distances <= protected_radius


def detect_defender_collisions(defender_positions: Array, collision_radius: float) -> list[tuple[int, int]]:
    distances = np.linalg.norm(defender_positions[:, None, :] - defender_positions[None, :, :], axis=-1)
    pair_mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    rows, cols = np.where((distances <= collision_radius) & pair_mask)
    return list(zip(rows.tolist(), cols.tolist()))


def team_reward(assessment: ThreatAssessment, actions: Array, config: RewardConfig) -> float:
    capture_reward = float(np.sum(assessment.captured) * config.capture)
    breach_penalty = float(np.sum(assessment.breached) * config.protected_zone_breach)
    distance_reward = float(-config.distance_shaping * np.mean(assessment.zone_distances))
    energy_cost = float(config.energy_penalty * np.mean(np.square(actions)))
    return capture_reward + breach_penalty + distance_reward - energy_cost + config.step_penalty
