from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401
from algorithms.hungarian_assignment import HungarianAssignmentPolicy
from algorithms.rule_based import RuleBasedPolicy
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config, initialize_scenario_state, scenario_metadata


Policy = RuleBasedPolicy | HungarianAssignmentPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    parser.add_argument("--policy", choices=["rule_based", "hungarian"], default="rule_based")
    parser.add_argument("--scenario", default="ScenarioB")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()

    base_config = load_env_config(args.config)
    env = CounterUAVEnv(apply_scenario_to_config(base_config, args.scenario))
    policy = build_policy(args.policy)
    metadata = scenario_metadata(args.scenario)
    episode_rows = [
        evaluate_episode(env, policy, args.scenario, seed=args.seed + idx, metadata=metadata)
        for idx in range(args.episodes)
    ]
    metrics = summarize_metrics(episode_rows, env, metadata)
    experiment_name = args.experiment_name or f"{args.policy}_{args.scenario}"
    save_evaluation_results(experiment_name, metrics, episode_rows)
    print(json.dumps(metrics, indent=2))


def build_policy(policy_name: str) -> Policy:
    if policy_name == "rule_based":
        return RuleBasedPolicy()
    if policy_name == "hungarian":
        return HungarianAssignmentPolicy()
    raise ValueError(f"Unsupported policy: {policy_name}")


def evaluate_episode(
    env: CounterUAVEnv,
    policy: Policy,
    scenario_name: str,
    seed: int,
    metadata: dict[str, float | int | str],
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    initialize_scenario_state(env, scenario_name, rng)
    info = env._infos()
    initial_info = info[env.defense_agents[0]]
    high_threat_mask = initial_info["threat_scores"] >= 0.7
    previous_intercepted = np.zeros(env.config.num_intruders, dtype=bool)
    previous_breached = np.zeros(env.config.num_intruders, dtype=bool)
    intercept_times = np.full(env.config.num_intruders, np.nan, dtype=np.float32)
    intercept_distances_to_asset: list[float] = []
    intercept_time_to_asset: list[float] = []

    total_energy = 0.0
    total_collisions = 0
    total_communication_links = 0.0
    total_defender_distance_to_asset = 0.0
    total_blocking_flags = 0.0
    total_blocking_denominator = 0.0
    steps = 0

    for _ in range(env.config.max_steps):
        agent_info = info[env.defense_agents[0]]
        actions = policy_action(env, policy, agent_info)
        _, _, terminations, truncations, info = env.step(actions)
        agent_info = info[env.defense_agents[0]]
        intercepted = np.asarray(agent_info["intercepted"], dtype=bool)
        breached = np.asarray(agent_info["breached"], dtype=bool)
        newly_intercepted = intercepted & ~previous_intercepted
        newly_breached = breached & ~previous_breached
        intercept_times[newly_intercepted] = env.step_count
        if np.any(newly_intercepted):
            intercepted_positions = agent_info["intruder_positions"][newly_intercepted]
            intercepted_velocities = agent_info["intruder_velocities"][newly_intercepted]
            distances_to_asset = np.linalg.norm(intercepted_positions - env.protected_asset[None, :], axis=1)
            speeds = np.linalg.norm(intercepted_velocities, axis=1)
            speeds = np.where(speeds < 1e-6, env.config.intruder_max_speed, speeds)
            intercept_distances_to_asset.extend(distances_to_asset.tolist())
            intercept_time_to_asset.extend((distances_to_asset / speeds).tolist())
        previous_intercepted = intercepted
        previous_breached |= newly_breached

        total_energy += float(np.mean(np.linalg.norm(actions, axis=1)))
        total_collisions += len(agent_info["collision_events"])
        total_communication_links += communication_cost(agent_info["communication_topology"], metadata)
        total_defender_distance_to_asset += float(
            np.mean(np.linalg.norm(agent_info["defender_positions"] - env.protected_asset[None, :], axis=1))
        )
        total_blocking_flags += float(np.sum(agent_info.get("blocking_flags", np.zeros(env.config.num_defenders))))
        total_blocking_denominator += float(env.config.num_defenders)
        steps += 1
        if terminations["__all__"] or truncations["__all__"]:
            break

    final_info = info[env.defense_agents[0]]
    intercepted = np.asarray(final_info["intercepted"], dtype=bool)
    breached = np.asarray(final_info["breached"], dtype=bool)
    high_threat_denominator = max(int(np.sum(high_threat_mask)), 1)
    valid_intercept_times = intercept_times[~np.isnan(intercept_times)]
    success = float(np.all(intercepted) and not np.any(breached))
    return {
        "intercept_rate": float(np.mean(intercepted)),
        "breach_rate": float(np.mean(breached)),
        "high_threat_intercept_rate": float(np.sum(intercepted & high_threat_mask) / high_threat_denominator),
        "average_intercept_time": float(np.mean(valid_intercept_times)) if len(valid_intercept_times) else float(env.config.max_steps),
        "average_energy_cost": float(total_energy / max(steps, 1)),
        "collision_rate": float(total_collisions / max(steps, 1)),
        "communication_cost": float(total_communication_links / max(steps, 1)),
        "success_rate": success,
        "average_defender_distance_to_asset": float(total_defender_distance_to_asset / max(steps, 1)),
        "average_intercept_distance_to_asset": float(np.mean(intercept_distances_to_asset)) if intercept_distances_to_asset else float("nan"),
        "average_intercept_time_to_asset": float(np.mean(intercept_time_to_asset)) if intercept_time_to_asset else float("nan"),
        "blocking_success_rate": float(total_blocking_flags / max(total_blocking_denominator, 1.0)),
        "steps": float(steps),
    }


def policy_action(env: CounterUAVEnv, policy: Policy, info: dict[str, Any]) -> np.ndarray:
    common = {
        "defender_positions": info["defender_positions"],
        "defender_velocities": info["defender_velocities"],
        "intruder_positions": info["intruder_positions"],
        "intruder_active": ~(info["intercepted"] | info["breached"]),
        "threat_scores": info["threat_scores"],
        "protected_asset_position": env.protected_asset,
        "world_size": env.config.world_size,
    }
    if isinstance(policy, HungarianAssignmentPolicy):
        return policy.act(intruder_velocities=info["intruder_velocities"], **common)
    return policy.act(**common)


def communication_cost(topology: np.ndarray, metadata: dict[str, float | int | str]) -> float:
    directed_links = float(np.sum(topology))
    undirected_links = directed_links / 2.0
    packet_loss = float(metadata.get("packet_loss", 0.0))
    delay = float(metadata.get("communication_delay", 0))
    return undirected_links * (1.0 + packet_loss + 0.1 * delay)


def summarize_metrics(
    episode_rows: list[dict[str, float]],
    env: CounterUAVEnv,
    metadata: dict[str, float | int | str],
) -> dict[str, float]:
    metrics = {
        key: float(np.mean([row[key] for row in episode_rows]))
        for key in episode_rows[0]
        if key != "steps"
    }
    metrics["scalability_score"] = scalability_score(metrics["intercept_rate"], env.config.num_defenders, env.config.num_intruders)
    metrics["robustness_score"] = robustness_score(metrics["success_rate"], metrics["breach_rate"], metadata)
    return metrics


def scalability_score(intercept_rate: float, num_defenders: int, num_intruders: int) -> float:
    baseline_size = 24.0
    scale_penalty = np.sqrt(baseline_size / max(num_defenders + num_intruders, 1))
    return float(np.clip(intercept_rate * scale_penalty, 0.0, 1.0))


def robustness_score(success_rate: float, breach_rate: float, metadata: dict[str, float | int | str]) -> float:
    packet_loss = float(metadata.get("packet_loss", 0.0))
    delay = float(metadata.get("communication_delay", 0))
    noise = float(metadata.get("observation_noise", 0.0))
    degradation = 1.0 + packet_loss + 0.1 * delay + 0.01 * noise
    return float(np.clip((success_rate * (1.0 - breach_rate)) / degradation, 0.0, 1.0))


def save_evaluation_results(
    experiment_name: str,
    metrics: dict[str, float],
    episode_rows: list[dict[str, float]],
) -> None:
    output_dir = Path("experiments") / "results" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (output_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", *episode_rows[0].keys()])
        writer.writeheader()
        for idx, row in enumerate(episode_rows):
            writer.writerow({"episode": idx, **row})


if __name__ == "__main__":
    main()
