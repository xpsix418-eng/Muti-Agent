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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    parser.add_argument("--policy", choices=["rule_based", "hungarian"], default="rule_based")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/results/baseline")
    args = parser.parse_args()

    env = CounterUAVEnv(load_env_config(args.config))
    policy = build_policy(args.policy)
    episode_metrics = [
        run_episode(env, policy, seed=args.seed + episode_idx) for episode_idx in range(args.episodes)
    ]
    summary = summarize_metrics(episode_metrics)
    save_results(args.output_dir, args.policy, summary, episode_metrics)
    print_summary(args.policy, summary)


def build_policy(policy_name: str) -> RuleBasedPolicy | HungarianAssignmentPolicy:
    if policy_name == "rule_based":
        return RuleBasedPolicy()
    if policy_name == "hungarian":
        return HungarianAssignmentPolicy()
    raise ValueError(f"Unsupported policy: {policy_name}")


def run_episode(
    env: CounterUAVEnv,
    policy: RuleBasedPolicy | HungarianAssignmentPolicy,
    seed: int,
) -> dict[str, float]:
    _, info = env.reset(seed=seed)
    total_energy = 0.0
    total_collisions = 0
    steps = 0

    for _ in range(env.config.max_steps):
        agent_info = info[env.defense_agents[0]]
        actions = policy_action(env, policy, agent_info)
        _, _, terminations, truncations, info = env.step(actions)
        total_energy += float(np.mean(np.linalg.norm(actions, axis=1)))
        total_collisions += len(info[env.defense_agents[0]]["collision_events"])
        steps += 1
        if terminations["__all__"] or truncations["__all__"]:
            break

    final_info = info[env.defense_agents[0]]
    intercepted = np.asarray(final_info["intercepted"], dtype=bool)
    breached = np.asarray(final_info["breached"], dtype=bool)
    return {
        "intercept_rate": float(np.mean(intercepted)),
        "breach_rate": float(np.mean(breached)),
        "collision_rate": float(total_collisions / max(steps, 1)),
        "average_energy": float(total_energy / max(steps, 1)),
        "steps": float(steps),
    }


def policy_action(
    env: CounterUAVEnv,
    policy: RuleBasedPolicy | HungarianAssignmentPolicy,
    info: dict[str, Any],
) -> np.ndarray:
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


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics[0].keys()
    return {key: float(np.mean([episode[key] for episode in metrics])) for key in keys}


def save_results(
    output_dir: str | Path,
    policy_name: str,
    summary: dict[str, float],
    episode_metrics: list[dict[str, float]],
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    summary_path = path / f"{policy_name}_summary.json"
    episodes_path = path / f"{policy_name}_episodes.csv"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with episodes_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(episode_metrics[0]))
        writer.writeheader()
        writer.writerows(episode_metrics)


def print_summary(policy_name: str, summary: dict[str, float]) -> None:
    print(f"policy={policy_name}")
    print(f"average_intercept_rate={summary['intercept_rate']:.3f}")
    print(f"average_breach_rate={summary['breach_rate']:.3f}")
    print(f"average_collision_rate={summary['collision_rate']:.3f}")
    print(f"average_energy={summary['average_energy']:.3f}")


if __name__ == "__main__":
    main()
