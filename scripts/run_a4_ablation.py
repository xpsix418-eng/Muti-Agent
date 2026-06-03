from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config
from visualize_rollout import MAPPOVisualizationPolicy


EXPERIMENTS = [
    {
        "name": "A3_intercept_point",
        "config": "configs/ablations/a3_intercept_point_reward.yaml",
        "result_dir": "experiments/results/ablations/A3_intercept_point_reward/Scenario5v5",
    },
    {
        "name": "A4_early_intercept",
        "config": "configs/train_mappo_a4_early_intercept.yaml",
        "result_dir": "experiments/results/ablations/A4_early_intercept/Scenario5v5",
    },
    {
        "name": "A4_dynamic_horizon",
        "config": "configs/train_mappo_a4_dynamic_horizon.yaml",
        "result_dir": "experiments/results/ablations/A4_dynamic_horizon/Scenario5v5",
    },
    {
        "name": "A4_soft_assignment",
        "config": "configs/train_mappo_a4_soft_assignment.yaml",
        "result_dir": "experiments/results/ablations/A4_soft_assignment/Scenario5v5",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        config_path = experiment["config"]
        result_dir = Path(experiment["result_dir"])
        checkpoint_path = result_dir / "checkpoints" / "latest.pt"
        if not args.skip_training:
            train(config_path, args.total_steps)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        if not args.skip_visualization:
            visualize(config_path, checkpoint_path, result_dir)
        metrics = evaluate_checkpoint(config_path, checkpoint_path, episodes=args.episodes, seed=args.seed)
        convergence = convergence_summary(result_dir)
        row = {"experiment": experiment["name"], **metrics, **convergence}
        rows.append(row)
        save_json(result_dir / "evaluation" / "metrics.json", row)

    output_path = Path("experiments") / "results" / "a4_ablation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary={output_path}")
    print(json.dumps(rows, indent=2))


def train(config_path: str, total_steps: int | None) -> None:
    command = [sys.executable, "scripts/train_mappo.py", "--config", config_path]
    if total_steps is not None:
        command.extend(["--total-steps", str(total_steps)])
    subprocess.run(command, check=True)


def visualize(config_path: str, checkpoint_path: Path, result_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/visualize_rollout.py",
            "--config",
            config_path,
            "--policy",
            "mappo",
            "--checkpoint",
            str(checkpoint_path),
            "--scenario",
            "Scenario5v5",
            "--max-steps",
            "200",
            "--output-dir",
            str(result_dir / "visualization"),
        ],
        check=True,
    )


def evaluate_checkpoint(config_path: str, checkpoint_path: Path, episodes: int, seed: int) -> dict[str, float]:
    env = CounterUAVEnv(apply_scenario_to_config(load_env_config(config_path), "Scenario5v5"))
    policy = MAPPOVisualizationPolicy(str(checkpoint_path), env)
    rows = [evaluate_episode(env, policy, seed + idx) for idx in range(episodes)]
    metrics: dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        metrics[key] = float(np.mean(finite)) if len(finite) else float("nan")
    return metrics


def evaluate_episode(env: CounterUAVEnv, policy: MAPPOVisualizationPolicy, seed: int) -> dict[str, float]:
    observations, info = env.reset(seed=seed)
    previous_intercepted = np.zeros(env.config.num_intruders, dtype=bool)
    total_energy = 0.0
    total_collisions = 0.0
    total_defender_distance = 0.0
    total_blocking = 0.0
    total_blocking_denominator = 0.0
    intercept_distances: list[float] = []
    intercept_times_to_asset: list[float] = []
    steps = 0

    for _ in range(env.config.max_steps):
        actions = policy.act(observations, env.defense_agents)
        observations, _, terminations, truncations, info = env.step(actions)
        agent_info = info[env.defense_agents[0]]
        intercepted = np.asarray(agent_info["intercepted"], dtype=bool)
        newly_intercepted = intercepted & ~previous_intercepted
        if np.any(newly_intercepted):
            positions = agent_info["intruder_positions"][newly_intercepted]
            velocities = agent_info["intruder_velocities"][newly_intercepted]
            distances = np.linalg.norm(positions - env.protected_asset[None, :], axis=1)
            speeds = np.linalg.norm(velocities, axis=1)
            speeds = np.where(speeds < 1e-6, env.config.intruder_max_speed, speeds)
            intercept_distances.extend(distances.tolist())
            intercept_times_to_asset.extend((distances / speeds).tolist())
        previous_intercepted = intercepted

        total_energy += float(np.mean(np.linalg.norm(actions, axis=1)))
        total_collisions += float(len(agent_info["collision_events"]))
        total_defender_distance += float(np.mean(np.linalg.norm(agent_info["defender_positions"] - env.protected_asset[None, :], axis=1)))
        total_blocking += float(np.sum(agent_info.get("blocking_flags", np.zeros(env.config.num_defenders))))
        total_blocking_denominator += float(env.config.num_defenders)
        steps += 1
        if terminations["__all__"] or truncations["__all__"]:
            break

    final_info = info[env.defense_agents[0]]
    return {
        "intercept_rate": float(np.mean(final_info["intercepted"])),
        "breach_rate": float(np.mean(final_info["breached"])),
        "collision_rate": float(total_collisions / max(steps, 1)),
        "blocking_success_rate": float(total_blocking / max(total_blocking_denominator, 1.0)),
        "average_intercept_distance_to_asset": float(np.mean(intercept_distances)) if intercept_distances else float("nan"),
        "average_intercept_time_to_asset": float(np.mean(intercept_times_to_asset)) if intercept_times_to_asset else float("nan"),
        "average_defender_distance_to_asset": float(total_defender_distance / max(steps, 1)),
        "average_energy_cost": float(total_energy / max(steps, 1)),
        "steps": float(steps),
    }


def convergence_summary(result_dir: Path) -> dict[str, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    accumulator = EventAccumulator(str(result_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    summary: dict[str, float] = {}
    for tag in ["summary/intercept_rate", "summary/blocking_success_rate", "summary/episode_reward", "summary/learning_rate"]:
        if tag not in tags:
            continue
        values = np.asarray([event.value for event in accumulator.Scalars(tag)], dtype=np.float64)
        if len(values) == 0:
            continue
        tail = values[-20:] if len(values) >= 20 else values
        prefix = tag.replace("summary/", "curve_")
        summary[f"{prefix}_tail_mean"] = float(np.mean(tail))
        summary[f"{prefix}_last"] = float(values[-1])
    return summary


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
