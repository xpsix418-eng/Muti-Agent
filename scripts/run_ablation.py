from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from algorithms.hierarchical_marl.high_level_policy import HighLevelPolicy
from algorithms.hierarchical_marl.low_level_policy import LowLevelPolicy
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config, scenario_metadata
from scripts.evaluate import build_policy, evaluate_episode, summarize_metrics


SCENARIOS = ["ScenarioA", "ScenarioB", "ScenarioC", "ScenarioD", "ScenarioE", "ScenarioF"]
ALGORITHMS = [
    "MAPPO",
    "MAPPO + Threat Reward",
    "GNN-MAPPO",
    "GNN-MAPPO + Communication Constraint",
    "Hierarchical GNN-MAPPO",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mappo-checkpoint", default=None)
    parser.add_argument("--gnn-checkpoint", default=None)
    parser.add_argument("--output-dir", default="experiments/results/ablation")
    args = parser.parse_args()

    rows = []
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            row = evaluate_algorithm(args, algorithm, scenario)
            rows.append(row)
    save_ablation_results(Path(args.output_dir), rows)
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))


def evaluate_algorithm(args: argparse.Namespace, algorithm: str, scenario: str) -> dict[str, float | str]:
    checkpoint = args.mappo_checkpoint if algorithm.startswith("MAPPO") else args.gnn_checkpoint
    base_config = apply_scenario_to_config(load_env_config(args.config), scenario)
    if "Communication Constraint" in algorithm:
        base_config = apply_scenario_to_config(load_env_config(args.config), "ScenarioE")
    env = CounterUAVEnv(base_config)
    metadata = scenario_metadata(scenario)

    if algorithm.startswith("MAPPO") or algorithm.startswith("GNN-MAPPO"):
        if checkpoint is None:
            return {"algorithm": algorithm, "scenario": scenario, "status": "missing_checkpoint"}
        policy = build_policy("mappo", env, checkpoint)
    else:
        policy = HierarchicalRolloutPolicy()

    episode_rows = [
        evaluate_episode(env, policy, scenario, seed=args.seed + idx, metadata=metadata)
        for idx in range(args.episodes)
    ]
    metrics = summarize_metrics(episode_rows, env, metadata)
    return {"algorithm": algorithm, "scenario": scenario, "status": "ok", **metrics}


class HierarchicalRolloutPolicy:
    def __init__(self) -> None:
        self.high = HighLevelPolicy(high_level_interval=10)
        self.low = LowLevelPolicy()
        self.step = 0

    def act(
        self,
        defender_positions,
        defender_velocities,
        intruder_positions,
        intruder_active,
        threat_scores,
        protected_asset_position,
        world_size,
    ):
        del world_size
        active_positions = intruder_positions[intruder_active]
        active_threats = threat_scores[intruder_active]
        info = {
            "threat_scores": active_threats,
            "intruder_positions": active_positions,
            "protected_asset_position": protected_asset_position,
        }
        high_action = self.high.select_action(self.step, info)
        self.step += 1
        return self.low.act(high_action, defender_positions, defender_velocities, active_positions, protected_asset_position)


def save_ablation_results(output_dir: Path, rows: list[dict[str, float | str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "ablation_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "ablation_results.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


if __name__ == "__main__":
    main()
