from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


EXPERIMENTS = [
    {
        "name": "SA-PMAPPO",
        "policy": "mappo",
        "config": "configs/ablations/a5_1_soft_assignment_lr_floor.yaml",
        "checkpoint": "experiments/results/ablations/A5_1_soft_assignment_lr_floor/Scenario5v5/checkpoints/latest.pt",
        "train": ["scripts/train_mappo.py", "--config", "configs/ablations/a5_1_soft_assignment_lr_floor.yaml"],
    },
    {
        "name": "GNN-MAPPO",
        "policy": "mappo",
        "config": "configs/train_gnn_mappo.yaml",
        "checkpoint": "experiments/results/gnn_mappo/Scenario5v5/checkpoints/latest.pt",
        "train": ["scripts/train_gnn_mappo.py", "--config", "configs/train_gnn_mappo.yaml"],
    },
    {
        "name": "IPGA-MAPPO",
        "policy": "ipga_mappo",
        "config": "configs/train_ipga_mappo_5v5.yaml",
        "checkpoint": "experiments/results/ipga_mappo/Scenario5v5/checkpoints/latest.pt",
        "train": ["scripts/train_ipga_mappo.py", "--config", "configs/train_ipga_mappo_5v5.yaml"],
    },
    {
        "name": "IPGA without assignment gate",
        "policy": "ipga_mappo",
        "config": "configs/train_ipga_no_assignment_gate.yaml",
        "checkpoint": "experiments/results/ipga_ablation/no_assignment_gate/Scenario5v5/checkpoints/latest.pt",
        "train": ["scripts/train_ipga_mappo.py", "--config", "configs/train_ipga_no_assignment_gate.yaml"],
    },
    {
        "name": "IPGA without ITA",
        "policy": "ipga_mappo",
        "config": "configs/train_ipga_no_ita.yaml",
        "checkpoint": "experiments/results/ipga_ablation/no_ita/Scenario5v5/checkpoints/latest.pt",
        "train": ["scripts/train_ipga_mappo.py", "--config", "configs/train_ipga_no_ita.yaml"],
    },
    {
        "name": "IPGA without assignment auxiliary loss",
        "policy": "ipga_mappo",
        "config": "configs/train_ipga_no_assignment_loss.yaml",
        "checkpoint": "experiments/results/ipga_ablation/no_assignment_loss/Scenario5v5/checkpoints/latest.pt",
        "train": ["scripts/train_ipga_mappo.py", "--config", "configs/train_ipga_no_assignment_loss.yaml"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="experiments/results/ipga_ablation_summary.csv")
    args = parser.parse_args()

    rows = []
    for experiment in EXPERIMENTS:
        checkpoint = Path(experiment["checkpoint"])
        if not args.skip_training and not checkpoint.exists():
            run_command([sys.executable, *experiment["train"]])
        if not checkpoint.exists():
            print(f"skip={experiment['name']} missing_checkpoint={checkpoint}")
            continue
        metrics_path = evaluate_experiment(experiment, checkpoint, args.episodes)
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        rows.append({"experiment": experiment["name"], **metrics})
    save_summary(rows, Path(args.output))
    print(f"summary={args.output}")


def evaluate_experiment(experiment: dict, checkpoint: Path, episodes: int) -> Path:
    experiment_name = "ipga_ablation/" + experiment["name"].lower().replace(" ", "_").replace("-", "_")
    run_command(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            experiment["config"],
            "--policy",
            experiment["policy"],
            "--checkpoint",
            str(checkpoint),
            "--scenario",
            "Scenario5v5",
            "--episodes",
            str(episodes),
            "--experiment-name",
            experiment_name,
        ]
    )
    return Path("experiments") / "results" / experiment_name / "metrics.json"


def save_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["experiment", *[key for key in rows[0] if key != "experiment"]]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
