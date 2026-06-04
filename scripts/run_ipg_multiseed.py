from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import _bootstrap  # noqa: F401
from scripts.run_ipg_validation import METHODS, check_shared_environment, load_extended_yaml, resolve_checkpoint


METRICS = [
    "intercept_rate",
    "breach_rate",
    "collision_rate",
    "blocking_success_rate",
    "success_rate",
    "average_intercept_time",
    "average_energy_cost",
    "assignment_entropy",
    "mean_interception_time_advantage",
    "graph_attention_sparsity",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario", default="Scenario5v5")
    parser.add_argument("--output-dir", default="experiments/results/ipg_multiseed")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    configs = {method.name: load_extended_yaml(method.config) for method in METHODS}
    check_shared_environment(configs)

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        checkpoint = resolve_checkpoint(method)
        for mode in ["deterministic", "stochastic"]:
            per_seed = []
            for seed in seeds:
                metrics = evaluate(method, checkpoint, args, seed, stochastic=(mode == "stochastic"))
                row = {"method": method.name, "mode": mode, "seed": seed, **metrics}
                raw_rows.append(row)
                per_seed.append(metrics)
            summary_rows.extend(summarize(method.name, mode, per_seed))
    write_csv(output_root / "raw_metrics.csv", raw_rows)
    write_csv(output_root / "mean_std_metrics.csv", summary_rows)
    with (output_root / "mean_std_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump({"rows": summary_rows}, file, sort_keys=False, allow_unicode=True)
    print(json.dumps(summary_rows, indent=2))


def evaluate(method, checkpoint: Path, args: argparse.Namespace, seed: int, stochastic: bool) -> dict[str, Any]:
    mode = "stochastic" if stochastic else "deterministic"
    experiment_name = f"ipg_multiseed/{method.slug}/{mode}/seed_{seed}"
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        method.config,
        "--policy",
        method.policy,
        "--checkpoint",
        str(checkpoint),
        "--scenario",
        args.scenario,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--experiment-name",
        experiment_name,
    ]
    if stochastic:
        command.append("--stochastic")
    subprocess.run(command, check=True)
    metrics_path = Path("experiments") / "results" / experiment_name / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def summarize(method_name: str, mode: str, per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        values = np.asarray([float(row.get(metric, np.nan)) for row in per_seed], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=0))
        rows.append(
            {
                "method": method_name,
                "mode": mode,
                "metric": metric,
                "mean": mean,
                "std": std,
                "mean_pm_std": f"{mean:.6f} ± {std:.6f}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
