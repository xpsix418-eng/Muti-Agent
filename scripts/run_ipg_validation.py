from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import _bootstrap  # noqa: F401
from envs.config import load_yaml


ENV_KEYS = [
    "num_defenders",
    "num_intruders",
    "intercept_radius",
    "protected_radius",
    "defender_max_speed",
    "intruder_max_speed",
    "max_steps",
    "intruder_behavior",
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    policy: str
    config: str
    checkpoint_candidates: tuple[str, ...]

    @property
    def slug(self) -> str:
        return self.name.lower().replace("+", "plus").replace(" ", "_").replace("-", "_")


METHODS = [
    MethodSpec(
        "SA-PMAPPO",
        "mappo",
        "configs/ablations/a5_1_soft_assignment_lr_floor.yaml",
        ("experiments/results/ablations/A5_1_soft_assignment_lr_floor/Scenario5v5/checkpoints/latest.pt",),
    ),
    MethodSpec(
        "IPG-MAPPO",
        "ipga_mappo",
        "configs/train_ipg_mappo_5v5.yaml",
        (
            "experiments/results/ipg_mappo/Scenario5v5/checkpoints/latest.pt",
            "experiments/results/ipga_ablation/no_assignment_gate/Scenario5v5/checkpoints/latest.pt",
        ),
    ),
    MethodSpec(
        "IPG-MAPPO without graph",
        "ipga_mappo",
        "configs/train_ipga_no_graph.yaml",
        ("experiments/results/ipga_ablation/no_graph/Scenario5v5/checkpoints/latest.pt",),
    ),
    MethodSpec(
        "IPG-MAPPO without ITA",
        "ipga_mappo",
        "configs/train_ipga_no_ita.yaml",
        ("experiments/results/ipga_ablation/no_ita/Scenario5v5/checkpoints/latest.pt",),
    ),
    MethodSpec(
        "IPG-MAPPO + Assignment Gate",
        "ipga_mappo",
        "configs/train_ipga_mappo_5v5.yaml",
        ("experiments/results/ipga_mappo/Scenario5v5/checkpoints/latest.pt",),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", default="Scenario5v5")
    parser.add_argument("--output-dir", default="experiments/results/ipg_validation")
    parser.add_argument("--max-visual-steps", type=int, default=160)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_configs = {method.name: load_extended_yaml(method.config) for method in METHODS}
    check_shared_environment(resolved_configs)
    rows = []
    for method in METHODS:
        checkpoint = resolve_checkpoint(method)
        method_dir = output_root / method.slug
        method_dir.mkdir(parents=True, exist_ok=True)
        resolved = resolved_configs[method.name]
        write_yaml(method_dir / "resolved_config.yaml", resolved)
        (method_dir / "checkpoint_path.txt").write_text(str(checkpoint), encoding="utf-8")
        deterministic = evaluate_method(method, checkpoint, args, method_dir, stochastic=False)
        stochastic = evaluate_method(method, checkpoint, args, method_dir, stochastic=True)
        save_rollout(method, checkpoint, args, method_dir)
        rows.append(
            {
                "method": method.name,
                "deterministic_intercept_rate": deterministic["intercept_rate"],
                "deterministic_success_rate": deterministic["success_rate"],
                "deterministic_collision_rate": deterministic["collision_rate"],
                "deterministic_blocking_success_rate": deterministic["blocking_success_rate"],
                "stochastic_intercept_rate": stochastic["intercept_rate"],
                "stochastic_success_rate": stochastic["success_rate"],
                "stochastic_collision_rate": stochastic["collision_rate"],
                "stochastic_blocking_success_rate": stochastic["blocking_success_rate"],
            }
        )
    write_yaml(output_root / "validation_summary.yaml", {"rows": rows})
    print(json.dumps(rows, indent=2))


def evaluate_method(
    method: MethodSpec,
    checkpoint: Path,
    args: argparse.Namespace,
    method_dir: Path,
    stochastic: bool,
) -> dict[str, Any]:
    filename = "stochastic_eval_100ep.json" if stochastic else "deterministic_eval_100ep.json"
    experiment_name = f"ipg_validation/{method.slug}/{'stochastic' if stochastic else 'deterministic'}"
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
        str(args.seed),
        "--experiment-name",
        experiment_name,
    ]
    if stochastic:
        command.append("--stochastic")
    run(command)
    metrics_path = Path("experiments") / "results" / experiment_name / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    with (method_dir / filename).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def save_rollout(method: MethodSpec, checkpoint: Path, args: argparse.Namespace, method_dir: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/visualize_rollout.py",
            "--config",
            method.config,
            "--policy",
            method.policy,
            "--checkpoint",
            str(checkpoint),
            "--scenario",
            args.scenario,
            "--max-steps",
            str(args.max_visual_steps),
            "--output-dir",
            str(method_dir),
        ]
    )


def resolve_checkpoint(method: MethodSpec) -> Path:
    for candidate in method.checkpoint_candidates:
        path = Path(candidate)
        if path.exists():
            return path
    candidates = ", ".join(method.checkpoint_candidates)
    raise FileNotFoundError(f"Missing checkpoint for {method.name}; checked: {candidates}")


def check_shared_environment(configs: dict[str, dict[str, Any]]) -> None:
    reference_name, reference = next(iter(configs.items()))
    reference_env = reference["env"]
    for key in ENV_KEYS:
        if reference_env.get(key) is None:
            raise ValueError(f"{reference_name} missing env.{key}")
    for method_name, config in configs.items():
        env = config["env"]
        mismatches = {
            key: (reference_env.get(key), env.get(key))
            for key in ENV_KEYS
            if reference_env.get(key) != env.get(key)
        }
        if mismatches:
            raise ValueError(f"Environment mismatch for {method_name}: {mismatches}")
    if reference_env["num_defenders"] != 5 or reference_env["num_intruders"] != 5:
        raise ValueError("Validation requires strict 5v5 environment")


def load_extended_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = load_yaml(path)
    parent = data.get("extends")
    if not parent:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent.parent / parent if path.parent.name == "configs" else path.parent / parent
        if not parent_path.exists():
            parent_path = Path.cwd() / parent
    merged = load_extended_yaml(parent_path)
    return deep_merge(merged, {key: value for key, value in data.items() if key != "extends"})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
