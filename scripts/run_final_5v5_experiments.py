from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

import _bootstrap  # noqa: F401
from envs.config import load_yaml


DEFAULT_METHODS = [
    "mappo",
    "dense_mappo",
    "pi_mappo",
    "sa_pmappo",
    "vanilla_gnn_mappo",
    "ipg_mappo",
    "ipg_no_ita",
    "ipg_no_graph",
    "ipg_with_assignment_gate",
]

METHODS = {
    "mappo": ("configs/final_5v5_mappo.yaml", "scripts/train_mappo.py", "mappo"),
    "dense_mappo": ("configs/final_5v5_dense_mappo.yaml", "scripts/train_mappo.py", "mappo"),
    "pi_mappo": ("configs/final_5v5_pi_mappo.yaml", "scripts/train_mappo.py", "mappo"),
    "sa_pmappo": ("configs/final_5v5_sa_pmappo.yaml", "scripts/train_mappo.py", "mappo"),
    "vanilla_gnn_mappo": ("configs/final_5v5_vanilla_gnn_mappo.yaml", "scripts/train_ipga_mappo.py", "ipga_mappo"),
    "ipg_mappo": ("configs/final_5v5_ipg_mappo.yaml", "scripts/train_ipga_mappo.py", "ipga_mappo"),
    "ipg_no_ita": ("configs/final_5v5_ipg_no_ita.yaml", "scripts/train_ipga_mappo.py", "ipga_mappo"),
    "ipg_no_graph": ("configs/final_5v5_ipg_no_graph.yaml", "scripts/train_ipga_mappo.py", "ipga_mappo"),
    "ipg_with_assignment_gate": ("configs/final_5v5_ipg_with_assignment_gate.yaml", "scripts/train_ipga_mappo.py", "ipga_mappo"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--total_env_steps", type=int, default=5_000_000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--results_dir", default="experiments/results/final_5v5")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    for method in args.methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {', '.join(METHODS)}")
        for seed in args.seeds:
            run_one(
                method=method,
                seed=seed,
                total_env_steps=args.total_env_steps,
                eval_episodes=args.eval_episodes,
                results_dir=results_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                device=args.device,
                torch_threads=args.torch_threads,
            )


def run_one(
    method: str,
    seed: int,
    total_env_steps: int,
    eval_episodes: int,
    results_dir: Path,
    skip_existing: bool,
    dry_run: bool,
    device: str | None,
    torch_threads: int,
) -> None:
    config_path, train_script, eval_policy = METHODS[method]
    seed_dir = results_dir / method / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = seed_dir / "resolved_config.yaml"
    latest_out = seed_dir / "checkpoint_latest.pt"
    det_out = seed_dir / "deterministic_eval_100ep.json"
    stoch_out = seed_dir / "stochastic_eval_100ep.json"
    if skip_existing and latest_out.exists() and det_out.exists() and stoch_out.exists():
        print(f"[skip] {method} seed={seed}")
        return

    config = load_extended_yaml(config_path)
    config = apply_overrides(config, method, seed, total_env_steps, seed_dir)
    write_yaml(resolved_path, config)
    train_cmd = [
        sys.executable,
        train_script,
        "--config",
        str(resolved_path),
        "--total-steps",
        str(total_env_steps),
    ]
    if train_script.endswith("train_ipga_mappo.py"):
        if device:
            train_cmd.extend(["--device", device])
        train_cmd.extend(["--torch-threads", str(torch_threads)])

    print(f"[train] {method} seed={seed} steps={total_env_steps}")
    print(" ".join(train_cmd))
    if dry_run:
        return
    run_logged(train_cmd, seed_dir / "train_stdout.log")

    source_checkpoint = find_latest_checkpoint(seed_dir)
    shutil.copy2(source_checkpoint, latest_out)
    shutil.copy2(source_checkpoint, seed_dir / "checkpoint_best_success.pt")
    shutil.copy2(source_checkpoint, seed_dir / "checkpoint_best_intercept.pt")
    export_training_curve(seed_dir / "logs", seed_dir / "training_curve.csv")

    eval_log = seed_dir / "eval_stdout.log"
    det_metrics = run_eval(resolved_path, eval_policy, latest_out, eval_episodes, seed, False, eval_log)
    stoch_metrics = run_eval(resolved_path, eval_policy, latest_out, eval_episodes, seed, True, eval_log)
    write_json(det_out, det_metrics)
    write_json(stoch_out, stoch_metrics)


def apply_overrides(config: dict[str, Any], method: str, seed: int, total_env_steps: int, seed_dir: Path) -> dict[str, Any]:
    config = deep_merge({}, config)
    training = config.setdefault("training", {})
    training["seed"] = seed
    training["total_env_steps"] = total_env_steps
    training["total_steps"] = total_env_steps
    training.setdefault("rollout_steps", 512)
    training.setdefault("rollout_length", training["rollout_steps"])
    training.setdefault("num_envs", 8)
    training.setdefault("ppo_epoch", 5)
    training.setdefault("epochs", training["ppo_epoch"])
    training.setdefault("mini_batch_size", 2048)
    training.setdefault("batch_size", training["mini_batch_size"])
    training.setdefault("learning_rate", 3e-4)
    training.setdefault("min_learning_rate", 3e-5)
    training.setdefault("use_linear_lr_decay", True)
    training.setdefault("lr_schedule", True)
    config.setdefault("env", {})["seed"] = seed
    config["env"]["num_defenders"] = 5
    config["env"]["num_intruders"] = 5
    config["scenario"] = "Scenario5v5"
    config.setdefault("logging", {})["log_dir"] = str(seed_dir / "logs")
    config["final_method"] = method
    return config


def run_eval(
    config_path: Path,
    policy: str,
    checkpoint: Path,
    episodes: int,
    seed: int,
    stochastic: bool,
    log_path: Path,
) -> dict[str, Any]:
    mode = "stochastic" if stochastic else "deterministic"
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        str(config_path),
        "--policy",
        policy,
        "--checkpoint",
        str(checkpoint),
        "--scenario",
        "Scenario5v5",
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--experiment-name",
        f"_final_5v5_tmp/{config_path.parent.parent.name}/{config_path.parent.name}/{mode}",
    ]
    if stochastic:
        cmd.append("--stochastic")
    output = run_logged(cmd, log_path, append=True)
    return parse_json_from_output(output)


def run_logged(cmd: list[str], log_path: Path, append: bool = False) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    mode = "a" if append else "w"
    collected: list[str] = []
    with log_path.open(mode, encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n")
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            collected.append(line)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, "".join(collected))
    return "".join(collected)


def find_latest_checkpoint(seed_dir: Path) -> Path:
    candidates = list((seed_dir / "logs").rglob("latest.pt"))
    if not candidates:
        raise FileNotFoundError(f"No latest.pt checkpoint found under {seed_dir / 'logs'}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def export_training_curve(log_root: Path, output_path: Path) -> None:
    rows = read_tensorboard_scalars(log_root)
    if not rows:
        rows = read_stdout_curve(log_root.parent / "train_stdout.log")
    wanted = [
        "step",
        "episode_reward",
        "intercept_rate",
        "success_rate",
        "breach_rate",
        "collision_rate",
        "blocking_success_rate",
        "entropy",
        "learning_rate",
        "policy_loss",
        "value_loss",
        "assignment_loss",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=wanted)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in wanted})


def read_tensorboard_scalars(log_root: Path) -> list[dict[str, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []
    merged: dict[int, dict[str, float]] = {}
    for event_file in log_root.rglob("events.out.tfevents.*"):
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            name = tag.split("/")[-1]
            for event in accumulator.Scalars(tag):
                row = merged.setdefault(int(event.step), {"step": float(event.step)})
                row[name] = float(event.value)
    return [merged[step] for step in sorted(merged)]


def read_stdout_curve(log_path: Path) -> list[dict[str, float]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("update="):
            continue
        row: dict[str, float] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "update":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
        if row:
            rows.append(row)
    return rows


def parse_json_from_output(output: str) -> dict[str, Any]:
    start = output.rfind("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Could not parse metrics JSON from output:\n{output[-1000:]}")
    return json.loads(output[start : end + 1])


def load_extended_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = load_yaml(path)
    parent = data.get("extends")
    if not parent:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent.parent / parent_path if path.parent.name == "configs" else path.parent / parent_path
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
        yaml.safe_dump(data, file, sort_keys=False)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    main()
