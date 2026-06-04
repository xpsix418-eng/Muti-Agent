from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_METHODS = ["mappo", "sa_pmappo", "vanilla_gnn_mappo", "ipg_mappo", "ipg_with_assignment_gate"]

POLICY_BY_METHOD = {
    "mappo": "mappo",
    "dense_mappo": "mappo",
    "pi_mappo": "mappo",
    "sa_pmappo": "mappo",
    "vanilla_gnn_mappo": "ipga_mappo",
    "ipg_mappo": "ipga_mappo",
    "ipg_no_ita": "ipga_mappo",
    "ipg_no_graph": "ipga_mappo",
    "ipg_with_assignment_gate": "ipga_mappo",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results_dir", default="experiments/results/final_5v5")
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    for method in args.methods:
        visualize_method(method, args.seed, results_dir, args.max_steps)


def visualize_method(method: str, seed: int, results_dir: Path, max_steps: int) -> None:
    if method not in POLICY_BY_METHOD:
        raise ValueError(f"Unknown method '{method}'")
    seed_dir = results_dir / method / f"seed_{seed}"
    config = seed_dir / "resolved_config.yaml"
    checkpoint = seed_dir / "checkpoint_best_intercept.pt"
    if not checkpoint.exists():
        checkpoint = seed_dir / "checkpoint_latest.pt"
    if not config.exists() or not checkpoint.exists():
        print(f"[skip] missing config/checkpoint for {method} seed={seed}")
        return
    output_dir = seed_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/visualize_rollout.py",
        "--config",
        str(config),
        "--policy",
        POLICY_BY_METHOD[method],
        "--checkpoint",
        str(checkpoint),
        "--scenario",
        "Scenario5v5",
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=Path.cwd(), check=True)
    normalize_outputs(output_dir, POLICY_BY_METHOD[method])


def normalize_outputs(output_dir: Path, policy: str) -> None:
    trajectory = output_dir / "trajectory.png"
    if policy == "ipga_mappo" and (output_dir / "ipga_assignment.png").exists():
        source = output_dir / "ipga_assignment.png"
        shutil.copy2(source, output_dir / "interception_points.png")
        shutil.copy2(source, output_dir / "graph_edges.png")
        if (output_dir / "ipga_rollout.gif").exists():
            shutil.copy2(output_dir / "ipga_rollout.gif", output_dir / "rollout.gif")
    elif trajectory.exists():
        shutil.copy2(trajectory, output_dir / "interception_points.png")
        shutil.copy2(trajectory, output_dir / "graph_edges.png")
    print(f"saved {output_dir / 'trajectory.png'}")
    print(f"saved {output_dir / 'rollout.gif'}")
    print(f"saved {output_dir / 'interception_points.png'}")
    print(f"saved {output_dir / 'graph_edges.png'}")


if __name__ == "__main__":
    main()
