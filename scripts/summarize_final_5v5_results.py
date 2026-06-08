from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/results/final_5v5")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = collect_rows(results_dir)
    summary = summarize(rows)
    write_csv(results_dir / "final_summary_mean_std.csv", summary)
    write_markdown(results_dir / "final_summary_mean_std.md", summary)
    print(f"wrote {results_dir / 'final_summary_mean_std.csv'}")
    print(f"wrote {results_dir / 'final_summary_mean_std.md'}")


def collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        for seed_dir in sorted(method_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for mode, filename in [
                ("deterministic", "deterministic_eval_100ep.json"),
                ("stochastic", "stochastic_eval_100ep.json"),
            ]:
                path = seed_dir / filename
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as file:
                    metrics = json.load(file)
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        rows.append(
                            {
                                "method": method_dir.name,
                                "seed": seed,
                                "mode": mode,
                                "metric": metric,
                                "value": float(value),
                            }
                        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["method"], row["mode"], row["metric"])
        grouped.setdefault(key, []).append(float(row["value"]))
    output: list[dict[str, Any]] = []
    for (method, mode, metric), values in sorted(grouped.items()):
        avg = mean(values)
        std = stdev(values) if len(values) > 1 else 0.0
        output.append(
            {
                "method": method,
                "mode": mode,
                "metric": metric,
                "n": len(values),
                "mean": avg,
                "std": std,
                "mean_std": f"{avg:.6f} +/- {std:.6f}",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "mode", "metric", "n", "mean", "std", "mean_std"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "intercept_rate",
        "success_rate",
        "breach_rate",
        "collision_rate",
        "collision_episode_rate",
        "average_collisions_per_episode",
        "blocking_success_rate",
        "average_intercept_distance_to_asset",
        "average_intercept_time_to_asset",
        "mean_interception_time_advantage",
        "graph_attention_sparsity",
        "assignment_entropy",
    ]
    lookup = {(row["method"], row["mode"], row["metric"]): row["mean_std"] for row in rows}
    methods = sorted({row["method"] for row in rows})
    modes = sorted({row["mode"] for row in rows})
    lines = ["# Final 5v5 Mean +/- Std", ""]
    for mode in modes:
        lines.extend([f"## {mode}", ""])
        header = ["method", *preferred]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for method in methods:
            values = [lookup.get((method, mode, metric), "") for metric in preferred]
            lines.append("| " + " | ".join([method, *values]) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
