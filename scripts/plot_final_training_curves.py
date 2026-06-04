from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "intercept_rate",
    "success_rate",
    "breach_rate",
    "collision_rate",
    "blocking_success_rate",
    "entropy",
    "learning_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/results/final_5v5")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curves(results_dir)
    for metric in METRICS:
        plot_metric(curves, metric, figures_dir / f"{metric}.png")
    print(f"wrote figures to {figures_dir}")


def load_curves(results_dir: Path) -> dict[str, list[list[dict[str, float]]]]:
    curves: dict[str, list[list[dict[str, float]]]] = {}
    for method_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        if method_dir.name == "figures":
            continue
        for curve_path in sorted(method_dir.glob("seed_*/training_curve.csv")):
            rows = read_curve(curve_path)
            if rows:
                curves.setdefault(method_dir.name, []).append(rows)
    return curves


def read_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            if "step" in row:
                rows.append(row)
    return rows


def plot_metric(curves: dict[str, list[list[dict[str, float]]]], metric: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    plotted = False
    for method, method_curves in sorted(curves.items()):
        series = aligned_series(method_curves, metric)
        if series is None:
            continue
        steps, mean_values, std_values = series
        plt.plot(steps, mean_values, label=method)
        if len(method_curves) > 1:
            plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("environment steps")
    plt.ylabel(metric)
    plt.title(f"Final 5v5 {metric}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def aligned_series(curves: list[list[dict[str, float]]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    usable = [curve for curve in curves if any(metric in row for row in curve)]
    if not usable:
        return None
    max_len = max(len(curve) for curve in usable)
    if max_len == 0:
        return None
    values = np.full((len(usable), max_len), np.nan, dtype=np.float32)
    steps = np.full(max_len, np.nan, dtype=np.float32)
    for idx, curve in enumerate(usable):
        for row_idx, row in enumerate(curve):
            steps[row_idx] = row.get("step", steps[row_idx] if not np.isnan(steps[row_idx]) else float(row_idx))
            if metric in row:
                values[idx, row_idx] = row[metric]
    valid = ~np.all(np.isnan(values), axis=0)
    if not np.any(valid):
        return None
    values = values[:, valid]
    steps = steps[valid]
    mean_values = np.nanmean(values, axis=0)
    std_values = np.nanstd(values, axis=0)
    return steps, mean_values, std_values


if __name__ == "__main__":
    main()
