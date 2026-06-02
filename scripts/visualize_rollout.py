from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

import _bootstrap  # noqa: F401
from algorithms.hungarian_assignment import HungarianAssignmentPolicy
from algorithms.rule_based import RuleBasedPolicy
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config, initialize_scenario_state


Policy = RuleBasedPolicy | HungarianAssignmentPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    parser.add_argument("--policy", choices=["rule_based", "hungarian"], default="rule_based")
    parser.add_argument("--scenario", default="ScenarioB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    env = CounterUAVEnv(apply_scenario_to_config(load_env_config(args.config), args.scenario))
    policy = build_policy(args.policy)
    rollout = collect_rollout(env, policy, args.scenario, seed=args.seed, max_steps=args.max_steps)
    output_dir = Path(args.output_dir or Path("experiments") / "results" / f"{args.policy}_{args.scenario}_rollout")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_png(env, rollout, output_dir / "trajectory.png")
    save_rollout_gif(env, rollout, output_dir / "rollout.gif")
    print(f"saved_trajectory={output_dir / 'trajectory.png'}")
    print(f"saved_gif={output_dir / 'rollout.gif'}")


def build_policy(policy_name: str) -> Policy:
    if policy_name == "rule_based":
        return RuleBasedPolicy()
    if policy_name == "hungarian":
        return HungarianAssignmentPolicy()
    raise ValueError(f"Unsupported policy: {policy_name}")


def collect_rollout(
    env: CounterUAVEnv,
    policy: Policy,
    scenario_name: str,
    seed: int,
    max_steps: int,
) -> dict[str, list[Any]]:
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    initialize_scenario_state(env, scenario_name, rng)
    info = env._infos()
    rollout: dict[str, list[Any]] = {
        "defenders": [],
        "intruders": [],
        "threat_scores": [],
        "topology": [],
        "intercepts": [],
        "breaches": [],
    }
    previous_intercepted = np.zeros(env.config.num_intruders, dtype=bool)
    previous_breached = np.zeros(env.config.num_intruders, dtype=bool)

    for _ in range(min(max_steps, env.config.max_steps)):
        agent_info = info[env.defense_agents[0]]
        previous_intercepted, previous_breached = append_frame(
            rollout, agent_info, previous_intercepted, previous_breached
        )
        actions = policy_action(env, policy, agent_info)
        _, _, terminations, truncations, info = env.step(actions)
        agent_info = info[env.defense_agents[0]]
        if terminations["__all__"] or truncations["__all__"]:
            append_frame(rollout, agent_info, previous_intercepted, previous_breached)
            break
    return rollout


def append_frame(
    rollout: dict[str, list[Any]],
    info: dict[str, Any],
    previous_intercepted: np.ndarray,
    previous_breached: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    intercepted = np.asarray(info["intercepted"], dtype=bool)
    breached = np.asarray(info["breached"], dtype=bool)
    rollout["defenders"].append(info["defender_positions"].copy())
    rollout["intruders"].append(info["intruder_positions"].copy())
    rollout["threat_scores"].append(info["threat_scores"].copy())
    rollout["topology"].append(info["communication_topology"].copy())
    rollout["intercepts"].append(info["intruder_positions"][intercepted & ~previous_intercepted].copy())
    rollout["breaches"].append(info["intruder_positions"][breached & ~previous_breached].copy())
    return intercepted, breached


def policy_action(env: CounterUAVEnv, policy: Policy, info: dict[str, Any]) -> np.ndarray:
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


def save_trajectory_png(env: CounterUAVEnv, rollout: dict[str, list[Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    configure_axes(env, ax)
    draw_static_rollout(env, rollout, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rollout_gif(env: CounterUAVEnv, rollout: dict[str, list[Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame_idx: int) -> list[Any]:
        ax.clear()
        configure_axes(env, ax)
        draw_frame(env, rollout, ax, frame_idx)
        return []

    animation = FuncAnimation(fig, update, frames=len(rollout["defenders"]), interval=120, blit=False)
    animation.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def configure_axes(env: CounterUAVEnv, ax: plt.Axes) -> None:
    ax.set_xlim(0, env.config.world_size)
    ax.set_ylim(0, env.config.world_size)
    ax.set_aspect("equal")
    ax.add_patch(Circle(env.protected_asset, env.config.protected_radius, fill=False, color="tab:green", linewidth=2))
    ax.scatter([env.protected_asset[0]], [env.protected_asset[1]], c="tab:green", marker="*", s=180, label="protected asset")
    ax.set_title("Counter-UAV MARL rollout")


def draw_static_rollout(env: CounterUAVEnv, rollout: dict[str, list[Any]], ax: plt.Axes) -> None:
    defenders = np.asarray(rollout["defenders"])
    intruders = np.asarray(rollout["intruders"])
    for idx in range(defenders.shape[1]):
        ax.plot(defenders[:, idx, 0], defenders[:, idx, 1], color="tab:blue", alpha=0.65, linewidth=1.2)
    for idx in range(intruders.shape[1]):
        ax.plot(intruders[:, idx, 0], intruders[:, idx, 1], color="tab:red", alpha=0.45, linewidth=1.0)
    draw_frame(env, rollout, ax, len(rollout["defenders"]) - 1)
    intercepts = np.concatenate([points for points in rollout["intercepts"] if len(points)], axis=0) if any(len(p) for p in rollout["intercepts"]) else np.empty((0, 2))
    breaches = np.concatenate([points for points in rollout["breaches"] if len(points)], axis=0) if any(len(p) for p in rollout["breaches"]) else np.empty((0, 2))
    if len(intercepts):
        ax.scatter(intercepts[:, 0], intercepts[:, 1], c="gold", edgecolors="black", marker="X", s=80, label="intercepts")
    if len(breaches):
        ax.scatter(breaches[:, 0], breaches[:, 1], c="black", marker="x", s=90, label="breaches")
    ax.legend(loc="upper right", fontsize=8)


def draw_frame(env: CounterUAVEnv, rollout: dict[str, list[Any]], ax: plt.Axes, frame_idx: int) -> None:
    defenders = rollout["defenders"][frame_idx]
    intruders = rollout["intruders"][frame_idx]
    threats = rollout["threat_scores"][frame_idx]
    topology = rollout["topology"][frame_idx]
    draw_communication_edges(defenders, topology, ax)
    ax.scatter(defenders[:, 0], defenders[:, 1], c="tab:blue", s=35, label="defenders")
    scatter = ax.scatter(intruders[:, 0], intruders[:, 1], c=threats, cmap="YlOrRd", vmin=0.0, vmax=1.0, s=45, label="intruders")
    for idx, point in enumerate(intruders):
        ax.text(point[0], point[1], f"{threats[idx]:.2f}", fontsize=6, color="black")
    if frame_idx == 0:
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="threat")


def draw_communication_edges(defender_positions: np.ndarray, topology: np.ndarray, ax: plt.Axes) -> None:
    segments = []
    rows, cols = np.where(np.triu(topology > 0.0, k=1))
    for row, col in zip(rows.tolist(), cols.tolist()):
        segments.append([defender_positions[row], defender_positions[col]])
    if segments:
        ax.add_collection(LineCollection(segments, colors="tab:cyan", linewidths=0.6, alpha=0.35))


if __name__ == "__main__":
    main()
