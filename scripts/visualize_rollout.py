from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

import _bootstrap  # noqa: F401
from algorithms.ipga_mappo.actor import IPGAActor
from algorithms.ipga_mappo.graph_encoder import InterceptionGraphEncoder
from algorithms.ipga_mappo.interception_graph import InterceptionGraphBuilder
from algorithms.ipga_mappo.soft_assignment_gate import SoftAssignmentGate
from algorithms.mappo.actor import MLPActor
from algorithms.mappo.utils import RunningMeanStd
from envs.config import config_from_mapping, load_env_config, load_yaml
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config, initialize_scenario_state


Policy = Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    parser.add_argument("--policy", choices=["mappo", "ipga_mappo"], default="ipga_mappo")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scenario", default="ScenarioB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    raw_config = load_extended_yaml(args.config)
    env_config = config_from_mapping(raw_config["env"]) if "env" in raw_config else load_env_config(args.config)
    env = CounterUAVEnv(apply_scenario_to_config(env_config, args.scenario))
    policy = build_policy(args.policy, env, args.checkpoint)
    rollout = collect_rollout(env, policy, args.scenario, seed=args.seed, max_steps=args.max_steps)
    output_dir = Path(args.output_dir or Path("experiments") / "results" / f"{args.policy}_{args.scenario}_rollout")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_png(env, rollout, output_dir / "trajectory.png")
    save_rollout_gif(env, rollout, output_dir / "rollout.gif")
    if args.policy == "ipga_mappo":
        save_rollout_gif(env, rollout, output_dir / "ipga_rollout.gif")
        save_ipga_assignment_png(env, rollout, output_dir / "ipga_assignment.png")
    print(f"saved_trajectory={output_dir / 'trajectory.png'}")
    print(f"saved_gif={output_dir / 'rollout.gif'}")
    if args.policy == "ipga_mappo":
        print(f"saved_ipga_gif={output_dir / 'ipga_rollout.gif'}")
        print(f"saved_ipga_assignment={output_dir / 'ipga_assignment.png'}")


class MAPPOVisualizationPolicy:
    def __init__(self, checkpoint_path: str, env: CounterUAVEnv):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        obs_dim = env.observation_spaces[env.defense_agents[0]].shape[0]
        hidden_dim = _infer_hidden_dim(checkpoint["actor"])
        self.actor = MLPActor(obs_dim, 2, hidden_dim)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()
        self.obs_rms = RunningMeanStd((obs_dim,))
        if "obs_rms" in checkpoint:
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])

    def act(self, observations: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
        obs = np.stack([observations[agent] for agent in agents]).astype(np.float32)
        norm_obs = self.obs_rms.normalize(obs)
        with torch.no_grad():
            actions = self.actor.deterministic(torch.as_tensor(norm_obs, dtype=torch.float32)).cpu().numpy()
        return np.clip(actions, -1.0, 1.0).astype(np.float32)


class IPGAVisualizationPolicy:
    def __init__(self, checkpoint_path: str, env: CounterUAVEnv):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        obs_dim = env.observation_spaces[env.defense_agents[0]].shape[0]
        graph_dim = int(cfg.get("graph_hidden_dim", 128))
        hidden_dim = int(cfg.get("hidden_dim", 128))
        self.builder = InterceptionGraphBuilder(
            env.config.world_size,
            env.config.defender_max_speed,
            env.config.intruder_max_speed,
            float(cfg.get("prediction_horizon", 5.0)),
            graph_type=str(cfg.get("graph_type", "ipg")),
            include_interception_point_nodes=bool(cfg.get("use_interception_point_nodes", True)),
            include_ita_edge_feature=bool(cfg.get("use_ita_features", True)),
        )
        sample = self.builder.build(env._infos()[env.defense_agents[0]])
        self.edge_index = torch.as_tensor(sample.edge_index, dtype=torch.long)
        self.graph_encoder = InterceptionGraphEncoder(
            sample.node_features.shape[1],
            sample.edge_features.shape[1],
            graph_dim,
            int(cfg.get("num_graph_layers", 2)),
            int(cfg.get("attention_heads", 4)),
        )
        self.assignment_gate = SoftAssignmentGate(graph_dim, sample.edge_features.shape[1], hidden_dim)
        self.actor = IPGAActor(obs_dim, graph_dim, 2, hidden_dim)
        self.graph_encoder.load_state_dict(checkpoint["graph_encoder"])
        self.assignment_gate.load_state_dict(checkpoint["assignment_gate"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.graph_encoder.eval()
        self.assignment_gate.eval()
        self.actor.eval()
        self.obs_rms = RunningMeanStd((obs_dim,))
        if "obs_rms" in checkpoint:
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])
        self.use_graph = bool(cfg.get("use_graph", True))
        self.use_interception_point_nodes = bool(cfg.get("use_interception_point_nodes", True))
        self.use_assignment_gate = bool(cfg.get("use_assignment_gate", True))
        self.use_ita_features = bool(cfg.get("use_ita_features", True))
        self.last_assignment_weights: np.ndarray | None = None
        self.last_attention: np.ndarray | None = None
        self.last_intercept_points: np.ndarray | None = None

    def act(self, observations: dict[str, np.ndarray], agents: list[str], info: dict[str, Any]) -> np.ndarray:
        obs = np.stack([observations[agent] for agent in agents]).astype(np.float32)
        norm_obs = self.obs_rms.normalize(obs)
        graph = self.builder.build(info)
        node_features = torch.as_tensor(graph.node_features[None, :, :], dtype=torch.float32)
        edge_features = torch.as_tensor(graph.edge_features[None, :, :], dtype=torch.float32)
        pair_features = torch.as_tensor(graph.pair_edge_features[None, :, :, :], dtype=torch.float32)
        if not self.use_ita_features:
            edge_features[..., 5] = 0.0
            pair_features[..., 5] = 0.0
        with torch.no_grad():
            node_embeddings, _, attention = self.graph_encoder(node_features, self.edge_index, edge_features)
            num_defenders = len(agents)
            num_intruders = graph.intruder_indices.shape[0]
            defender_embeddings = node_embeddings[:, :num_defenders]
            intruder_embeddings = node_embeddings[:, num_defenders : num_defenders + num_intruders]
            point_start = num_defenders + num_intruders + 1
            if self.use_interception_point_nodes:
                point_embeddings = node_embeddings[:, point_start : point_start + num_intruders]
            else:
                point_embeddings = torch.zeros_like(intruder_embeddings)
            if self.use_assignment_gate:
                weights, context = self.assignment_gate(defender_embeddings, intruder_embeddings, point_embeddings, pair_features)
            else:
                weights = torch.zeros(1, num_defenders, num_intruders)
                context = torch.zeros_like(defender_embeddings)
            if not self.use_graph:
                defender_embeddings = torch.zeros_like(defender_embeddings)
                context = torch.zeros_like(context)
            elif not self.use_assignment_gate:
                context = torch.zeros_like(context)
            actions = self.actor.deterministic(
                torch.as_tensor(norm_obs, dtype=torch.float32),
                defender_embeddings[0],
                context[0],
            ).cpu().numpy()
        self.last_assignment_weights = weights[0].cpu().numpy()
        self.last_attention = attention[0].cpu().numpy()
        self.last_intercept_points = graph.predicted_intercept_points.copy()
        return np.clip(actions, -1.0, 1.0).astype(np.float32)


def build_policy(policy_name: str, env: CounterUAVEnv, checkpoint_path: str | None = None) -> Policy:
    if policy_name == "mappo":
        if checkpoint_path is None:
            raise ValueError("--policy mappo requires --checkpoint PATH")
        return MAPPOVisualizationPolicy(checkpoint_path, env)
    if policy_name == "ipga_mappo":
        if checkpoint_path is None:
            raise ValueError("--policy ipga_mappo requires --checkpoint PATH")
        return IPGAVisualizationPolicy(checkpoint_path, env)
    raise ValueError(f"Unsupported policy: {policy_name}")


def collect_rollout(
    env: CounterUAVEnv,
    policy: Policy,
    scenario_name: str,
    seed: int,
    max_steps: int,
) -> dict[str, list[Any]]:
    rng = np.random.default_rng(seed)
    observations, info = env.reset(seed=seed)
    initialize_scenario_state(env, scenario_name, rng)
    observations = env._observations()
    info = env._infos()
    rollout: dict[str, list[Any]] = {
        "defenders": [],
        "intruders": [],
        "threat_scores": [],
        "topology": [],
        "predicted_intercept_points": [],
        "assignment_weights": [],
        "graph_attention": [],
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
        actions = policy_action(env, policy, agent_info, observations=observations)
        append_ipga_frame(rollout, policy, agent_info)
        next_obs, _, terminations, truncations, info = env.step(actions)
        agent_info = info[env.defense_agents[0]]
        observations = next_obs
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


def append_ipga_frame(rollout: dict[str, list[Any]], policy: Policy, info: dict[str, Any]) -> None:
    if isinstance(policy, IPGAVisualizationPolicy):
        rollout["predicted_intercept_points"].append(
            policy.last_intercept_points.copy()
            if policy.last_intercept_points is not None
            else info.get("predicted_intercept_points", np.empty((0, 2))).copy()
        )
        rollout["assignment_weights"].append(
            policy.last_assignment_weights.copy()
            if policy.last_assignment_weights is not None
            else np.empty((0, 0), dtype=np.float32)
        )
        rollout["graph_attention"].append(
            policy.last_attention.copy() if policy.last_attention is not None else np.empty(0, dtype=np.float32)
        )
    else:
        rollout["predicted_intercept_points"].append(info.get("predicted_intercept_points", np.empty((0, 2))).copy())
        rollout["assignment_weights"].append(np.empty((0, 0), dtype=np.float32))
        rollout["graph_attention"].append(np.empty(0, dtype=np.float32))


def policy_action(env: CounterUAVEnv, policy: Policy, info: dict[str, Any], observations: dict[str, np.ndarray] | None) -> np.ndarray:
    if isinstance(policy, MAPPOVisualizationPolicy):
        if observations is None:
            observations = env._observations()
        return policy.act(observations, env.defense_agents)
    if isinstance(policy, IPGAVisualizationPolicy):
        if observations is None:
            observations = env._observations()
        return policy.act(observations, env.defense_agents, info)
    raise TypeError(f"Unsupported policy instance: {type(policy).__name__}")


def _infer_hidden_dim(actor_state: dict[str, torch.Tensor]) -> int:
    first_weight = actor_state.get("net.0.weight")
    if first_weight is None:
        return 128
    return int(first_weight.shape[0])


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
    draw_ipga_overlays(rollout, ax, frame_idx, defenders, intruders)
    ax.scatter(defenders[:, 0], defenders[:, 1], c="tab:blue", s=35, label="defenders")
    scatter = ax.scatter(intruders[:, 0], intruders[:, 1], c=threats, cmap="YlOrRd", vmin=0.0, vmax=1.0, s=45, label="intruders")
    for idx, point in enumerate(intruders):
        ax.text(point[0], point[1], f"{threats[idx]:.2f}", fontsize=6, color="black")
    if frame_idx == 0:
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="threat")


def draw_ipga_overlays(
    rollout: dict[str, list[Any]],
    ax: plt.Axes,
    frame_idx: int,
    defenders: np.ndarray,
    intruders: np.ndarray,
) -> None:
    if frame_idx >= len(rollout.get("predicted_intercept_points", [])):
        return
    points = rollout["predicted_intercept_points"][frame_idx]
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], c="tab:purple", marker="^", s=35, alpha=0.8, label="predicted intercept")
    weights = rollout["assignment_weights"][frame_idx]
    if weights.size == 0:
        return
    segments = []
    widths = []
    for defender_idx in range(min(weights.shape[0], defenders.shape[0])):
        for intruder_idx in range(min(weights.shape[1], intruders.shape[0])):
            weight = float(weights[defender_idx, intruder_idx])
            if weight < 0.12:
                continue
            target = points[intruder_idx] if len(points) > intruder_idx else intruders[intruder_idx]
            segments.append([defenders[defender_idx], target])
            widths.append(0.5 + 3.0 * weight)
    if segments:
        ax.add_collection(LineCollection(segments, colors="tab:purple", linewidths=widths, alpha=0.35))


def save_ipga_assignment_png(env: CounterUAVEnv, rollout: dict[str, list[Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    configure_axes(env, ax)
    frame_idx = max(0, len(rollout["defenders"]) - 1)
    draw_frame(env, rollout, ax, frame_idx)
    ax.set_title("IPGA soft assignment and interception points")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_communication_edges(defender_positions: np.ndarray, topology: np.ndarray, ax: plt.Axes) -> None:
    segments = []
    rows, cols = np.where(np.triu(topology > 0.0, k=1))
    for row, col in zip(rows.tolist(), cols.tolist()):
        segments.append([defender_positions[row], defender_positions[col]])
    if segments:
        ax.add_collection(LineCollection(segments, colors="tab:cyan", linewidths=0.6, alpha=0.35))


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


if __name__ == "__main__":
    main()
