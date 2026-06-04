from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import _bootstrap  # noqa: F401
from algorithms.hungarian_assignment import HungarianAssignmentPolicy
from algorithms.ipga_mappo.actor import IPGAActor
from algorithms.ipga_mappo.critic import IPGACritic
from algorithms.ipga_mappo.graph_encoder import InterceptionGraphEncoder
from algorithms.ipga_mappo.interception_graph import InterceptionGraphBuilder
from algorithms.ipga_mappo.soft_assignment_gate import SoftAssignmentGate
from algorithms.ipga_mappo.utils import assignment_entropy, graph_attention_sparsity, mean_interception_time_advantage
from algorithms.mappo.actor import MLPActor
from algorithms.mappo.utils import RunningMeanStd
from algorithms.rule_based import RuleBasedPolicy
from envs.config import config_from_mapping, load_env_config, load_yaml
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config, initialize_scenario_state, scenario_metadata


Policy = Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    parser.add_argument("--policy", choices=["rule_based", "hungarian", "mappo", "ipga_mappo"], default="rule_based")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scenario", default="ScenarioB")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    raw_config = load_extended_yaml(args.config)
    base_config = config_from_mapping(raw_config["env"]) if "env" in raw_config else load_env_config(args.config)
    env = CounterUAVEnv(apply_scenario_to_config(base_config, args.scenario))
    policy = build_policy(args.policy, env, args.checkpoint)
    metadata = scenario_metadata(args.scenario)
    episode_rows = [
        evaluate_episode(
            env,
            policy,
            args.scenario,
            seed=args.seed + idx,
            metadata=metadata,
            deterministic=not args.stochastic,
        )
        for idx in range(args.episodes)
    ]
    metrics = summarize_metrics(episode_rows, env, metadata)
    experiment_name = args.experiment_name or f"{args.policy}_{args.scenario}"
    save_evaluation_results(experiment_name, metrics, episode_rows)
    print(json.dumps(metrics, indent=2))


def build_policy(policy_name: str, env: CounterUAVEnv, checkpoint_path: str | None = None) -> Policy:
    if policy_name == "mappo":
        if checkpoint_path is None:
            raise ValueError("--policy mappo requires --checkpoint PATH")
        return MAPPOEvaluationPolicy(checkpoint_path, env)
    if policy_name == "ipga_mappo":
        if checkpoint_path is None:
            raise ValueError("--policy ipga_mappo requires --checkpoint PATH")
        return IPGAEvaluationPolicy(checkpoint_path, env)
    if policy_name == "rule_based":
        return RuleBasedPolicy()
    if policy_name == "hungarian":
        return HungarianAssignmentPolicy()
    raise ValueError(f"Unsupported policy: {policy_name}")


class MAPPOEvaluationPolicy:
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

    def act(
        self,
        observations: dict[str, np.ndarray],
        agents: list[str],
        info: dict[str, Any],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict[str, float]]:
        del info
        obs = np.stack([observations[agent] for agent in agents]).astype(np.float32)
        norm_obs = self.obs_rms.normalize(obs)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32)
            if deterministic:
                actions = self.actor.deterministic(obs_tensor).cpu().numpy()
            else:
                actions, _, _ = self.actor.sample(obs_tensor)
                actions = actions.cpu().numpy()
        return np.clip(actions, -1.0, 1.0).astype(np.float32), {}


class IPGAEvaluationPolicy:
    def __init__(self, checkpoint_path: str, env: CounterUAVEnv):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        obs_dim = env.observation_spaces[env.defense_agents[0]].shape[0]
        state_dim = env.state_space.shape[0]
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
        self.critic = IPGACritic(state_dim, graph_dim, hidden_dim)
        self.graph_encoder.load_state_dict(checkpoint["graph_encoder"])
        self.assignment_gate.load_state_dict(checkpoint["assignment_gate"])
        self.actor.load_state_dict(checkpoint["actor"])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
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

    def act(
        self,
        observations: dict[str, np.ndarray],
        agents: list[str],
        info: dict[str, Any],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict[str, float]]:
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
            obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32)
            if deterministic:
                actions = self.actor.deterministic(obs_tensor, defender_embeddings[0], context[0]).cpu().numpy()
            else:
                actions, _, _ = self.actor.sample(obs_tensor, defender_embeddings[0], context[0])
                actions = actions.cpu().numpy()
        diagnostics = {
            "assignment_entropy": assignment_entropy(weights.cpu().numpy()),
            "mean_interception_time_advantage": mean_interception_time_advantage(graph.interception_time_advantage),
            "graph_attention_sparsity": graph_attention_sparsity(attention.cpu().numpy()),
        }
        return np.clip(actions, -1.0, 1.0).astype(np.float32), diagnostics


def evaluate_episode(
    env: CounterUAVEnv,
    policy: Policy,
    scenario_name: str,
    seed: int,
    metadata: dict[str, float | int | str],
    deterministic: bool = True,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observations, info = env.reset(seed=seed)
    initialize_scenario_state(env, scenario_name, rng)
    observations = env._observations()
    info = env._infos()
    initial_info = info[env.defense_agents[0]]
    high_threat_mask = initial_info["threat_scores"] >= 0.7
    previous_intercepted = np.zeros(env.config.num_intruders, dtype=bool)
    previous_breached = np.zeros(env.config.num_intruders, dtype=bool)
    intercept_times = np.full(env.config.num_intruders, np.nan, dtype=np.float32)
    intercept_distances_to_asset: list[float] = []
    intercept_time_to_asset: list[float] = []

    total_energy = 0.0
    total_collisions = 0
    total_communication_links = 0.0
    total_defender_distance_to_asset = 0.0
    total_blocking_flags = 0.0
    total_blocking_denominator = 0.0
    total_assignment_entropy = 0.0
    total_ita = 0.0
    total_attention_sparsity = 0.0
    steps = 0

    for _ in range(env.config.max_steps):
        agent_info = info[env.defense_agents[0]]
        actions, diagnostics = policy_action(env, policy, agent_info, observations, deterministic=deterministic)
        observations, _, terminations, truncations, info = env.step(actions)
        agent_info = info[env.defense_agents[0]]
        intercepted = np.asarray(agent_info["intercepted"], dtype=bool)
        breached = np.asarray(agent_info["breached"], dtype=bool)
        newly_intercepted = intercepted & ~previous_intercepted
        newly_breached = breached & ~previous_breached
        intercept_times[newly_intercepted] = env.step_count
        if np.any(newly_intercepted):
            intercepted_positions = agent_info["intruder_positions"][newly_intercepted]
            intercepted_velocities = agent_info["intruder_velocities"][newly_intercepted]
            distances_to_asset = np.linalg.norm(intercepted_positions - env.protected_asset[None, :], axis=1)
            speeds = np.linalg.norm(intercepted_velocities, axis=1)
            speeds = np.where(speeds < 1e-6, env.config.intruder_max_speed, speeds)
            intercept_distances_to_asset.extend(distances_to_asset.tolist())
            intercept_time_to_asset.extend((distances_to_asset / speeds).tolist())
        previous_intercepted = intercepted
        previous_breached |= newly_breached

        total_energy += float(np.mean(np.linalg.norm(actions, axis=1)))
        total_collisions += len(agent_info["collision_events"])
        total_communication_links += communication_cost(agent_info["communication_topology"], metadata)
        total_defender_distance_to_asset += float(
            np.mean(np.linalg.norm(agent_info["defender_positions"] - env.protected_asset[None, :], axis=1))
        )
        total_blocking_flags += float(np.sum(agent_info.get("blocking_flags", np.zeros(env.config.num_defenders))))
        total_blocking_denominator += float(env.config.num_defenders)
        total_assignment_entropy += diagnostics.get("assignment_entropy", 0.0)
        total_ita += diagnostics.get("mean_interception_time_advantage", 0.0)
        total_attention_sparsity += diagnostics.get("graph_attention_sparsity", 0.0)
        steps += 1
        if terminations["__all__"] or truncations["__all__"]:
            break

    final_info = info[env.defense_agents[0]]
    intercepted = np.asarray(final_info["intercepted"], dtype=bool)
    breached = np.asarray(final_info["breached"], dtype=bool)
    high_threat_denominator = max(int(np.sum(high_threat_mask)), 1)
    valid_intercept_times = intercept_times[~np.isnan(intercept_times)]
    success = float(np.all(intercepted) and not np.any(breached))
    average_collisions_per_step = float(total_collisions / max(steps, 1))
    return {
        "intercept_rate": float(np.mean(intercepted)),
        "breach_rate": float(np.mean(breached)),
        "high_threat_intercept_rate": float(np.sum(intercepted & high_threat_mask) / high_threat_denominator),
        "average_intercept_time": float(np.mean(valid_intercept_times)) if len(valid_intercept_times) else float(env.config.max_steps),
        "average_energy_cost": float(total_energy / max(steps, 1)),
        "collision_rate": average_collisions_per_step,
        "average_collisions_per_step": average_collisions_per_step,
        "average_collisions_per_episode": float(total_collisions),
        "collision_episode_rate": float(total_collisions > 0),
        "communication_cost": float(total_communication_links / max(steps, 1)),
        "success_rate": success,
        "average_defender_distance_to_asset": float(total_defender_distance_to_asset / max(steps, 1)),
        "average_intercept_distance_to_asset": float(np.mean(intercept_distances_to_asset)) if intercept_distances_to_asset else float("nan"),
        "average_intercept_time_to_asset": float(np.mean(intercept_time_to_asset)) if intercept_time_to_asset else float("nan"),
        "blocking_success_rate": float(total_blocking_flags / max(total_blocking_denominator, 1.0)),
        "assignment_entropy": float(total_assignment_entropy / max(steps, 1)),
        "mean_interception_time_advantage": float(total_ita / max(steps, 1)),
        "graph_attention_sparsity": float(total_attention_sparsity / max(steps, 1)),
        "steps": float(steps),
    }


def policy_action(
    env: CounterUAVEnv,
    policy: Policy,
    info: dict[str, Any],
    observations: dict[str, np.ndarray],
    deterministic: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    if isinstance(policy, (MAPPOEvaluationPolicy, IPGAEvaluationPolicy)):
        return policy.act(observations, env.defense_agents, info, deterministic=deterministic)
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
        return policy.act(intruder_velocities=info["intruder_velocities"], **common), {}
    return policy.act(**common), {}


def communication_cost(topology: np.ndarray, metadata: dict[str, float | int | str]) -> float:
    directed_links = float(np.sum(topology))
    undirected_links = directed_links / 2.0
    packet_loss = float(metadata.get("packet_loss", 0.0))
    delay = float(metadata.get("communication_delay", 0))
    return undirected_links * (1.0 + packet_loss + 0.1 * delay)


def summarize_metrics(
    episode_rows: list[dict[str, float]],
    env: CounterUAVEnv,
    metadata: dict[str, float | int | str],
) -> dict[str, float]:
    metrics = {
        key: float(np.mean([row[key] for row in episode_rows]))
        for key in episode_rows[0]
        if key != "steps"
    }
    metrics["scalability_score"] = scalability_score(metrics["intercept_rate"], env.config.num_defenders, env.config.num_intruders)
    metrics["robustness_score"] = robustness_score(metrics["success_rate"], metrics["breach_rate"], metadata)
    return metrics


def scalability_score(intercept_rate: float, num_defenders: int, num_intruders: int) -> float:
    baseline_size = 24.0
    scale_penalty = np.sqrt(baseline_size / max(num_defenders + num_intruders, 1))
    return float(np.clip(intercept_rate * scale_penalty, 0.0, 1.0))


def robustness_score(success_rate: float, breach_rate: float, metadata: dict[str, float | int | str]) -> float:
    packet_loss = float(metadata.get("packet_loss", 0.0))
    delay = float(metadata.get("communication_delay", 0))
    noise = float(metadata.get("observation_noise", 0.0))
    degradation = 1.0 + packet_loss + 0.1 * delay + 0.01 * noise
    return float(np.clip((success_rate * (1.0 - breach_rate)) / degradation, 0.0, 1.0))


def save_evaluation_results(
    experiment_name: str,
    metrics: dict[str, float],
    episode_rows: list[dict[str, float]],
) -> None:
    output_dir = Path("experiments") / "results" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (output_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", *episode_rows[0].keys()])
        writer.writeheader()
        for idx, row in enumerate(episode_rows):
            writer.writerow({"episode": idx, **row})


def _infer_hidden_dim(actor_state: dict[str, torch.Tensor]) -> int:
    first_weight = actor_state.get("net.0.weight")
    if first_weight is None:
        return 128
    return int(first_weight.shape[0])


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
