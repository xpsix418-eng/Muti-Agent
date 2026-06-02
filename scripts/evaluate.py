from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import _bootstrap  # noqa: F401
from algorithms.hungarian_assignment import HungarianAssignmentPolicy
from algorithms.mappo.actor import MLPActor
from algorithms.mappo.utils import RunningMeanStd
from algorithms.rule_based import RuleBasedPolicy
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVConfig, CounterUAVEnv
from envs.scenarios import apply_scenario_to_config, initialize_scenario_state, scenario_metadata


Policy = Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    parser.add_argument("--policy", choices=["rule_based", "hungarian", "mappo"], default="rule_based")
    parser.add_argument("--scenario", default="ScenarioB")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--robustness_test", action="store_true")
    args = parser.parse_args()

    base_config = load_env_config(args.config)
    scenario_config = apply_scenario_to_config(base_config, args.scenario)
    if args.robustness_test:
        run_robustness_test(args, scenario_config)
        return
    env = CounterUAVEnv(scenario_config)
    policy = build_policy(args.policy, env, args.checkpoint)
    metadata = scenario_metadata(args.scenario)
    episode_rows = [
        evaluate_episode(env, policy, args.scenario, seed=args.seed + idx, metadata=metadata)
        for idx in range(args.episodes)
    ]
    metrics = summarize_metrics(episode_rows, env, metadata)
    experiment_name = args.experiment_name or f"{args.policy}_{args.scenario}"
    save_evaluation_results(experiment_name, metrics, episode_rows)
    print(json.dumps(metrics, indent=2))


class MAPPOCheckpointPolicy:
    def __init__(self, checkpoint_path: str, env: CounterUAVEnv):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        obs_dim = env.observation_spaces[env.defense_agents[0]].shape[0]
        action_dim = 2
        hidden_dim = _infer_hidden_dim(checkpoint["actor"])
        self.actor = MLPActor(obs_dim, action_dim, hidden_dim)
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


def build_policy(policy_name: str, env: CounterUAVEnv, checkpoint_path: str | None = None) -> Policy:
    if checkpoint_path:
        return MAPPOCheckpointPolicy(checkpoint_path, env)
    if policy_name == "rule_based":
        return RuleBasedPolicy()
    if policy_name == "hungarian":
        return HungarianAssignmentPolicy()
    if policy_name == "mappo":
        raise ValueError("--policy mappo requires --checkpoint PATH")
    raise ValueError(f"Unsupported policy: {policy_name}")


def evaluate_episode(
    env: CounterUAVEnv,
    policy: Policy,
    scenario_name: str,
    seed: int,
    metadata: dict[str, float | int | str],
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observations, info = env.reset(seed=seed)
    initialize_scenario_state(env, scenario_name, rng)
    info = env._infos()
    initial_info = info[env.defense_agents[0]]
    high_threat_mask = initial_info["threat_scores"] >= 0.7
    previous_intercepted = np.zeros(env.config.num_intruders, dtype=bool)
    previous_breached = np.zeros(env.config.num_intruders, dtype=bool)
    intercept_times = np.full(env.config.num_intruders, np.nan, dtype=np.float32)

    total_energy = 0.0
    total_collisions = 0
    total_communication_links = 0.0
    steps = 0

    for _ in range(env.config.max_steps):
        agent_info = info[env.defense_agents[0]]
        actions = policy_action(env, policy, agent_info, observations)
        observations, _, terminations, truncations, info = env.step(actions)
        agent_info = info[env.defense_agents[0]]
        intercepted = np.asarray(agent_info["intercepted"], dtype=bool)
        breached = np.asarray(agent_info["breached"], dtype=bool)
        newly_intercepted = intercepted & ~previous_intercepted
        newly_breached = breached & ~previous_breached
        intercept_times[newly_intercepted] = env.step_count
        previous_intercepted = intercepted
        previous_breached |= newly_breached

        total_energy += float(np.mean(np.linalg.norm(actions, axis=1)))
        total_collisions += len(agent_info["collision_events"])
        total_communication_links += communication_cost(agent_info["communication_topology"], metadata)
        steps += 1
        if terminations["__all__"] or truncations["__all__"]:
            break

    final_info = info[env.defense_agents[0]]
    intercepted = np.asarray(final_info["intercepted"], dtype=bool)
    breached = np.asarray(final_info["breached"], dtype=bool)
    high_threat_denominator = max(int(np.sum(high_threat_mask)), 1)
    valid_intercept_times = intercept_times[~np.isnan(intercept_times)]
    success = float(np.all(intercepted) and not np.any(breached))
    return {
        "intercept_rate": float(np.mean(intercepted)),
        "breach_rate": float(np.mean(breached)),
        "high_threat_intercept_rate": float(np.sum(intercepted & high_threat_mask) / high_threat_denominator),
        "average_intercept_time": float(np.mean(valid_intercept_times)) if len(valid_intercept_times) else float(env.config.max_steps),
        "average_energy_cost": float(total_energy / max(steps, 1)),
        "collision_rate": float(total_collisions / max(steps, 1)),
        "communication_cost": float(total_communication_links / max(steps, 1)),
        "success_rate": success,
        "steps": float(steps),
    }


def policy_action(env: CounterUAVEnv, policy: Policy, info: dict[str, Any], observations: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(policy, MAPPOCheckpointPolicy):
        return policy.act(observations, env.defense_agents)
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


def run_robustness_test(args: argparse.Namespace, scenario_config: CounterUAVConfig) -> None:
    if not args.checkpoint:
        raise ValueError("--robustness_test requires --checkpoint PATH")
    rows = []
    for condition_name, config in robustness_conditions(scenario_config).items():
        env = CounterUAVEnv(config)
        policy = build_policy("mappo", env, args.checkpoint)
        metadata = scenario_metadata(args.scenario)
        metadata = {
            **metadata,
            "packet_loss": config.packet_loss_prob,
            "communication_delay": config.comm_delay_steps,
            "observation_noise": config.noisy_observation_std,
        }
        episode_rows = [
            evaluate_episode(env, policy, args.scenario, seed=args.seed + idx, metadata=metadata)
            for idx in range(args.episodes)
        ]
        metrics = summarize_metrics(episode_rows, env, metadata)
        rows.append({"condition": condition_name, **metrics})
    experiment_name = args.experiment_name or f"robustness_{args.scenario}"
    save_robustness_results(experiment_name, rows)
    summary = {
        "robustness_score": float(np.mean([row["robustness_score"] for row in rows])),
        "conditions": rows,
    }
    print(json.dumps(summary, indent=2))


def robustness_conditions(base: CounterUAVConfig) -> dict[str, CounterUAVConfig]:
    return {
        "no_comm_limit": replace(
            base,
            comm_radius=base.world_size * 2.0,
            packet_loss_prob=0.0,
            comm_delay_steps=0,
            agent_dropout_prob=0.0,
            noisy_observation_std=0.0,
            partial_observation=False,
        ),
        "limited_comm_radius": replace(base, comm_radius=min(base.comm_radius, 120.0), partial_observation=True),
        "packet_loss_10": replace(base, packet_loss_prob=0.10, partial_observation=True),
        "packet_loss_30": replace(base, packet_loss_prob=0.30, partial_observation=True),
        "delay_2_steps": replace(base, comm_delay_steps=2, partial_observation=True),
        "agent_dropout_20": replace(base, agent_dropout_prob=0.20, partial_observation=True),
        "noisy_observation": replace(base, noisy_observation_std=5.0, partial_observation=True),
        "combined_disturbance": replace(
            base,
            comm_radius=min(base.comm_radius, 120.0),
            packet_loss_prob=0.30,
            comm_delay_steps=2,
            agent_dropout_prob=0.20,
            noisy_observation_std=5.0,
            partial_observation=True,
        ),
    }


def save_robustness_results(experiment_name: str, rows: list[dict[str, float | str]]) -> None:
    output_dir = Path("experiments") / "results" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "robustness_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "robustness_score": float(np.mean([float(row["robustness_score"]) for row in rows])),
                "conditions": rows,
            },
            file,
            indent=2,
        )


def _infer_hidden_dim(actor_state: dict[str, torch.Tensor]) -> int:
    first_weight = actor_state.get("net.0.weight")
    if first_weight is None:
        return 128
    return int(first_weight.shape[0])


if __name__ == "__main__":
    main()
