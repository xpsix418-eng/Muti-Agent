from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from algorithms.gnn_mappo.trainer import GNNMAPPOConfig, GNNMAPPOTrainer
from envs.config import load_env_config, load_yaml, resolve_env_config_path
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_gnn_mappo.yaml")
    parser.add_argument("--scenario", default="ScenarioA")
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    env = CounterUAVEnv(apply_scenario_to_config(load_env_config(resolve_env_config_path(args.config)), args.scenario))
    training = raw.get("training", {})
    graph = raw.get("graph", {})
    logging = raw.get("logging", {})
    cfg = GNNMAPPOConfig(
        total_steps=int(args.total_steps or training.get("total_steps", 100000)),
        rollout_length=int(training.get("rollout_length", 128)),
        gamma=float(training.get("gamma", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        clip_ratio=float(training.get("clip_ratio", 0.2)),
        learning_rate=float(training.get("learning_rate", 3e-4)),
        message_passing_steps=int(graph.get("message_passing_steps", raw.get("model", {}).get("message_passing_steps", 2))),
        log_dir=f"{logging.get('log_dir', 'experiments/results/gnn_mappo')}/{args.scenario}",
        checkpoint_dir=f"{logging.get('log_dir', 'experiments/results/gnn_mappo')}/{args.scenario}/checkpoints",
    )
    GNNMAPPOTrainer(env, cfg, hidden_dim=int(raw.get("model", {}).get("hidden_dim", 128))).train()


if __name__ == "__main__":
    main()
