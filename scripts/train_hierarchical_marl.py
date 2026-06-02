from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from algorithms.hierarchical_marl.trainer import HierarchicalMARLConfig, HierarchicalMARLTrainer
from envs.config import load_env_config, load_yaml, resolve_env_config_path
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hierarchical_marl.yaml")
    parser.add_argument("--scenario", default="ScenarioA")
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    env = CounterUAVEnv(apply_scenario_to_config(load_env_config(resolve_env_config_path(args.config)), args.scenario))
    training = raw.get("training", {})
    logging = raw.get("logging", {})
    cfg = HierarchicalMARLConfig(
        total_steps=int(args.total_steps or training.get("total_steps", 100000)),
        high_level_interval=int(training.get("option_horizon", training.get("high_level_interval", 10))),
        log_dir=f"{logging.get('log_dir', 'experiments/results/hierarchical_marl')}/{args.scenario}",
        seed=int(training.get("seed", 42)),
    )
    HierarchicalMARLTrainer(env, cfg).train()


if __name__ == "__main__":
    main()
