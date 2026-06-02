from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from algorithms.mappo.trainer import MAPPOConfig, MAPPOTrainer
from envs.config import load_env_config, load_yaml, resolve_env_config_path
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_mappo.yaml")
    parser.add_argument("--scenario", choices=["ScenarioA", "ScenarioB"], default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    env_config = load_env_config(resolve_env_config_path(args.config))
    scenario = args.scenario or raw_config.get("scenario", "ScenarioA")
    env = CounterUAVEnv(apply_scenario_to_config(env_config, scenario))
    trainer_config = build_trainer_config(raw_config, scenario, args.total_steps)
    hidden_dim = int(raw_config.get("model", {}).get("hidden_dim", 128))
    trainer = MAPPOTrainer(env, trainer_config, hidden_dim=hidden_dim)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()
    print(f"checkpoint={Path(trainer_config.checkpoint_dir) / 'latest.pt'}")


def build_trainer_config(raw_config: dict, scenario: str, total_steps_override: int | None) -> MAPPOConfig:
    training = raw_config.get("training", {})
    logging = raw_config.get("logging", {})
    total_steps = int(total_steps_override or training.get("total_steps", 100000))
    log_dir = str(logging.get("log_dir", "experiments/results/mappo"))
    scenario_log_dir = str(Path(log_dir) / scenario)
    return MAPPOConfig(
        total_steps=total_steps,
        rollout_length=int(training.get("rollout_length", 256)),
        gamma=float(training.get("gamma", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        clip_ratio=float(training.get("clip_ratio", 0.2)),
        entropy_coef=float(training.get("entropy_coef", 0.01)),
        value_coef=float(training.get("value_coef", 0.5)),
        learning_rate=float(training.get("learning_rate", 3e-4)),
        batch_size=int(training.get("batch_size", 1024)),
        epochs=int(training.get("epochs", 4)),
        max_grad_norm=float(training.get("max_grad_norm", 0.5)),
        value_clip=float(training.get("value_clip", 0.2)),
        reward_normalization=bool(training.get("reward_normalization", True)),
        observation_normalization=bool(training.get("observation_normalization", True)),
        lr_schedule=bool(training.get("lr_schedule", True)),
        seed=int(training.get("seed", 42)),
        log_dir=scenario_log_dir,
        checkpoint_dir=str(Path(scenario_log_dir) / "checkpoints"),
    )


if __name__ == "__main__":
    main()
