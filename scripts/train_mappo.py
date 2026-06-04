from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from algorithms.mappo.trainer import MAPPOConfig, MAPPOTrainer
from envs.config import config_from_mapping, load_env_config, load_yaml, resolve_env_config_path
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_5v5_mappo.yaml")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    raw_config = load_extended_yaml(args.config)
    env_config = config_from_mapping(raw_config["env"]) if "env" in raw_config else load_env_config(resolve_env_config_path(args.config))
    if "curriculum" in raw_config:
        train_curriculum(raw_config, env_config, args.total_steps, args.checkpoint)
        return
    scenario = args.scenario or raw_config.get("scenario", "ScenarioA")
    env = CounterUAVEnv(apply_scenario_to_config(env_config, scenario))
    trainer_config = build_trainer_config(raw_config, scenario, args.total_steps)
    hidden_dim = int(raw_config.get("model", {}).get("hidden_dim", 128))
    trainer = MAPPOTrainer(env, trainer_config, hidden_dim=hidden_dim)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()
    print(f"checkpoint={Path(trainer_config.checkpoint_dir) / 'latest.pt'}")


def train_curriculum(
    raw_config: dict,
    base_env_config,
    total_steps_override: int | None,
    checkpoint: str | None,
) -> None:
    stages = raw_config.get("curriculum", {}).get("stages", [])
    if not stages:
        raise ValueError("curriculum.stages must contain at least one stage")
    previous_checkpoint = checkpoint
    hidden_dim = int(raw_config.get("model", {}).get("hidden_dim", 128))
    for stage in stages:
        stage_name = str(stage["name"])
        stage_steps = int(total_steps_override or stage.get("total_steps", raw_config.get("training", {}).get("total_steps", 100000)))
        env_config = replace(
            base_env_config,
            num_defenders=int(stage.get("num_defenders", base_env_config.num_defenders)),
            num_intruders=int(stage.get("num_intruders", base_env_config.num_intruders)),
        )
        env = CounterUAVEnv(env_config)
        trainer_config = build_trainer_config(raw_config, stage_name, stage_steps)
        trainer = MAPPOTrainer(env, trainer_config, hidden_dim=hidden_dim)
        if previous_checkpoint:
            load_actor_checkpoint_if_compatible(trainer, previous_checkpoint)
        trainer.train()
        previous_checkpoint = str(Path(trainer_config.checkpoint_dir) / "latest.pt")
        print(f"stage={stage_name} checkpoint={previous_checkpoint}")


def load_actor_checkpoint_if_compatible(trainer: MAPPOTrainer, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    try:
        trainer.actor.load_state_dict(checkpoint["actor"])
        if "obs_rms" in checkpoint:
            trainer.obs_rms.load_state_dict(checkpoint["obs_rms"])
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot transfer actor from {checkpoint_path}: {exc}") from exc


def build_trainer_config(raw_config: dict, scenario: str, total_steps_override: int | None) -> MAPPOConfig:
    training = raw_config.get("training", {})
    logging = raw_config.get("logging", {})
    total_steps = int(total_steps_override or training.get("total_steps", 100000))
    learning_rate = float(training.get("learning_rate", 3e-4))
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0.0")
    min_learning_rate = float(training.get("min_learning_rate", 0.0))
    if min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative")
    if min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")
    log_dir = str(logging.get("log_dir", "experiments/results/mappo"))
    scenario_log_dir = str(Path(log_dir) / scenario)
    return MAPPOConfig(
        total_steps=total_steps,
        rollout_length=int(training.get("rollout_length", 256)),
        gamma=float(training.get("gamma", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        clip_ratio=float(training.get("clip_ratio", 0.2)),
        entropy_coef=float(training.get("entropy_coef", 0.01)),
        entropy_coef_end=float(training["entropy_coef_end"]) if "entropy_coef_end" in training else None,
        value_coef=float(training.get("value_coef", 0.5)),
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
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


def load_extended_yaml(path: str | Path) -> dict:
    path = Path(path)
    data = load_yaml(path)
    parent = data.get("extends")
    if not parent:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent.parent / parent_path if path.parent.name == "configs" else path.parent / parent_path
        if not parent_path.exists():
            parent_path = Path.cwd() / parent
    merged = load_extended_yaml(parent_path)
    return deep_merge(merged, {key: value for key, value in data.items() if key != "extends"})


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
