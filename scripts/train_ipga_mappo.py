from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from algorithms.ipga_mappo.trainer import IPGAMAPPOConfig, IPGAMAPPOTrainer
from envs.config import config_from_mapping, load_yaml
from envs.counter_uav_env import CounterUAVEnv
from envs.scenarios import apply_scenario_to_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_ipga_mappo_5v5.yaml")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(max(1, args.torch_threads))

    raw_config = load_extended_yaml(args.config)
    env_config = config_from_mapping(raw_config.get("env", raw_config))
    scenario = args.scenario or raw_config.get("scenario", "Scenario5v5")
    env = CounterUAVEnv(apply_scenario_to_config(env_config, scenario))
    trainer_config = build_trainer_config(raw_config, scenario, args.total_steps)
    trainer = IPGAMAPPOTrainer(env, trainer_config, device=args.device)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()
    print(f"checkpoint={Path(trainer_config.checkpoint_dir) / 'latest.pt'}")


def build_trainer_config(raw_config: dict, scenario: str, total_steps_override: int | None) -> IPGAMAPPOConfig:
    training = raw_config.get("training", {})
    model = raw_config.get("model", {})
    ipga = raw_config.get("ipga", {})
    logging = raw_config.get("logging", {})
    total_steps = int(total_steps_override or training.get("total_env_steps", training.get("total_steps", 2_000_000)))
    learning_rate = float(training.get("learning_rate", 3e-4))
    min_learning_rate = float(training.get("min_learning_rate", 3e-5))
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0.0")
    if min_learning_rate < 0.0 or min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate must be in [0, learning_rate]")
    rollout_length = int(training.get("rollout_steps", training.get("rollout_length", 512)))
    batch_size = int(training.get("mini_batch_size", training.get("batch_size", 2048)))
    epochs = int(training.get("ppo_epoch", training.get("epochs", 5)))
    log_dir = str(Path(str(logging.get("log_dir", "experiments/results/ipga_mappo"))) / scenario)
    return IPGAMAPPOConfig(
        total_steps=total_steps,
        rollout_length=rollout_length,
        gamma=float(training.get("gamma", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        clip_ratio=float(training.get("clip_ratio", 0.2)),
        entropy_coef=float(training.get("entropy_coef", 0.008)),
        entropy_coef_end=float(training["entropy_coef_end"]) if "entropy_coef_end" in training else None,
        value_coef=float(training.get("value_coef", 0.5)),
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        max_grad_norm=float(training.get("max_grad_norm", 0.5)),
        value_clip=float(training.get("value_clip", 0.2)),
        reward_normalization=bool(training.get("reward_normalization", True)),
        observation_normalization=bool(training.get("observation_normalization", True)),
        lr_schedule=bool(training.get("lr_schedule", True)),
        seed=int(training.get("seed", 42)),
        prediction_horizon=float(ipga.get("prediction_horizon", raw_config.get("env", {}).get("prediction_horizon", 5.0))),
        hidden_dim=int(model.get("hidden_dim", 128)),
        graph_hidden_dim=int(model.get("graph_hidden_dim", 128)),
        num_graph_layers=int(model.get("num_graph_layers", 2)),
        attention_heads=int(model.get("attention_heads", 4)),
        assignment_loss_start=float(ipga.get("assignment_loss_start", 0.04)),
        assignment_loss_end=float(ipga.get("assignment_loss_end", 0.0)),
        assignment_loss_decay_steps=int(ipga.get("assignment_loss_decay_steps", 1_000_000)),
        use_graph=bool(ipga.get("use_graph", True)),
        graph_type=str(ipga.get("graph_type", "ipg")),
        use_interception_point_nodes=bool(ipga.get("use_interception_point_nodes", True)),
        use_assignment_gate=bool(ipga.get("use_assignment_gate", True)),
        use_ita_features=bool(ipga.get("use_ita_features", True)),
        use_assignment_loss=bool(ipga.get("use_assignment_loss", True)),
        log_dir=log_dir,
        checkpoint_dir=str(Path(log_dir) / "checkpoints"),
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
