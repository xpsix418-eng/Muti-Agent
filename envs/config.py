from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from envs.counter_uav_env import CounterUAVConfig
from envs.reward import RewardConfig
from envs.scenarios import ScenarioConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_env_config(path: str | Path) -> CounterUAVConfig:
    data = load_yaml(path)
    env_data = data.get("env", data)
    reward_data = env_data["reward"]
    scenario_data = env_data["scenario"]
    world_size = tuple(float(value) for value in env_data["world_size"])

    reward = RewardConfig(
        capture=float(reward_data["capture"]),
        protected_zone_breach=float(reward_data["protected_zone_breach"]),
        distance_shaping=float(reward_data["distance_shaping"]),
        energy_penalty=float(reward_data["energy_penalty"]),
        step_penalty=float(reward_data["step_penalty"]),
    )
    scenario = ScenarioConfig(
        dim=int(env_data["dim"]),
        world_size=world_size,
        num_defenders=int(env_data["num_defenders"]),
        num_intruders=int(env_data["num_intruders"]),
        defender_spawn_radius=float(scenario_data["defender_spawn_radius"]),
        intruder_spawn_min_radius=float(scenario_data["intruder_spawn_min_radius"]),
        intruder_spawn_max_radius=float(scenario_data["intruder_spawn_max_radius"]),
    )
    return CounterUAVConfig(
        dim=int(env_data["dim"]),
        max_steps=int(env_data["max_steps"]),
        dt=float(env_data["dt"]),
        world_size=world_size,
        num_defenders=int(env_data["num_defenders"]),
        num_intruders=int(env_data["num_intruders"]),
        defender_speed=float(env_data["defender_speed"]),
        intruder_speed=float(env_data["intruder_speed"]),
        capture_radius=float(env_data["capture_radius"]),
        protected_zone_radius=float(env_data["protected_zone_radius"]),
        reward=reward,
        scenario=scenario,
    )


def resolve_env_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    data = load_yaml(config_path)
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict) or "env_config" not in defaults:
        return config_path
    env_path = Path(defaults["env_config"])
    if env_path.is_absolute():
        return env_path
    return config_path.parent.parent / env_path
