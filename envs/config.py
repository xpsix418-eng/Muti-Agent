from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from envs.counter_uav_env import CounterUAVConfig, config_from_mapping


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_env_config(path: str | Path) -> CounterUAVConfig:
    data = load_yaml(path)
    env_data = data.get("env", data)
    return config_from_mapping(env_data)


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
