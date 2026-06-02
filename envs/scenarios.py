from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from envs.counter_uav_env import CounterUAVConfig


Array = np.ndarray
ScenarioName = Literal["ScenarioA", "ScenarioB", "ScenarioC", "ScenarioD", "ScenarioE", "ScenarioF"]


@dataclass(frozen=True)
class ScenarioConfig:
    dim: int
    world_size: tuple[float, ...]
    num_defenders: int
    num_intruders: int
    defender_spawn_radius: float
    intruder_spawn_min_radius: float
    intruder_spawn_max_radius: float


@dataclass(frozen=True)
class ScenarioSpec:
    name: ScenarioName
    description: str
    num_defenders: int
    num_intruders: int
    comm_radius: float | None = None
    intruder_behavior: str | None = None
    packet_loss: float = 0.0
    communication_delay: int = 0
    observation_noise: float = 0.0
    high_threat_fraction: float = 0.0
    changed_intruder_directions: bool = False


SCENARIOS: dict[str, ScenarioSpec] = {
    "ScenarioA": ScenarioSpec("ScenarioA", "Small-scale basic scenario", 4, 8),
    "ScenarioB": ScenarioSpec("ScenarioB", "Medium-scale cooperation scenario", 8, 16),
    "ScenarioC": ScenarioSpec("ScenarioC", "Large-scale stress scenario", 16, 32),
    "ScenarioD": ScenarioSpec(
        "ScenarioD",
        "High-threat prioritization scenario",
        8,
        16,
        high_threat_fraction=0.35,
        intruder_behavior="straight_attack",
    ),
    "ScenarioE": ScenarioSpec(
        "ScenarioE",
        "Communication-limited scenario",
        8,
        16,
        comm_radius=120.0,
        packet_loss=0.35,
        communication_delay=2,
        observation_noise=3.0,
        intruder_behavior="random_maneuver",
    ),
    "ScenarioF": ScenarioSpec(
        "ScenarioF",
        "Generalization test with changed agent counts and approach directions",
        10,
        20,
        intruder_behavior="random_maneuver",
        changed_intruder_directions=True,
    ),
}


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name}. Available: {', '.join(SCENARIOS)}")
    return SCENARIOS[name]


def apply_scenario_to_config(config: CounterUAVConfig, scenario_name: str) -> CounterUAVConfig:
    scenario = get_scenario(scenario_name)
    updates: dict[str, Any] = {
        "num_defenders": scenario.num_defenders,
        "num_intruders": scenario.num_intruders,
    }
    if scenario.comm_radius is not None:
        updates["comm_radius"] = scenario.comm_radius
    if scenario.packet_loss > 0.0:
        updates["packet_loss_prob"] = scenario.packet_loss
    if scenario.communication_delay > 0:
        updates["comm_delay_steps"] = scenario.communication_delay
    if scenario.observation_noise > 0.0:
        updates["noisy_observation_std"] = scenario.observation_noise
        updates["partial_observation"] = True
    if scenario.intruder_behavior is not None:
        updates["intruder_behavior"] = scenario.intruder_behavior
    return replace(config, **updates)


def initialize_scenario_state(env: Any, scenario_name: str, rng: np.random.Generator) -> None:
    scenario = get_scenario(scenario_name)
    if scenario.high_threat_fraction > 0.0:
        _place_high_threat_intruders(env, scenario.high_threat_fraction, rng)
    if scenario.changed_intruder_directions:
        _redistribute_intruders_by_direction(env, rng)
    if scenario.observation_noise > 0.0:
        noise = rng.normal(0.0, scenario.observation_noise, size=env.intruder_positions.shape)
        env.intruder_positions = np.clip(env.intruder_positions + noise, 0.0, env.config.world_size).astype(np.float32)
    env.intruder_velocities = env._intruder_guidance()


def scenario_metadata(scenario_name: str) -> dict[str, float | int | str]:
    scenario = get_scenario(scenario_name)
    return {
        "name": scenario.name,
        "packet_loss": scenario.packet_loss,
        "communication_delay": scenario.communication_delay,
        "observation_noise": scenario.observation_noise,
        "comm_radius": scenario.comm_radius if scenario.comm_radius is not None else -1.0,
        "num_defenders": scenario.num_defenders,
        "num_intruders": scenario.num_intruders,
    }


def random_2d(config: ScenarioConfig, rng: np.random.Generator) -> tuple[Array, Array, Array]:
    center = np.asarray(config.world_size, dtype=np.float32) / 2.0
    defender_angles = rng.uniform(0.0, 2.0 * np.pi, size=config.num_defenders)
    defender_radii = rng.uniform(0.0, config.defender_spawn_radius, size=config.num_defenders)
    defenders = center + np.stack(
        [np.cos(defender_angles) * defender_radii, np.sin(defender_angles) * defender_radii],
        axis=-1,
    )

    intruder_angles = rng.uniform(0.0, 2.0 * np.pi, size=config.num_intruders)
    intruder_radii = rng.uniform(
        config.intruder_spawn_min_radius,
        config.intruder_spawn_max_radius,
        size=config.num_intruders,
    )
    intruders = center + np.stack(
        [np.cos(intruder_angles) * intruder_radii, np.sin(intruder_angles) * intruder_radii],
        axis=-1,
    )
    return defenders.astype(np.float32), intruders.astype(np.float32), center.astype(np.float32)


def _place_high_threat_intruders(env: Any, fraction: float, rng: np.random.Generator) -> None:
    count = max(1, int(round(env.config.num_intruders * fraction)))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radii = rng.uniform(env.config.protected_radius * 1.25, env.config.protected_radius * 2.5, size=count)
    offsets = np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
    env.intruder_positions[:count] = np.clip(env.protected_asset[None, :] + offsets, 0.0, env.config.world_size)


def _redistribute_intruders_by_direction(env: Any, rng: np.random.Generator) -> None:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=env.config.num_intruders)
    radii = rng.uniform(env.config.world_size * 0.35, env.config.world_size * 0.5, size=env.config.num_intruders)
    positions = env.protected_asset[None, :] + np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
    env.intruder_positions = np.clip(positions, 0.0, env.config.world_size).astype(np.float32)
