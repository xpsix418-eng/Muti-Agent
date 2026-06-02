from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces


Array = np.ndarray


@dataclass(frozen=True)
class CounterUAVConfig:
    world_size: float
    num_defenders: int
    num_intruders: int
    max_steps: int
    protected_asset_position: tuple[float, float]
    protected_radius: float
    intercept_radius: float
    defender_max_speed: float
    intruder_max_speed: float
    dt: float
    nearest_intruders: int
    nearest_teammates: int
    communication_radius: float
    collision_radius: float
    defender_acceleration_scale: float
    initial_energy: float
    energy_cost: float
    intercept_reward: float
    breach_penalty: float
    collision_penalty: float
    step_penalty: float
    threat_reward_scale: float


class CounterUAVEnv(gym.Env):
    """2D multi-agent interception simulation for MARL research only."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: CounterUAVConfig | None = None, config_path: str | Path | None = None):
        super().__init__()
        self.config = config or load_counter_uav_config(config_path or _default_config_path())
        self.rng = np.random.default_rng()
        self.step_count = 0

        self.defense_agents = [f"defender_{idx}" for idx in range(self.config.num_defenders)]
        self.intruder_agents = [f"intruder_{idx}" for idx in range(self.config.num_intruders)]
        self.possible_agents = list(self.defense_agents)
        self.agents = list(self.defense_agents)

        self.protected_asset = np.asarray(self.config.protected_asset_position, dtype=np.float32)
        self.defender_positions = np.zeros((self.config.num_defenders, 2), dtype=np.float32)
        self.defender_velocities = np.zeros((self.config.num_defenders, 2), dtype=np.float32)
        self.defender_energy = np.full(self.config.num_defenders, self.config.initial_energy, dtype=np.float32)
        self.intruder_positions = np.zeros((self.config.num_intruders, 2), dtype=np.float32)
        self.intruder_velocities = np.zeros((self.config.num_intruders, 2), dtype=np.float32)
        self.intruder_active = np.ones(self.config.num_intruders, dtype=bool)
        self.intercepted = np.zeros(self.config.num_intruders, dtype=bool)
        self.breached = np.zeros(self.config.num_intruders, dtype=bool)
        self.collision_events: list[tuple[str, str]] = []

        obs_dim = self._observation_dim()
        state_dim = self._global_state_dim()
        self.observation_spaces = {
            agent: spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32) for agent in self.defense_agents
        }
        self.action_spaces = {
            agent: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32) for agent in self.defense_agents
        }
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)
        self.state_space = spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], dict[str, dict[str, Any]]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        del options

        self.step_count = 0
        self.agents = list(self.defense_agents)
        self.defender_positions = self._spawn_defenders()
        self.defender_velocities.fill(0.0)
        self.defender_energy.fill(self.config.initial_energy)
        self.intruder_positions = self._spawn_intruders()
        self.intruder_velocities = self._intruder_guidance()
        self.intruder_active.fill(True)
        self.intercepted.fill(False)
        self.breached.fill(False)
        self.collision_events = []
        return self._observations(), self._infos()

    def step(
        self, actions: dict[str, Array] | Array
    ) -> tuple[dict[str, Array], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        action_array = self._normalize_actions(actions)
        self._update_defenders(action_array)
        self._update_intruders()

        intercepted_now = self._detect_intercepts()
        breached_now = self._detect_breaches()
        collisions = self._detect_collisions()
        self.collision_events = collisions
        self.step_count += 1

        all_done = bool(np.all(~self.intruder_active) or np.any(self.breached))
        truncated = self.step_count >= self.config.max_steps
        rewards = self._rewards(action_array, intercepted_now, breached_now, collisions)
        terminations = {agent: all_done for agent in self.defense_agents}
        terminations["__all__"] = all_done
        truncations = {agent: truncated for agent in self.defense_agents}
        truncations["__all__"] = truncated
        return self._observations(), rewards, terminations, truncations, self._infos()

    def get_observation(self, agent_id: str) -> Array:
        if agent_id not in self.defense_agents:
            raise KeyError(f"Unknown defense agent: {agent_id}")
        return self._agent_observation(self.defense_agents.index(agent_id))

    def get_global_state(self) -> Array:
        threat_scores = self._threat_scores()
        topology = self._communication_topology().astype(np.float32).ravel()
        state = np.concatenate(
            [
                self.defender_positions.ravel(),
                self.defender_velocities.ravel(),
                self.intruder_positions.ravel(),
                self.intruder_velocities.ravel(),
                threat_scores,
                self.protected_asset,
                np.asarray([self.step_count / max(self.config.max_steps, 1)], dtype=np.float32),
                topology,
            ]
        )
        return state.astype(np.float32)

    def get_available_agents(self) -> list[str]:
        return list(self.agents)

    def render(self, mode: str = "human") -> Array | None:
        if mode == "human":
            print(
                f"step={self.step_count} active_intruders={int(np.sum(self.intruder_active))} "
                f"breached={int(np.sum(self.breached))} intercepted={int(np.sum(self.intercepted))}"
            )
            return None
        if mode == "rgb_array":
            return self._render_rgb_array()
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        return None

    def _update_defenders(self, actions: Array) -> None:
        acceleration = np.clip(actions, -1.0, 1.0) * self.config.defender_acceleration_scale
        active_energy = (self.defender_energy > 0.0).astype(np.float32)[:, None]
        self.defender_velocities += acceleration * self.config.dt * active_energy
        self.defender_velocities = _limit_speed(self.defender_velocities, self.config.defender_max_speed)
        self.defender_positions += self.defender_velocities * self.config.dt * active_energy
        self.defender_positions = np.clip(self.defender_positions, 0.0, self.config.world_size)
        energy_delta = self.config.energy_cost * np.linalg.norm(acceleration, axis=1) * self.config.dt
        self.defender_energy = np.maximum(0.0, self.defender_energy - energy_delta.astype(np.float32))

    def _update_intruders(self) -> None:
        self.intruder_velocities = self._intruder_guidance()
        active = self.intruder_active.astype(np.float32)[:, None]
        self.intruder_positions += self.intruder_velocities * self.config.dt * active
        self.intruder_positions = np.clip(self.intruder_positions, 0.0, self.config.world_size)

    def _intruder_guidance(self) -> Array:
        direction = self.protected_asset[None, :] - self.intruder_positions
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        velocity = direction / np.maximum(norms, 1e-6) * self.config.intruder_max_speed
        velocity[~self.intruder_active] = 0.0
        return velocity.astype(np.float32)

    def _detect_intercepts(self) -> Array:
        distances = np.linalg.norm(
            self.defender_positions[:, None, :] - self.intruder_positions[None, :, :], axis=-1
        )
        intercepted_now = (distances.min(axis=0) <= self.config.intercept_radius) & self.intruder_active
        self.intercepted |= intercepted_now
        self.intruder_active[intercepted_now] = False
        self.intruder_velocities[intercepted_now] = 0.0
        return intercepted_now

    def _detect_breaches(self) -> Array:
        distances = np.linalg.norm(self.intruder_positions - self.protected_asset[None, :], axis=1)
        breached_now = (distances <= self.config.protected_radius) & self.intruder_active
        self.breached |= breached_now
        self.intruder_active[breached_now] = False
        self.intruder_velocities[breached_now] = 0.0
        return breached_now

    def _detect_collisions(self) -> list[tuple[str, str]]:
        collisions: list[tuple[str, str]] = []
        distances = np.linalg.norm(
            self.defender_positions[:, None, :] - self.defender_positions[None, :, :], axis=-1
        )
        pair_mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        rows, cols = np.where((distances <= self.config.collision_radius) & pair_mask)
        for row, col in zip(rows.tolist(), cols.tolist()):
            collisions.append((self.defense_agents[row], self.defense_agents[col]))
        return collisions

    def _rewards(
        self,
        actions: Array,
        intercepted_now: Array,
        breached_now: Array,
        collisions: list[tuple[str, str]],
    ) -> dict[str, float]:
        team_reward = (
            float(np.sum(intercepted_now)) * self.config.intercept_reward
            + float(np.sum(breached_now)) * self.config.breach_penalty
            + self.config.step_penalty
        )
        threat_pressure = float(np.mean(self._threat_scores())) if self.config.num_intruders else 0.0
        energy_penalty = 0.01 * np.linalg.norm(actions, axis=1)
        rewards = {}
        collision_agents = {agent for pair in collisions for agent in pair}
        for idx, agent in enumerate(self.defense_agents):
            reward = team_reward - self.config.threat_reward_scale * threat_pressure - float(energy_penalty[idx])
            if agent in collision_agents:
                reward += self.config.collision_penalty
            rewards[agent] = float(reward)
        return rewards

    def _agent_observation(self, defender_idx: int) -> Array:
        own_pos = self.defender_positions[defender_idx]
        own_vel = self.defender_velocities[defender_idx]
        own_energy = np.asarray([self.defender_energy[defender_idx]], dtype=np.float32)

        intruder_idx = self._nearest_indices(self.intruder_positions, own_pos, self.config.nearest_intruders)
        intruder_rel_pos = self._relative_block(self.intruder_positions, own_pos, intruder_idx, self.config.nearest_intruders)
        intruder_rel_vel = self._relative_block(
            self.intruder_velocities, own_vel, intruder_idx, self.config.nearest_intruders
        )
        threat_scores = self._threat_scores()
        intruder_threat = self._scalar_block(threat_scores, intruder_idx, self.config.nearest_intruders)

        teammate_positions = np.delete(self.defender_positions, defender_idx, axis=0)
        teammate_velocities = np.delete(self.defender_velocities, defender_idx, axis=0)
        teammate_idx = self._nearest_indices(teammate_positions, own_pos, self.config.nearest_teammates)
        teammate_rel_pos = self._relative_block(teammate_positions, own_pos, teammate_idx, self.config.nearest_teammates)
        teammate_rel_vel = self._relative_block(
            teammate_velocities, own_vel, teammate_idx, self.config.nearest_teammates
        )

        communication_available = np.asarray([float(np.any(self._communication_topology()[defender_idx]))], dtype=np.float32)
        asset_rel_pos = self.protected_asset - own_pos
        observation = np.concatenate(
            [
                own_pos,
                own_vel,
                own_energy,
                intruder_rel_pos,
                intruder_rel_vel,
                intruder_threat,
                teammate_rel_pos,
                teammate_rel_vel,
                communication_available,
                asset_rel_pos,
            ]
        )
        return observation.astype(np.float32)

    def _observations(self) -> dict[str, Array]:
        return {agent: self.get_observation(agent) for agent in self.defense_agents}

    def _infos(self) -> dict[str, dict[str, Any]]:
        global_info = {
            "step_count": self.step_count,
            "global_state": self.get_global_state(),
            "defender_positions": self.defender_positions.copy(),
            "defender_velocities": self.defender_velocities.copy(),
            "intruder_positions": self.intruder_positions.copy(),
            "intruder_velocities": self.intruder_velocities.copy(),
            "intercepted": self.intercepted.copy(),
            "breached": self.breached.copy(),
            "collision_events": list(self.collision_events),
            "communication_topology": self._communication_topology(),
        }
        return {agent: dict(global_info) for agent in self.defense_agents}

    def _threat_scores(self) -> Array:
        distances = np.linalg.norm(self.intruder_positions - self.protected_asset[None, :], axis=1)
        normalized = 1.0 - distances / max(self.config.world_size, 1e-6)
        scores = np.clip(normalized, 0.0, 1.0)
        scores[~self.intruder_active] = 0.0
        return scores.astype(np.float32)

    def _communication_topology(self) -> Array:
        distances = np.linalg.norm(
            self.defender_positions[:, None, :] - self.defender_positions[None, :, :], axis=-1
        )
        topology = (distances <= self.config.communication_radius).astype(np.float32)
        np.fill_diagonal(topology, 0.0)
        return topology

    def _spawn_defenders(self) -> Array:
        radius = self.config.protected_radius * 0.5
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.config.num_defenders)
        radii = self.rng.uniform(0.0, radius, size=self.config.num_defenders)
        positions = self.protected_asset[None, :] + np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
        return np.clip(positions, 0.0, self.config.world_size).astype(np.float32)

    def _spawn_intruders(self) -> Array:
        side = self.rng.integers(0, 4, size=self.config.num_intruders)
        positions = self.rng.uniform(0.0, self.config.world_size, size=(self.config.num_intruders, 2))
        positions[side == 0, 0] = 0.0
        positions[side == 1, 0] = self.config.world_size
        positions[side == 2, 1] = 0.0
        positions[side == 3, 1] = self.config.world_size
        return positions.astype(np.float32)

    def _normalize_actions(self, actions: dict[str, Array] | Array) -> Array:
        if isinstance(actions, dict):
            action_array = np.zeros((self.config.num_defenders, 2), dtype=np.float32)
            for idx, agent in enumerate(self.defense_agents):
                action_array[idx] = np.asarray(actions.get(agent, np.zeros(2)), dtype=np.float32)
            return np.clip(action_array, -1.0, 1.0)
        return np.clip(np.asarray(actions, dtype=np.float32).reshape(self.config.num_defenders, 2), -1.0, 1.0)

    def _nearest_indices(self, positions: Array, center: Array, count: int) -> Array:
        if len(positions) == 0 or count <= 0:
            return np.asarray([], dtype=np.int64)
        distances = np.linalg.norm(positions - center[None, :], axis=1)
        return np.argsort(distances)[: min(count, len(positions))]

    def _relative_block(self, values: Array, reference: Array, indices: Array, count: int) -> Array:
        block = np.zeros((count, 2), dtype=np.float32)
        if len(indices) > 0:
            selected = values[indices] - reference[None, :]
            block[: len(indices)] = selected.astype(np.float32)
        return block.ravel()

    def _scalar_block(self, values: Array, indices: Array, count: int) -> Array:
        block = np.zeros(count, dtype=np.float32)
        if len(indices) > 0:
            block[: len(indices)] = values[indices].astype(np.float32)
        return block

    def _observation_dim(self) -> int:
        k_intruders = self.config.nearest_intruders
        k_teammates = self.config.nearest_teammates
        return 2 + 2 + 1 + 2 * k_intruders + 2 * k_intruders + k_intruders + 2 * k_teammates + 2 * k_teammates + 1 + 2

    def _global_state_dim(self) -> int:
        return (
            self.config.num_defenders * 2
            + self.config.num_defenders * 2
            + self.config.num_intruders * 2
            + self.config.num_intruders * 2
            + self.config.num_intruders
            + 2
            + 1
            + self.config.num_defenders * self.config.num_defenders
        )

    def _render_rgb_array(self) -> Array:
        image_size = 512
        image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
        scale = (image_size - 1) / self.config.world_size
        asset = np.round(self.protected_asset * scale).astype(int)
        image[max(asset[1] - 3, 0) : min(asset[1] + 4, image_size), max(asset[0] - 3, 0) : min(asset[0] + 4, image_size)] = (
            0,
            180,
            0,
        )
        for point in np.round(self.defender_positions * scale).astype(int):
            image[max(point[1] - 2, 0) : min(point[1] + 3, image_size), max(point[0] - 2, 0) : min(point[0] + 3, image_size)] = (
                0,
                80,
                220,
            )
        for idx, point in enumerate(np.round(self.intruder_positions * scale).astype(int)):
            color = (220, 40, 40) if self.intruder_active[idx] else (160, 160, 160)
            image[max(point[1] - 2, 0) : min(point[1] + 3, image_size), max(point[0] - 2, 0) : min(point[0] + 3, image_size)] = color
        return image


def _limit_speed(velocities: Array, max_speed: float) -> Array:
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    scales = np.minimum(1.0, max_speed / np.maximum(norms, 1e-6))
    return (velocities * scales).astype(np.float32)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "env_2d.yaml"


def load_counter_uav_config(path: str | Path) -> CounterUAVConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    env_data = data.get("env", data)
    return config_from_mapping(env_data)


def config_from_mapping(env_data: dict[str, Any]) -> CounterUAVConfig:
    world_size = env_data["world_size"]
    if isinstance(world_size, list):
        world_size_value = float(world_size[0])
    else:
        world_size_value = float(world_size)
    protected_asset_position = tuple(float(value) for value in env_data["protected_asset_position"])
    return CounterUAVConfig(
        world_size=world_size_value,
        num_defenders=int(env_data["num_defenders"]),
        num_intruders=int(env_data["num_intruders"]),
        max_steps=int(env_data["max_steps"]),
        protected_asset_position=(protected_asset_position[0], protected_asset_position[1]),
        protected_radius=float(env_data["protected_radius"]),
        intercept_radius=float(env_data["intercept_radius"]),
        defender_max_speed=float(env_data["defender_max_speed"]),
        intruder_max_speed=float(env_data["intruder_max_speed"]),
        dt=float(env_data["dt"]),
        nearest_intruders=int(env_data.get("nearest_intruders", 4)),
        nearest_teammates=int(env_data.get("nearest_teammates", 3)),
        communication_radius=float(env_data.get("communication_radius", 250.0)),
        collision_radius=float(env_data.get("collision_radius", 5.0)),
        defender_acceleration_scale=float(env_data.get("defender_acceleration_scale", 4.0)),
        initial_energy=float(env_data.get("initial_energy", 100.0)),
        energy_cost=float(env_data.get("energy_cost", 0.05)),
        intercept_reward=float(env_data.get("reward", {}).get("intercept", 10.0)),
        breach_penalty=float(env_data.get("reward", {}).get("breach", -25.0)),
        collision_penalty=float(env_data.get("reward", {}).get("collision", -2.0)),
        step_penalty=float(env_data.get("reward", {}).get("step", -0.01)),
        threat_reward_scale=float(env_data.get("reward", {}).get("threat_scale", 0.1)),
    )


def make_default_config() -> CounterUAVConfig:
    return load_counter_uav_config(_default_config_path())
