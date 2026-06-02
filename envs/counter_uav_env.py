from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.dynamics import PointMassConfig, integrate_velocity
from envs.reward import RewardConfig, team_reward
from envs.scenarios import ScenarioConfig, random_2d
from envs.threat_model import assess_threats


@dataclass(frozen=True)
class CounterUAVConfig:
    dim: int
    max_steps: int
    dt: float
    world_size: tuple[float, ...]
    num_defenders: int
    num_intruders: int
    defender_speed: float
    intruder_speed: float
    capture_radius: float
    protected_zone_radius: float
    reward: RewardConfig
    scenario: ScenarioConfig


class CounterUAVEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: CounterUAVConfig):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng()
        self.step_count = 0
        obs_dim = (config.num_defenders + config.num_intruders + 1) * config.dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-config.defender_speed,
            high=config.defender_speed,
            shape=(config.num_defenders, config.dim),
            dtype=np.float32,
        )
        self.defenders = np.zeros((config.num_defenders, config.dim), dtype=np.float32)
        self.intruders = np.zeros((config.num_intruders, config.dim), dtype=np.float32)
        self.protected_center = np.zeros(config.dim, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.defenders, self.intruders, self.protected_center = random_2d(self.config.scenario, self.rng)
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        actions = np.asarray(action, dtype=np.float32).reshape(self.config.num_defenders, self.config.dim)
        self.defenders = integrate_velocity(
            self.defenders,
            actions,
            PointMassConfig(self.config.dt, self.config.defender_speed, self.config.world_size),
        )
        self._move_intruders()
        assessment = assess_threats(
            self.defenders,
            self.intruders,
            self.protected_center,
            self.config.capture_radius,
            self.config.protected_zone_radius,
        )
        reward = team_reward(assessment, actions, self.config.reward)
        self.step_count += 1
        terminated = bool(np.all(assessment.captured) or np.any(assessment.breached))
        truncated = self.step_count >= self.config.max_steps
        info = self._info()
        info["captured"] = assessment.captured
        info["breached"] = assessment.breached
        return self._observation(), reward, terminated, truncated, info

    def _move_intruders(self) -> None:
        direction = self.protected_center[None, :] - self.intruders
        norms = np.linalg.norm(direction, axis=-1, keepdims=True)
        velocity = direction / np.maximum(norms, 1e-6) * self.config.intruder_speed
        self.intruders = integrate_velocity(
            self.intruders,
            velocity,
            PointMassConfig(self.config.dt, self.config.intruder_speed, self.config.world_size),
        )

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [self.defenders.ravel(), self.intruders.ravel(), self.protected_center.ravel()]
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "defenders": self.defenders.copy(),
            "intruders": self.intruders.copy(),
            "protected_center": self.protected_center.copy(),
        }


def make_default_config() -> CounterUAVConfig:
    reward = RewardConfig(10.0, -20.0, 0.05, 0.01, -0.01)
    scenario = ScenarioConfig(2, (100.0, 100.0), 3, 2, 10.0, 35.0, 48.0)
    return CounterUAVConfig(2, 200, 0.1, (100.0, 100.0), 3, 2, 8.0, 5.0, 2.5, 8.0, reward, scenario)
