from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np

from algorithms.hierarchical_marl.high_level_policy import HighLevelPolicy
from algorithms.hierarchical_marl.low_level_policy import LowLevelPolicy
from envs.counter_uav_env import CounterUAVEnv


@dataclass
class HierarchicalMARLConfig:
    total_steps: int = 100_000
    high_level_interval: int = 10
    log_dir: str = "experiments/results/hierarchical_marl"
    seed: int = 42


class HierarchicalMARLTrainer:
    def __init__(self, env: CounterUAVEnv, config: HierarchicalMARLConfig):
        self.env = env
        self.config = config
        self.high_level = HighLevelPolicy(config.high_level_interval)
        self.low_level = LowLevelPolicy()

    def train(self) -> None:
        obs, info = self.env.reset(seed=self.config.seed)
        del obs
        rewards: list[float] = []
        intercept_rates: list[float] = []
        for _ in range(self.config.total_steps):
            agent_info = info[self.env.defense_agents[0]]
            high_action = self.high_level.select_action(self.env.step_count, {**agent_info, "protected_asset_position": self.env.protected_asset})
            active = ~(agent_info["intercepted"] | agent_info["breached"])
            actions = self.low_level.act(
                high_action,
                agent_info["defender_positions"],
                agent_info["defender_velocities"],
                agent_info["intruder_positions"][active],
                self.env.protected_asset,
            )
            _, reward_dict, terms, truncs, info = self.env.step(actions)
            rewards.append(float(np.mean(list(reward_dict.values()))))
            intercept_rates.append(float(np.mean(info[self.env.defense_agents[0]]["intercepted"])))
            if terms["__all__"] or truncs["__all__"]:
                _, info = self.env.reset(seed=self.config.seed + len(rewards))
        self.save_summary({"episode_reward": float(np.mean(rewards)), "intercept_rate": float(np.mean(intercept_rates))})

    def save_summary(self, metrics: dict[str, float]) -> None:
        path = Path(self.config.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
