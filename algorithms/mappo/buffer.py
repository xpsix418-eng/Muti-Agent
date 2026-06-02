from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        self.observations.append(observation.copy())
        self.actions.append(action.copy())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
