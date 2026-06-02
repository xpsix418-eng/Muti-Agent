from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.threat_model import ThreatAssessment


@dataclass(frozen=True)
class RewardConfig:
    capture: float
    protected_zone_breach: float
    distance_shaping: float
    energy_penalty: float
    step_penalty: float


def team_reward(assessment: ThreatAssessment, actions: np.ndarray, config: RewardConfig) -> float:
    capture_reward = float(np.sum(assessment.captured) * config.capture)
    breach_penalty = float(np.sum(assessment.breached) * config.protected_zone_breach)
    distance_reward = float(-config.distance_shaping * np.mean(assessment.zone_distances))
    energy_cost = float(config.energy_penalty * np.mean(np.square(actions)))
    return capture_reward + breach_penalty + distance_reward - energy_cost + config.step_penalty
