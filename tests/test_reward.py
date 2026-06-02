import numpy as np

from envs.reward import RewardConfig, team_reward
from envs.threat_model import ThreatAssessment


def test_capture_reward_positive() -> None:
    assessment = ThreatAssessment(
        nearest_distances=np.array([1.0]),
        zone_distances=np.array([50.0]),
        captured=np.array([True]),
        breached=np.array([False]),
    )
    reward = team_reward(assessment, np.zeros((1, 2)), RewardConfig(10.0, -20.0, 0.0, 0.0, 0.0))
    assert reward == 10.0
