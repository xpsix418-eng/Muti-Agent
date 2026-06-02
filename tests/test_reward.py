import numpy as np

from envs.reward import (
    RewardConfig,
    RewardEvents,
    compute_reward,
    detect_breaches,
    detect_defender_collisions,
    detect_intercepts,
    team_reward,
)
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


def test_composed_reward_is_finite() -> None:
    rewards = compute_reward(
        actions=np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float32),
        threat_scores=np.array([0.9, 0.2], dtype=np.float32),
        events=RewardEvents(
            intercepted=np.array([True, False]),
            breached=np.array([False, False]),
            defender_collision_pairs=[],
            communication_links=1,
        ),
        config=RewardConfig(),
        num_defenders=2,
        mode="team_reward",
    )
    assert all(np.isfinite(value) for value in rewards.values())


def test_intercept_breach_and_collision_events_trigger() -> None:
    intercepted, nearest = detect_intercepts(
        defender_positions=np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32),
        intruder_positions=np.array([[5.0, 0.0], [200.0, 200.0]], dtype=np.float32),
        intercept_radius=10.0,
    )
    breached = detect_breaches(
        intruder_positions=np.array([[505.0, 500.0], [900.0, 900.0]], dtype=np.float32),
        protected_asset_position=np.array([500.0, 500.0], dtype=np.float32),
        protected_radius=10.0,
    )
    collisions = detect_defender_collisions(
        defender_positions=np.array([[0.0, 0.0], [3.0, 4.0], [100.0, 100.0]], dtype=np.float32),
        collision_radius=5.0,
    )
    assert intercepted.tolist() == [True, False]
    assert nearest.tolist() == [0, 1]
    assert breached.tolist() == [True, False]
    assert collisions == [(0, 1)]
