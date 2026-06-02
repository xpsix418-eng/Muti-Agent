import numpy as np

from envs.reward import (
    RewardConfig,
    RewardEvents,
    blocking_position_reward,
    compute_reward,
    detect_breaches,
    detect_defender_collisions,
    detect_intercepts,
    intercept_point_approach_reward,
    intruder_progress_penalty,
    predict_intercept_points,
    team_reward,
    ttc_advantage_reward,
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


def test_intercept_point_reward_terms_are_finite_and_directional() -> None:
    previous_defenders = np.array([[0.0, 0.0]], dtype=np.float32)
    current_defenders = np.array([[2.0, 0.0]], dtype=np.float32)
    intruders = np.array([[10.0, 0.0]], dtype=np.float32)
    velocities = np.array([[1.0, 0.0]], dtype=np.float32)
    asset = np.array([20.0, 0.0], dtype=np.float32)
    assignments = np.array([0], dtype=np.int64)
    intercept_points = predict_intercept_points(intruders, velocities, prediction_horizon=2.0)
    approach = intercept_point_approach_reward(
        previous_defenders,
        current_defenders,
        intercept_points,
        intercept_points,
        assignments,
    )
    blocking = blocking_position_reward(current_defenders, intruders, asset, assignments, blocking_sigma=10.0)
    ttc = ttc_advantage_reward(current_defenders, intruders, velocities, asset, intercept_points, assignments, defender_max_speed=12.0)
    progress_penalty = intruder_progress_penalty(
        previous_intruder_positions=np.array([[8.0, 0.0]], dtype=np.float32),
        current_intruder_positions=intruders,
        protected_asset_position=asset,
        active_mask=np.array([True]),
    )
    assert intercept_points.tolist() == [[12.0, 0.0]]
    assert approach[0] > 0.0
    assert np.isfinite(blocking[0])
    assert ttc[0] > 0.0
    assert progress_penalty > 0.0
