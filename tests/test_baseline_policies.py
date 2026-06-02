import numpy as np

from algorithms.hungarian_assignment import HungarianAssignmentPolicy
from algorithms.rule_based import RuleBasedPolicy


def _common_inputs() -> dict[str, np.ndarray | float]:
    return {
        "defender_positions": np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
        "defender_velocities": np.zeros((2, 2), dtype=np.float32),
        "intruder_positions": np.array([[20.0, 0.0], [100.0, 0.0]], dtype=np.float32),
        "intruder_active": np.array([True, True]),
        "threat_scores": np.array([0.9, 0.2], dtype=np.float32),
        "protected_asset_position": np.array([50.0, 50.0], dtype=np.float32),
        "world_size": 1000.0,
    }


def test_rule_based_policy_outputs_bounded_actions() -> None:
    policy = RuleBasedPolicy()
    actions = policy.act(**_common_inputs())
    assert actions.shape == (2, 2)
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)


def test_hungarian_policy_outputs_bounded_actions() -> None:
    policy = HungarianAssignmentPolicy()
    actions = policy.act(
        intruder_velocities=np.array([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        **_common_inputs(),
    )
    assert actions.shape == (2, 2)
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)
