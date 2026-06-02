import numpy as np

from envs.threat_model import ThreatModel, assess_threats


def test_assess_threats_marks_capture() -> None:
    assessment = assess_threats(
        defender_positions=np.array([[0.0, 0.0]]),
        intruder_positions=np.array([[1.0, 0.0]]),
        protected_center=np.array([10.0, 10.0]),
        capture_radius=2.0,
        protected_zone_radius=1.0,
    )
    assert assessment.captured.tolist() == [True]
    assert assessment.breached.tolist() == [False]


def test_threat_score_is_normalized() -> None:
    model = ThreatModel(world_size=1000.0, protected_radius=80.0, intruder_max_speed=8.0)
    scores = model.score(
        intruder_positions=np.array([[900.0, 500.0], [530.0, 500.0], [100.0, 100.0]], dtype=np.float32),
        intruder_velocities=np.array([[-8.0, 0.0], [-4.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        protected_asset_position=np.array([500.0, 500.0], dtype=np.float32),
    )
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert scores.shape == (3,)
