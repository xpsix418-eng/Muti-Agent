import numpy as np

from envs.threat_model import assess_threats


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
