import numpy as np

from envs.dynamics import (
    DynamicsConfig,
    update_defender_dynamics,
    update_intruder_dynamics,
)


def test_defender_dynamics_limits_speed_and_bounds() -> None:
    config = DynamicsConfig(
        dt=1.0,
        world_size=100.0,
        defender_max_speed=2.0,
        intruder_max_speed=1.0,
        defender_acceleration_scale=10.0,
    )
    positions, velocities = update_defender_dynamics(
        positions=np.array([[99.0, 99.0]], dtype=np.float32),
        velocities=np.array([[0.0, 0.0]], dtype=np.float32),
        actions=np.array([[1.0, 1.0]], dtype=np.float32),
        config=config,
    )
    assert np.linalg.norm(velocities[0]) <= config.defender_max_speed + 1e-6
    assert np.all(positions >= 0.0)
    assert np.all(positions <= config.world_size)


def test_intruder_behaviors_keep_positions_in_bounds() -> None:
    config = DynamicsConfig(dt=1.0, world_size=100.0, defender_max_speed=2.0, intruder_max_speed=4.0)
    for behavior in ["straight_attack", "random_maneuver", "evasive_intruder"]:
        positions, velocities = update_intruder_dynamics(
            positions=np.array([[0.0, 50.0], [100.0, 50.0]], dtype=np.float32),
            velocities=np.zeros((2, 2), dtype=np.float32),
            protected_asset_position=np.array([50.0, 50.0], dtype=np.float32),
            defender_positions=np.array([[10.0, 50.0]], dtype=np.float32),
            config=config,
            behavior=behavior,
            rng=np.random.default_rng(1),
        )
        assert np.all(positions >= 0.0)
        assert np.all(positions <= config.world_size)
        assert np.all(np.linalg.norm(velocities, axis=1) <= config.intruder_max_speed + 1e-6)
