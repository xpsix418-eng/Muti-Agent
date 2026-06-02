from envs.counter_uav_env import CounterUAVEnv, make_default_config


def test_env_reset_observation_shape() -> None:
    env = CounterUAVEnv(make_default_config())
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert info["step_count"] == 0
