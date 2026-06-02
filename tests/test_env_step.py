import numpy as np

from envs.counter_uav_env import CounterUAVEnv, make_default_config


def test_env_step_returns_gymnasium_tuple() -> None:
    env = CounterUAVEnv(make_default_config())
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step_count"] == 1
