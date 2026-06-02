import numpy as np

from envs.counter_uav_env import CounterUAVEnv, make_default_config


def test_env_step_returns_gymnasium_tuple() -> None:
    env = CounterUAVEnv(make_default_config())
    env.reset(seed=1)
    actions = np.zeros((env.config.num_defenders, 2), dtype=np.float32)
    obs, rewards, terminations, truncations, info = env.step(actions)
    assert set(obs) == set(env.defense_agents)
    assert set(rewards) == set(env.defense_agents)
    assert isinstance(terminations["__all__"], bool)
    assert isinstance(truncations["__all__"], bool)
    assert info[env.defense_agents[0]]["step_count"] == 1


def test_step_keeps_positions_in_bounds() -> None:
    env = CounterUAVEnv(make_default_config())
    env.reset(seed=1)
    actions = np.ones((env.config.num_defenders, 2), dtype=np.float32)
    for _ in range(5):
        env.step(actions)
    assert np.all(env.defender_positions >= 0.0)
    assert np.all(env.defender_positions <= env.config.world_size)
    assert np.all(env.intruder_positions >= 0.0)
    assert np.all(env.intruder_positions <= env.config.world_size)


def test_global_state_shape_is_stable() -> None:
    env = CounterUAVEnv(make_default_config())
    env.reset(seed=1)
    first_shape = env.get_global_state().shape
    env.step(np.zeros((env.config.num_defenders, 2), dtype=np.float32))
    assert env.get_global_state().shape == first_shape


def test_available_agents_are_defenders() -> None:
    env = CounterUAVEnv(make_default_config())
    env.reset(seed=1)
    assert env.get_available_agents() == env.defense_agents
