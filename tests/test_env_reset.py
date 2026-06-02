from envs.counter_uav_env import CounterUAVEnv, make_default_config


def test_env_reset_observation_shape() -> None:
    env = CounterUAVEnv(make_default_config())
    obs, info = env.reset(seed=1)
    assert set(obs) == set(env.defense_agents)
    assert all(agent_obs.shape == env.observation_spaces[agent].shape for agent, agent_obs in obs.items())
    assert info[env.defense_agents[0]]["step_count"] == 0


def test_seeded_reset_is_reproducible() -> None:
    env = CounterUAVEnv(make_default_config())
    first_obs, _ = env.reset(seed=7)
    second_obs, _ = env.reset(seed=7)
    assert all((first_obs[agent] == second_obs[agent]).all() for agent in env.defense_agents)
