from dataclasses import replace

import numpy as np

from envs.counter_uav_env import CounterUAVEnv, make_default_config


def test_comm_radius_builds_adjacency() -> None:
    config = replace(make_default_config(), num_defenders=2, num_intruders=1, comm_radius=10.0)
    env = CounterUAVEnv(config)
    env.reset(seed=1)
    env.defender_positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    env._update_communication_state()
    assert env.comm_adj[0, 1] == 1.0
    env.defender_positions = np.array([[0.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    env._update_communication_state()
    assert env.comm_adj[0, 1] == 0.0


def test_packet_loss_can_remove_all_edges() -> None:
    config = replace(make_default_config(), num_defenders=2, num_intruders=1, comm_radius=100.0, packet_loss_prob=1.0)
    env = CounterUAVEnv(config)
    env.reset(seed=1)
    env.defender_positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    env._update_communication_state()
    assert np.sum(env.comm_adj) == 0.0


def test_agent_dropout_blocks_communication() -> None:
    config = replace(
        make_default_config(),
        num_defenders=2,
        num_intruders=1,
        comm_radius=100.0,
        agent_dropout_prob=1.0,
        agent_dropout_duration_steps=2,
    )
    env = CounterUAVEnv(config)
    env.reset(seed=1)
    assert np.sum(env.comm_adj) == 0.0
    assert np.all(env.dropout_remaining > 0)


def test_partial_observation_keeps_shape_with_no_visible_intruders() -> None:
    config = replace(
        make_default_config(),
        num_defenders=1,
        num_intruders=2,
        partial_observation=True,
        observation_radius=1.0,
    )
    env = CounterUAVEnv(config)
    obs, _ = env.reset(seed=1)
    env.intruder_positions = np.array([[1000.0, 1000.0], [900.0, 900.0]], dtype=np.float32)
    agent_obs = env.get_observation(env.defense_agents[0])
    assert agent_obs.shape == next(iter(obs.values())).shape


def test_comm_delay_uses_history_buffer() -> None:
    config = replace(make_default_config(), num_defenders=2, num_intruders=1, comm_delay_steps=1)
    env = CounterUAVEnv(config)
    env.reset(seed=1)
    initial_len = len(env.observation_history)
    env.step(np.zeros((env.config.num_defenders, 2), dtype=np.float32))
    assert len(env.observation_history) >= initial_len
    assert "defender_positions" in env._delayed_observation_snapshot()
