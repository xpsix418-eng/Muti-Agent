import numpy as np
import torch

from algorithms.mappo.actor import MLPActor
from algorithms.mappo.buffer import RolloutBuffer
from algorithms.mappo.critic import MLPCritic


def test_mappo_actor_outputs_bounded_actions() -> None:
    actor = MLPActor(obs_dim=5, action_dim=2, hidden_dim=16)
    obs = torch.zeros((4, 5))
    actions, log_probs, entropy = actor.sample(obs)
    assert actions.shape == (4, 2)
    assert log_probs.shape == (4,)
    assert entropy.shape == (4,)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)


def test_mappo_critic_outputs_values() -> None:
    critic = MLPCritic(state_dim=7, hidden_dim=16)
    values = critic(torch.zeros((3, 7)))
    assert values.shape == (3,)


def test_rollout_buffer_computes_gae_and_batches() -> None:
    buffer = RolloutBuffer(
        rollout_length=4,
        num_agents=2,
        obs_dim=3,
        state_dim=5,
        action_dim=2,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    )
    for _ in range(4):
        buffer.add(
            observations=np.zeros((2, 3), dtype=np.float32),
            global_states=np.zeros((2, 5), dtype=np.float32),
            actions=np.zeros((2, 2), dtype=np.float32),
            log_probs=np.zeros(2, dtype=np.float32),
            rewards=np.ones(2, dtype=np.float32),
            dones=np.zeros(2, dtype=np.float32),
            values=np.zeros(2, dtype=np.float32),
        )
    buffer.compute_returns_and_advantages(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    batches = list(buffer.mini_batches(batch_size=4))
    assert len(batches) == 2
    assert np.all(buffer.advantages > 0.0)
