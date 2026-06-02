from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class MiniBatch:
    observations: torch.Tensor
    global_states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.rollout_length = rollout_length
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.reset()

    def reset(self) -> None:
        shape = (self.rollout_length, self.num_agents)
        self.observations = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self.global_states = np.zeros((*shape, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        global_states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        if self.position >= self.rollout_length:
            raise RuntimeError("RolloutBuffer overflow")
        self.observations[self.position] = observations
        self.global_states[self.position] = global_states
        self.actions[self.position] = actions
        self.log_probs[self.position] = log_probs
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        self.values[self.position] = values
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        gae = np.zeros(self.num_agents, dtype=np.float32)
        for step in reversed(range(self.position)):
            if step == self.position - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
        self.returns[: self.position] = self.advantages[: self.position] + self.values[: self.position]

    def mini_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[MiniBatch]:
        total = self.position * self.num_agents
        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)

        observations = self.observations[: self.position].reshape(total, self.obs_dim)
        global_states = self.global_states[: self.position].reshape(total, self.state_dim)
        actions = self.actions[: self.position].reshape(total, self.action_dim)
        log_probs = self.log_probs[: self.position].reshape(total)
        advantages = self.advantages[: self.position].reshape(total)
        returns = self.returns[: self.position].reshape(total)
        values = self.values[: self.position].reshape(total)

        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield MiniBatch(
                observations=self._tensor(observations[batch_idx]),
                global_states=self._tensor(global_states[batch_idx]),
                actions=self._tensor(actions[batch_idx]),
                old_log_probs=self._tensor(log_probs[batch_idx]),
                advantages=self._tensor(advantages[batch_idx]),
                returns=self._tensor(returns[batch_idx]),
                old_values=self._tensor(values[batch_idx]),
            )

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)
