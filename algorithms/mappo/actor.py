from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class ActionDistribution:
    mean: torch.Tensor
    log_std: torch.Tensor
    std: torch.Tensor


class MLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std_init: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, local_observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.net(local_observation))
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        return mean, log_std.expand_as(mean)

    def distribution(self, local_observation: torch.Tensor) -> Normal:
        mean, log_std = self(local_observation)
        return Normal(mean, log_std.exp())

    def sample(self, local_observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(local_observation)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def deterministic(self, local_observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self(local_observation)
        return torch.clamp(mean, -1.0, 1.0)

    def evaluate_actions(self, local_observation: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(clipped_actions)
        dist = self.distribution(local_observation)
        log_prob = self._squashed_log_prob(dist, raw_actions, clipped_actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def _squashed_log_prob(self, dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)
