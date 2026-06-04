from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class IPGAActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        graph_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + graph_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(
        self,
        local_observation: torch.Tensor,
        defender_graph_embedding: torch.Tensor,
        assigned_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([local_observation, defender_graph_embedding, assigned_context], dim=-1)
        mean = torch.tanh(self.net(features))
        log_std = torch.clamp(self.log_std, -5.0, 1.0).expand_as(mean)
        return mean, log_std

    def distribution(
        self,
        local_observation: torch.Tensor,
        defender_graph_embedding: torch.Tensor,
        assigned_context: torch.Tensor,
    ) -> Normal:
        mean, log_std = self(local_observation, defender_graph_embedding, assigned_context)
        return Normal(mean, log_std.exp())

    def sample(
        self,
        local_observation: torch.Tensor,
        defender_graph_embedding: torch.Tensor,
        assigned_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(local_observation, defender_graph_embedding, assigned_context)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def deterministic(
        self,
        local_observation: torch.Tensor,
        defender_graph_embedding: torch.Tensor,
        assigned_context: torch.Tensor,
    ) -> torch.Tensor:
        mean, _ = self(local_observation, defender_graph_embedding, assigned_context)
        return torch.clamp(mean, -1.0, 1.0)

    def evaluate_actions(
        self,
        local_observation: torch.Tensor,
        defender_graph_embedding: torch.Tensor,
        assigned_context: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(clipped_actions)
        dist = self.distribution(local_observation, defender_graph_embedding, assigned_context)
        log_prob = self._squashed_log_prob(dist, raw_actions, clipped_actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def _squashed_log_prob(self, dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)
