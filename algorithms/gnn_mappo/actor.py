from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class GNNActor(nn.Module):
    def __init__(self, obs_dim: int, graph_embedding_dim: int, action_dim: int, hidden_dim: int = 128, log_std_init: float = -0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + graph_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, local_observation: torch.Tensor, agent_graph_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([local_observation, agent_graph_embedding], dim=-1)
        mean = torch.tanh(self.net(x))
        log_std = torch.clamp(self.log_std, -5.0, 2.0).expand_as(mean)
        return mean, log_std

    def sample(self, local_observation: torch.Tensor, agent_graph_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(local_observation, agent_graph_embedding)
        dist = Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, local_observation: torch.Tensor, agent_graph_embedding: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = torch.clamp(actions, -0.999999, 0.999999)
        raw = torch.atanh(clipped)
        mean, log_std = self(local_observation, agent_graph_embedding)
        dist = Normal(mean, log_std.exp())
        log_prob = (dist.log_prob(raw) - torch.log(torch.clamp(1.0 - clipped.pow(2), min=1e-6))).sum(dim=-1)
        return log_prob, dist.entropy().sum(dim=-1)
