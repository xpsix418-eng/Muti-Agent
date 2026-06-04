from __future__ import annotations

import torch
from torch import nn


class IPGACritic(nn.Module):
    def __init__(self, state_dim: int, graph_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + graph_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor, pooled_graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([global_state, pooled_graph_embedding], dim=-1)).squeeze(-1)
