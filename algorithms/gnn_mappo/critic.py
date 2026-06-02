from __future__ import annotations

import torch
from torch import nn


class GNNCritic(nn.Module):
    def __init__(self, state_dim: int, graph_embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim + graph_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor, global_graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.value(torch.cat([global_state, global_graph_embedding], dim=-1)).squeeze(-1)
