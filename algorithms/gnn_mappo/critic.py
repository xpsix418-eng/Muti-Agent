from __future__ import annotations

import torch
from torch import nn


class GNNCritic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, encoded_nodes: torch.Tensor) -> torch.Tensor:
        pooled = encoded_nodes.mean(dim=-2)
        return self.value(pooled).squeeze(-1)
