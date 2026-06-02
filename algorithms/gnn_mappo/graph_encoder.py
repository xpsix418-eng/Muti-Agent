from __future__ import annotations

import torch
from torch import nn


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.node_mlp = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        messages = adjacency @ self.node_mlp(node_features)
        return torch.relu(messages)
