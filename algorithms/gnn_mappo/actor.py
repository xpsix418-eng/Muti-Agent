from __future__ import annotations

import torch
from torch import nn


class GNNActor(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, encoded_nodes: torch.Tensor) -> torch.Tensor:
        return self.head(encoded_nodes)
