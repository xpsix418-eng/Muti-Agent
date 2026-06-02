from __future__ import annotations

import importlib.util

import torch
from torch import nn


PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 128, message_passing_steps: int = 2):
        super().__init__()
        self.uses_pyg = False
        self.message_passing_steps = message_passing_steps
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        h = self.node_proj(node_features)
        mask = adjacency > 0.0
        for _ in range(self.message_passing_steps):
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)
            scores = torch.matmul(q, k.transpose(-1, -2)) / max(q.shape[-1] ** 0.5, 1e-6)
            scores = scores.masked_fill(~mask, -1e9)
            attn = torch.softmax(scores, dim=-1)
            messages = torch.matmul(attn, v)
            h = self.norm(h + self.out_proj(torch.cat([h, messages], dim=-1)))
        return h.squeeze(0) if squeeze else h

    def pool(self, encoded_nodes: torch.Tensor) -> torch.Tensor:
        return encoded_nodes.mean(dim=-2)
