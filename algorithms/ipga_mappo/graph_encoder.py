from __future__ import annotations

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = node_embeddings.shape
        src = edge_index[0].long()
        dst = edge_index[1].long()
        q = self.query(node_embeddings).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.key(node_embeddings).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.value(node_embeddings).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        edge_bias = self.edge_proj(edge_features).view(batch_size, -1, self.num_heads, self.head_dim)
        scores = ((q[:, dst] * (k[:, src] + edge_bias)).sum(dim=-1)) / (self.head_dim**0.5)
        attention = torch.zeros_like(scores)
        for node_idx in range(num_nodes):
            mask = dst == node_idx
            if torch.any(mask):
                attention[:, mask] = torch.softmax(scores[:, mask], dim=1)
        messages = attention.unsqueeze(-1) * (v[:, src] + edge_bias)
        aggregated = torch.zeros(batch_size, num_nodes, self.num_heads, self.head_dim, device=node_embeddings.device)
        for edge_pos in range(src.numel()):
            aggregated[:, dst[edge_pos]] += messages[:, edge_pos]
        aggregated = aggregated.reshape(batch_size, num_nodes, self.hidden_dim)
        return self.norm(node_embeddings + self.out(torch.relu(aggregated))), attention.mean(dim=-1)


class InterceptionGraphEncoder(nn.Module):
    """Lightweight graph attention encoder with optional PyG-free operation."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim, edge_dim, attention_heads) for _ in range(num_layers)]
        )
        self.last_attention: torch.Tensor | None = None
        self.has_torch_geometric = self._detect_torch_geometric()

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if node_features.ndim == 2:
            node_features = node_features.unsqueeze(0)
        if edge_features.ndim == 2:
            edge_features = edge_features.unsqueeze(0)
        embeddings = self.node_proj(node_features)
        attention = torch.zeros(
            embeddings.shape[0],
            edge_index.shape[1],
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        for layer in self.layers:
            embeddings, attention = layer(embeddings, edge_index, edge_features)
        self.last_attention = attention.detach()
        pooled = embeddings.mean(dim=1)
        return embeddings, pooled, attention

    @staticmethod
    def _detect_torch_geometric() -> bool:
        try:
            import torch_geometric  # noqa: F401
        except Exception:
            return False
        return True
