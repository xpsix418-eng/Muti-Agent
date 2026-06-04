from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SoftAssignmentGate(nn.Module):
    def __init__(self, graph_hidden_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(graph_hidden_dim * 3 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        defender_embeddings: torch.Tensor,
        intruder_embeddings: torch.Tensor,
        interception_point_embeddings: torch.Tensor,
        pair_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_defenders, hidden_dim = defender_embeddings.shape
        num_intruders = intruder_embeddings.shape[1]
        defender_expanded = defender_embeddings[:, :, None, :].expand(batch_size, num_defenders, num_intruders, hidden_dim)
        intruder_expanded = intruder_embeddings[:, None, :, :].expand(batch_size, num_defenders, num_intruders, hidden_dim)
        point_expanded = interception_point_embeddings[:, None, :, :].expand(batch_size, num_defenders, num_intruders, hidden_dim)
        score_input = torch.cat([defender_expanded, intruder_expanded, point_expanded, pair_edge_features], dim=-1)
        scores = self.score_net(score_input).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bdi,bih->bdh", weights, interception_point_embeddings)
        return weights, context

    def auxiliary_loss(self, assignment_weights: torch.Tensor, heuristic_assignments: torch.Tensor) -> torch.Tensor:
        batch_size, num_defenders, num_intruders = assignment_weights.shape
        targets = heuristic_assignments.reshape(batch_size, num_defenders).long()
        valid = (targets >= 0) & (targets < num_intruders)
        if not torch.any(valid):
            return assignment_weights.sum() * 0.0
        logits = torch.log(torch.clamp(assignment_weights, min=1e-8))
        return F.nll_loss(logits[valid], targets[valid], reduction="mean")

    @staticmethod
    def entropy(assignment_weights: torch.Tensor) -> torch.Tensor:
        return -(assignment_weights * torch.log(torch.clamp(assignment_weights, min=1e-8))).sum(dim=-1).mean()
