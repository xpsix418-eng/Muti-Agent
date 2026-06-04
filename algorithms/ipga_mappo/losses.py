from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IPGALossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    assignment_loss: torch.Tensor


def assignment_loss_weight(
    global_step: int,
    start: float = 0.04,
    end: float = 0.0,
    decay_steps: int = 1_000_000,
) -> float:
    progress = min(max(global_step, 0) / max(decay_steps, 1), 1.0)
    return float(start + (end - start) * progress)


def clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()


def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    value_clip: float,
) -> torch.Tensor:
    clipped = old_values + torch.clamp(values - old_values, -value_clip, value_clip)
    value_loss_unclipped = (values - returns).pow(2)
    value_loss_clipped = (clipped - returns).pow(2)
    return 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()


def total_ipga_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    entropy: torch.Tensor,
    assignment_aux_loss: torch.Tensor,
    value_coef: float,
    entropy_coef: float,
    lambda_assign: float,
) -> IPGALossOutput:
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy + lambda_assign * assignment_aux_loss
    return IPGALossOutput(total, policy_loss, value_loss, entropy, assignment_aux_loss)
