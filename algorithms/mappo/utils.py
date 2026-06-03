from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def clip_gradients(module: nn.Module, max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm).item())


def linear_lr_schedule(
    optimizer: Optimizer,
    initial_lr: float,
    progress_remaining: float,
    min_lr: float = 0.0,
) -> float:
    lr = max(initial_lr * max(progress_remaining, 0.0), min_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@dataclass
class RunningMeanStd:
    shape: tuple[int, ...]
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = self.epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.size == 0:
            return
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, batch: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (batch - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count
