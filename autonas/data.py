"""Deterministic synthetic classification tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Split:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def in_dim(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def n_classes(self) -> int:
        return int(self.y_train.max().item()) + 1


def spirals(n_per_class: int = 300, n_classes: int = 3, noise: float = 0.2, turns: float = 1.5, seed: int = 0) -> Split:
    """Interleaved spirals in 2-D, split 50/25/25 into train/val/test."""
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(n_classes):
        r = torch.linspace(0.05, 1.0, n_per_class)
        t = r * turns * 2 * torch.pi + 2 * torch.pi * c / n_classes + noise * torch.randn(n_per_class, generator=g)
        xs.append(torch.stack([r * torch.cos(t), r * torch.sin(t)], 1))
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n = len(y)
    a, b = n // 2, 3 * n // 4
    return Split(x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:])
