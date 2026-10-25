"""Seeded training of a single genotype."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from autonas.data import Split
from autonas.space import Genotype, SearchSpace, build_network, count_params


@dataclass(frozen=True)
class TrainConfig:
    width: int = 32
    steps: int = 300
    lr: float = 1e-2
    seed: int = 0


def accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean().item())


def train_genotype(space: SearchSpace, g: Genotype, data: Split, cfg: TrainConfig = TrainConfig()) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    model = build_network(space, g, data.in_dim, data.n_classes, cfg.width)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.steps):  # full batch: the dataset is tiny
        loss = F.cross_entropy(model(data.x_train), data.y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return {
        "train_loss": float(loss.item()),
        "val_acc": accuracy(model, data.x_val, data.y_val),
        "test_acc": accuracy(model, data.x_test, data.y_test),
        "params": float(count_params(model)),
    }
