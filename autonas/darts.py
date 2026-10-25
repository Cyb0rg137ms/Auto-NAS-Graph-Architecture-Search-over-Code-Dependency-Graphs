"""DARTS (Liu et al., ICLR 2019) on the cell space, first-order approximation.

Every edge holds all candidate ops mixed by ``softmax(alpha_edge)``.
Network weights ``w`` are updated on the training split and architecture
logits ``alpha`` on the validation split, alternately (first-order bilevel).
The discrete genotype is the per-edge argmax of ``alpha``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from autonas.data import Split
from autonas.space import Genotype, SearchSpace, make_op


class MixedEdge(nn.Module):
    def __init__(self, ops: tuple[str, ...], width: int) -> None:
        super().__init__()
        self.ops = nn.ModuleList(make_op(o, width) for o in ops)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        out = torch.stack([op(x) for op in self.ops])  # (n_ops, B, W)
        return torch.einsum("o,obw->bw", weights, out)


class Supernet(nn.Module):
    def __init__(self, space: SearchSpace, in_dim: int, n_classes: int, width: int) -> None:
        super().__init__()
        self.space = space
        self.stem = nn.Linear(in_dim, width)
        self.edges = nn.ModuleList(MixedEdge(space.ops, width) for _ in space.edges)
        self.head = nn.Linear(width, n_classes)
        self.alpha = nn.Parameter(1e-3 * torch.randn(len(space.edges), len(space.ops)))

    def weights(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n != "alpha"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.alpha, dim=-1)
        nodes = [self.stem(x)]
        k = 0
        for j in range(1, self.space.n_nodes):
            acc = torch.zeros_like(nodes[0])
            for i in range(j):
                acc = acc + self.edges[k](nodes[i], w[k])
                k += 1
            nodes.append(acc)
        return self.head(F.relu(nodes[-1]))

    def genotype(self) -> Genotype:
        return Genotype(tuple(int(i) for i in self.alpha.argmax(-1)))


@dataclass(frozen=True)
class DartsConfig:
    width: int = 32
    steps: int = 300
    w_lr: float = 1e-2
    alpha_lr: float = 3e-2
    seed: int = 0


def darts_search(space: SearchSpace, data: Split, cfg: DartsConfig = DartsConfig()) -> dict:
    torch.manual_seed(cfg.seed)
    net = Supernet(space, data.in_dim, data.n_classes, cfg.width)
    w_opt = torch.optim.Adam(net.weights(), lr=cfg.w_lr)
    a_opt = torch.optim.Adam([net.alpha], lr=cfg.alpha_lr, betas=(0.5, 0.999))
    t0 = time.perf_counter()
    for _ in range(cfg.steps):
        a_opt.zero_grad()
        F.cross_entropy(net(data.x_val), data.y_val).backward()
        a_opt.step()
        w_opt.zero_grad()
        F.cross_entropy(net(data.x_train), data.y_train).backward()
        w_opt.step()
    return {
        "genotype": net.genotype(),
        "alpha_softmax": F.softmax(net.alpha.detach(), -1),
        "seconds": time.perf_counter() - t0,
    }
