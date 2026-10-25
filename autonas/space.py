"""Cell search space.

A cell is a DAG over ``n_nodes`` nodes; node 0 is the cell input and every
node j>0 sums an operation applied to each predecessor i<j.  The cell output
is the last node.  A *genotype* assigns one operation index to every edge.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import nn

OPS: tuple[str, ...] = ("zero", "skip", "linear", "relu_linear", "tanh_linear")


class Zero(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.0


def make_op(name: str, width: int) -> nn.Module:
    if name == "zero":
        return Zero()
    if name == "skip":
        return nn.Identity()
    if name == "linear":
        return nn.Linear(width, width)
    if name == "relu_linear":
        return nn.Sequential(nn.ReLU(), nn.Linear(width, width))
    if name == "tanh_linear":
        return nn.Sequential(nn.Tanh(), nn.Linear(width, width))
    raise ValueError(f"unknown op {name!r}; choose from {OPS}")


@dataclass(frozen=True)
class SearchSpace:
    n_nodes: int = 3
    ops: tuple[str, ...] = OPS

    def __post_init__(self) -> None:
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be >= 2")

    @property
    def edges(self) -> list[tuple[int, int]]:
        return [(i, j) for j in range(1, self.n_nodes) for i in range(j)]

    @property
    def size(self) -> int:
        return len(self.ops) ** len(self.edges)

    def all_genotypes(self) -> Iterator[Genotype]:
        for ops in itertools.product(range(len(self.ops)), repeat=len(self.edges)):
            yield Genotype(ops)

    def random(self, rng: torch.Generator | None = None) -> Genotype:
        idx = torch.randint(len(self.ops), (len(self.edges),), generator=rng)
        return Genotype(tuple(int(i) for i in idx))

    def neighbours(self, g: Genotype) -> list[Genotype]:
        out = []
        for e in range(len(self.edges)):
            for o in range(len(self.ops)):
                if o != g.ops[e]:
                    out.append(g.with_edge(e, o))
        return out

    def describe(self, g: Genotype) -> str:
        return ", ".join(f"{i}->{j}:{self.ops[o]}" for (i, j), o in zip(self.edges, g.ops, strict=True))


@dataclass(frozen=True, order=True)
class Genotype:
    ops: tuple[int, ...]

    def with_edge(self, edge: int, op: int) -> Genotype:
        lst = list(self.ops)
        lst[edge] = op
        return Genotype(tuple(lst))

    def one_hot(self, n_ops: int) -> torch.Tensor:
        v = torch.zeros(len(self.ops) * n_ops)
        for e, o in enumerate(self.ops):
            v[e * n_ops + o] = 1.0
        return v


class Cell(nn.Module):
    def __init__(self, space: SearchSpace, genotype: Genotype, width: int) -> None:
        super().__init__()
        if len(genotype.ops) != len(space.edges):
            raise ValueError("genotype length does not match search space")
        self.space = space
        self.edge_ops = nn.ModuleList(make_op(space.ops[o], width) for o in genotype.ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = [x]
        k = 0
        for j in range(1, self.space.n_nodes):
            acc = torch.zeros_like(x)
            for i in range(j):
                acc = acc + self.edge_ops[k](nodes[i])
                k += 1
            nodes.append(acc)
        return nodes[-1]


def build_network(space: SearchSpace, genotype: Genotype, in_dim: int, n_classes: int, width: int = 32) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, width), Cell(space, genotype, width), nn.ReLU(), nn.Linear(width, n_classes))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


OpFactory = Callable[[str, int], nn.Module]
