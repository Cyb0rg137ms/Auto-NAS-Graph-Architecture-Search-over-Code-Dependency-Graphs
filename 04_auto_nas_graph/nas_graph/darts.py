"""
darts.py
========
Differentiable Architecture Search (DARTS) cell search space.

References:
  - Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019.
    https://arxiv.org/abs/1806.09055
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class MixedOp(nn.Module):
    """A mixed operation combining multiple candidate operations via softmax weights."""

    def __init__(self, in_channels: int, out_channels: int, n_ops: int) -> None:
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        ])
        assert len(self.ops) == n_ops

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted sum of operation outputs.

        Args:
            x: Input feature representation.
            weights: Softmax weights over candidate operations, shape (n_ops,).
        """
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class DARTSCell(nn.Module):
    """
    A single DARTS cell containing mixed operations between nodes.

    We use a simplified 4-node DAG cell:
      - Node 0: Input node
      - Node 1: Intermediate node (mixed op from 0)
      - Node 2: Intermediate node (mixed ops from 0 and 1)
      - Node 3: Output node (concatenation/sum of intermediate nodes)
    """

    def __init__(self, in_channels: int, out_channels: int, n_ops: int = 4) -> None:
        super().__init__()
        self.n_ops = n_ops

        # Architecture weights (alpha) parameters for intermediate edges
        self.arch_weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_ops) * 1e-3, requires_grad=True)
            for _ in range(3)  # Edges: 0->1, 0->2, 1->2
        ])

        # Operations on edges
        self.op_0_1 = MixedOp(in_channels, out_channels, n_ops)
        self.op_0_2 = MixedOp(in_channels, out_channels, n_ops)
        self.op_1_2 = MixedOp(in_channels, out_channels, n_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax over edge operations
        w_0_1 = F.softmax(self.arch_weights[0], dim=-1)
        w_0_2 = F.softmax(self.arch_weights[1], dim=-1)
        w_1_2 = F.softmax(self.arch_weights[2], dim=-1)

        # Node 0 is input x
        node_0 = x

        # Node 1 = op_0_1(node_0)
        node_1 = self.op_0_1(node_0, w_0_1)

        # Node 2 = op_0_2(node_0) + op_1_2(node_1)
        node_2 = self.op_0_2(node_0, w_0_2) + self.op_1_2(node_1, w_1_2)

        # Output is node 2
        return node_2

    def discretise(self) -> int:
        """Returns the index of the most probable operation on the 0->1 edge."""
        with torch.no_grad():
            return int(self.arch_weights[0].argmax().item())
