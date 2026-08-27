"""
predictor.py
============
Simple MLP architecture performance predictor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArchPredictor(nn.Module):
    """
    Predicts the performance (accuracy) of a discrete architecture vector.

    Used as a surrogate model to avoid expensive evaluations.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def train_epochs(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 50,
        lr: float = 0.01,
    ) -> float:
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(n_epochs):
            pred = self.forward(X)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        return float(loss.item())

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            return float(F.mse_loss(self.forward(X), y).item())
