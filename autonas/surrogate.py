"""Performance predictor evaluated by rank correlation on held-out architectures."""

from __future__ import annotations

import torch

from autonas.table import TabularBenchmark
from nas_graph.predictor import ArchPredictor


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties share the mean rank)."""
    order = torch.argsort(x)
    r = torch.empty_like(x)
    r[order] = torch.arange(len(x), dtype=x.dtype)
    for v in torch.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _ranks(a.double()), _ranks(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb / (ra.norm() * rb.norm() + 1e-12)).item())


def surrogate_rank_correlation(tb: TabularBenchmark, n_train: int, seed: int = 0, epochs: int = 300) -> float:
    gs = sorted(tb.val)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(gs), generator=g).tolist()
    tr, te = [gs[i] for i in perm[:n_train]], [gs[i] for i in perm[n_train:]]
    n_ops = len(tb.space.ops)
    X = torch.stack([x.one_hot(n_ops) for x in tr])
    y = torch.tensor([[tb.val[x]] for x in tr])
    torch.manual_seed(seed)
    model = ArchPredictor(X.shape[1], 64)
    model.train_epochs(X, y, n_epochs=epochs, lr=1e-2)
    pred = model.predict(torch.stack([x.one_hot(n_ops) for x in te])).squeeze(1)
    return spearman(pred, torch.tensor([tb.val[x] for x in te]))
