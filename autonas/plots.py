"""Render the README hero figure from a ``autonas bench --json`` result (matplotlib optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_bench(result: dict[str, Any], path: str | Path) -> Path:
    """Three panels: test regret and P(found best) vs budget per strategy, surrogate Spearman vs data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = {"random": "random", "evolution": "regularised evolution", "local": "local search"}
    for name, per_budget in result["strategies"].items():
        budgets = sorted(per_budget, key=int)
        xs = [int(b) for b in budgets]
        axes[0].semilogy(xs, [per_budget[b]["mean_test_regret"] for b in budgets], "o-", label=labels.get(name, name))
        axes[1].plot(xs, [per_budget[b]["p_found_best_val"] for b in budgets], "o-", label=labels.get(name, name))
    darts = result.get("darts") or []
    if darts:
        mean = sum(d["test_regret"] for d in darts) / len(darts)
        axes[0].axhline(mean, color="grey", ls="--", label=f"DARTS (mean of {len(darts)} seeds)")
    n = result["table"]["archs"]
    axes[0].set(
        xlabel="architectures evaluated", ylabel="mean test regret", title=f"Search on exhaustive table ({n} archs)"
    )
    axes[1].set(xlabel="architectures evaluated", ylabel="P(found best-val arch)", ylim=(0, 1), title="Hit rate")
    sur = result["surrogate"]
    ks = sorted(sur, key=int)
    axes[2].plot([int(k) for k in ks], [sur[k] for k in ks], "s-", color="tab:purple")
    axes[2].set(xlabel="training architectures", ylabel="Spearman rho (held-out)", ylim=(0, 1), title="MLP surrogate")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
