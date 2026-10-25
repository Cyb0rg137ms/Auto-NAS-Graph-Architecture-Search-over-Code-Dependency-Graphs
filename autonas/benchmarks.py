"""Reproducible benchmark suite (`autonas bench`)."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch

from autonas.darts import DartsConfig, darts_search
from autonas.data import spirals
from autonas.space import SearchSpace
from autonas.strategies import local_search, random_search, regularized_evolution
from autonas.surrogate import surrogate_rank_correlation
from autonas.table import TabularBenchmark
from autonas.train import TrainConfig

TASK: dict[str, Any] = {"n_per_class": 200, "n_classes": 3, "noise": 0.3, "turns": 2.0, "seed": 0}


def get_table(cache: str | Path | None, quick: bool) -> TabularBenchmark:
    if cache and Path(cache).exists():
        return TabularBenchmark.load(cache)
    space = SearchSpace(n_nodes=3)
    cfg = TrainConfig(width=16 if quick else 32, steps=60 if quick else 200)
    tb = TabularBenchmark.build(space, spirals(**TASK), cfg, seeds=(0,) if quick else (0, 1))
    if cache:
        tb.save(cache)
    return tb


def run(quick: bool = False, table_cache: str | Path | None = None) -> dict[str, Any]:
    torch.set_num_threads(1)
    t_all = time.perf_counter()
    res: dict[str, Any] = {"task": TASK}
    tb = get_table(table_cache, quick)
    space = tb.space
    vals = sorted(tb.val.values())
    res["table"] = {
        "archs": len(tb.val),
        "build_s": tb.build_seconds,
        "val_min": vals[0],
        "val_median": statistics.median(vals),
        "val_max": vals[-1],
        "best": space.describe(tb.best_val),
        "best_test": tb.test[tb.best_val],
        "oracle_test": max(tb.test.values()),
    }
    seeds = range(5 if quick else 50)
    budgets = [5, 10, 20, 40]
    strat: dict[str, Any] = {"random": random_search, "evolution": regularized_evolution, "local": local_search}
    curves: dict[str, dict[int, dict[str, float]]] = {}
    for name, fn in strat.items():
        curves[name] = {}
        traces = [fn(space, tb.query, max(budgets), seed=s) for s in seeds]
        for b in budgets:
            inc = [t[b - 1] for t in traces]
            curves[name][b] = {
                "mean_test_regret": statistics.mean(tb.test_regret(g) for g in inc),
                "p_found_best_val": statistics.mean(float(tb.val[g] == tb.val[tb.best_val]) for g in inc),
            }
    res["strategies"] = curves
    data = spirals(**TASK)
    darts = []
    for s in range(2 if quick else 5):
        out = darts_search(space, data, DartsConfig(width=16 if quick else 32, steps=60 if quick else 200, seed=s))
        g = out["genotype"]
        darts.append(
            {
                "seed": s,
                "genotype": space.describe(g),
                "val": tb.val[g],
                "test_regret": tb.test_regret(g),
                "percentile": tb.val_percentile(g),
                "seconds": out["seconds"],
            }
        )
    res["darts"] = darts
    res["surrogate"] = {
        n: statistics.mean(surrogate_rank_correlation(tb, n, seed=s) for s in range(3)) for n in (10, 25, 50, 100)
    }
    res["total_s"] = time.perf_counter() - t_all
    return res


def to_markdown(r: dict[str, Any]) -> str:
    t = r["table"]
    out = [
        f"## Tabular benchmark ({t['archs']} archs, built in {t['build_s']:.1f} s)",
        f"val acc min/median/max = {t['val_min']:.4f} / {t['val_median']:.4f} / {t['val_max']:.4f}; "
        f"best-val arch `{t['best']}` test {t['best_test']:.4f} (oracle test {t['oracle_test']:.4f})",
        "",
        "## Search strategies (mean test regret | P(found best-val arch))",
        "| strategy | " + " | ".join(f"{b} evals" for b in next(iter(r["strategies"].values()))) + " |",
        "|---|" + "---|" * len(next(iter(r["strategies"].values()))),
    ]
    for name, c in r["strategies"].items():
        out.append(
            f"| {name} | "
            + " | ".join(f"{v['mean_test_regret']:.4f} / {v['p_found_best_val']:.2f}" for v in c.values())
            + " |"
        )
    out += [
        "",
        "## DARTS (first-order, one supernet run each)",
        "| seed | genotype | val | test regret | percentile | s |",
        "|---|---|---|---|---|---|",
    ]
    for d in r["darts"]:
        out.append(
            f"| {d['seed']} | {d['genotype']} | {d['val']:.4f} | {d['test_regret']:.4f} | "
            f"{d['percentile']:.2f} | {d['seconds']:.2f} |"
        )
    out += ["", "## Surrogate (MLP on one-hot) Spearman on held-out archs", "| train archs | Spearman |", "|---|---|"]
    out += [f"| {n} | {v:.3f} |" for n, v in r["surrogate"].items()]
    out.append(f"\ntotal {r['total_s']:.1f} s")
    return "\n".join(out)
