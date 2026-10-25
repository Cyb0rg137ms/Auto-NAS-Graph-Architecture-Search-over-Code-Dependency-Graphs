"""Black-box search strategies over a query function (higher is better).

Each returns the incumbent trajectory: ``trace[k]`` is the best genotype
after ``k+1`` evaluations, so anytime regret curves can be computed.
"""

from __future__ import annotations

import collections
from collections.abc import Callable

import torch

from autonas.space import Genotype, SearchSpace

Query = Callable[[Genotype], float]


def _track(trace: list[Genotype], best: list, g: Genotype, v: float) -> None:
    if best[0] is None or v > best[1]:
        best[0], best[1] = g, v
    trace.append(best[0])


def random_search(space: SearchSpace, query: Query, budget: int, seed: int = 0) -> list[Genotype]:
    rng = torch.Generator().manual_seed(seed)
    trace: list[Genotype] = []
    best: list = [None, float("-inf")]
    for _ in range(budget):
        g = space.random(rng)
        _track(trace, best, g, query(g))
    return trace


def regularized_evolution(
    space: SearchSpace, query: Query, budget: int, population: int = 10, sample: int = 3, seed: int = 0
) -> list[Genotype]:
    """Aging evolution (Real et al., AAAI 2019): tournament selection, one-edge mutation, oldest removed."""
    if sample > population:
        raise ValueError("sample must be <= population")
    rng = torch.Generator().manual_seed(seed)
    pop: collections.deque[tuple[Genotype, float]] = collections.deque()
    trace: list[Genotype] = []
    best: list = [None, float("-inf")]
    while len(trace) < budget:
        if len(pop) < population:
            g = space.random(rng)
        else:
            idx = torch.randperm(len(pop), generator=rng)[:sample].tolist()
            parent = max((pop[i] for i in idx), key=lambda t: t[1])[0]
            e = int(torch.randint(len(space.edges), (1,), generator=rng))
            o = int(torch.randint(len(space.ops) - 1, (1,), generator=rng))
            o = o if o < parent.ops[e] else o + 1  # guaranteed different op
            g = parent.with_edge(e, o)
        v = query(g)
        pop.append((g, v))
        if len(pop) > population:
            pop.popleft()
        _track(trace, best, g, v)
    return trace


def local_search(space: SearchSpace, query: Query, budget: int, seed: int = 0) -> list[Genotype]:
    """First-improvement hill climbing with random restarts."""
    rng = torch.Generator().manual_seed(seed)
    trace: list[Genotype] = []
    best: list = [None, float("-inf")]
    while len(trace) < budget:
        cur = space.random(rng)
        cur_v = query(cur)
        _track(trace, best, cur, cur_v)
        improved = True
        while improved and len(trace) < budget:
            improved = False
            nbrs = space.neighbours(cur)
            for i in torch.randperm(len(nbrs), generator=rng).tolist():
                if len(trace) >= budget:
                    break
                v = query(nbrs[i])
                _track(trace, best, nbrs[i], v)
                if v > cur_v:
                    cur, cur_v, improved = nbrs[i], v, True
                    break
    return trace
