"""DAG container with cycle check and topological sort.

v1's ``benchmark.py`` called ``add_node``/``add_edge``, which did not exist.
They are now implemented; nodes may be ints (default chain) or any hashable.
Topological sort is Kahn's algorithm (iterative, no recursion limit) and
raises on cycles instead of returning a meaningless order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable


class CycleError(ValueError):
    pass


class NASGraph:
    def __init__(self, n_nodes: int = 5, n_ops: int = 3, chain: bool = True) -> None:
        self.n_ops = n_ops
        self.adj: dict[Hashable, list[Hashable]] = {i: [] for i in range(n_nodes)}
        self.ops: dict[Hashable, str | None] = dict.fromkeys(range(n_nodes))
        if chain:
            for i in range(n_nodes - 1):
                self.adj[i].append(i + 1)

    @property
    def n_nodes(self) -> int:
        return len(self.adj)

    def add_node(self, name: Hashable, op: str | None = None) -> None:
        self.adj.setdefault(name, [])
        self.ops[name] = op

    def add_edge(self, u: Hashable, v: Hashable) -> None:
        for n in (u, v):
            if n not in self.adj:
                raise KeyError(f"unknown node {n!r}")
        if v not in self.adj[u]:
            self.adj[u].append(v)

    def _kahn(self) -> list[Hashable]:
        indeg = dict.fromkeys(self.adj, 0)
        for vs in self.adj.values():
            for v in vs:
                indeg[v] += 1
        q = deque(n for n, d in indeg.items() if d == 0)
        order = []
        while q:
            n = q.popleft()
            order.append(n)
            for v in self.adj[n]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return order

    def is_acyclic(self) -> bool:
        return len(self._kahn()) == len(self.adj)

    def topological_sort(self) -> list[Hashable]:
        order = self._kahn()
        if len(order) != len(self.adj):
            raise CycleError("graph has a cycle")
        return order
