"""
graph.py
========
DAG validations and topological sort for neural architecture search graphs.
"""

from __future__ import annotations

from typing import Dict, List, Set


class NASGraph:
    """Represents a neural cell graph structure to validate DAG properties."""

    def __init__(self, n_nodes: int = 5, n_ops: int = 3) -> None:
        self.n_nodes = n_nodes
        self.n_ops = n_ops
        # Direct acyclic adjacency list (forward edges)
        # Standard feedforward structure: Node i can connect to Node j if i < j
        self.adj: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        self._init_default_edges()

    def _init_default_edges(self) -> None:
        # Standard Feedforward sequence: 0 -> 1 -> 2 -> 3 -> 4
        for i in range(self.n_nodes - 1):
            self.adj[i].append(i + 1)

    def is_acyclic(self) -> bool:
        """Returns True if the graph contains no cycles (DAG)."""
        visited = [0] * self.n_nodes  # 0 = unvisited, 1 = visiting, 2 = visited

        def dfs(node: int) -> bool:
            visited[node] = 1
            for neighbor in self.adj.get(node, []):
                if visited[neighbor] == 1:
                    return False  # Cycle detected
                elif visited[neighbor] == 0:
                    if not dfs(neighbor):
                        return False
            visited[node] = 2
            return True

        for i in range(self.n_nodes):
            if visited[i] == 0:
                if not dfs(i):
                    return False
        return True

    def topological_sort(self) -> List[int]:
        """Returns a topologically sorted list of nodes."""
        visited = set()
        stack = []

        def dfs(node: int) -> None:
            visited.add(node)
            for neighbor in self.adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node)

        for i in range(self.n_nodes):
            if i not in visited:
                dfs(i)

        return stack[::-1]
