"""AST code-base miner: per-file class/function counts, McCabe cyclomatic
complexity, and a name-level dependency graph (inheritance + calls).

v2 fixes: files are keyed by relative path (v1 keyed by basename, so
``a/utils.py`` and ``b/utils.py`` overwrote each other); ``async def`` is
counted; method calls (``self.f()``/``mod.f()``) are recorded; syntax errors
are collected in ``errors`` instead of being silently swallowed; ``.git``,
``__pycache__``, ``venv`` and ``site-packages`` are skipped.
"""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "site-packages", ".tox", "build", "dist"}
_BRANCHES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.IfExp,
    ast.comprehension,
    ast.match_case,
)


def cyclomatic_complexity(fn: ast.AST) -> int:
    """McCabe complexity: 1 + decision points (boolean operators count once per extra operand)."""
    c = 1
    for node in ast.walk(fn):
        if isinstance(node, _BRANCHES):
            c += 1
        elif isinstance(node, ast.BoolOp):
            c += len(node.values) - 1
    return c


class CodebaseASTParser:
    """Parses Python code bases using ``ast`` to map dependencies and complexity."""

    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self.graph: defaultdict[str, set[str]] = defaultdict(set)
        self.metrics: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}

    def scan(self) -> dict[str, set[str]]:
        for path in sorted(self.root.rglob("*.py")):
            if SKIP_DIRS.intersection(path.relative_to(self.root).parts):
                continue
            self.parse_file(path)
        return self.graph

    def parse_file(self, filepath: Path) -> None:
        key = filepath.relative_to(self.root).as_posix() if filepath.is_relative_to(self.root) else str(filepath)
        try:
            tree = ast.parse(filepath.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError) as e:
            self.errors[key] = f"{type(e).__name__}: {e}"
            log.warning("skip %s: %s", key, e)
            return
        m: dict[str, Any] = {"classes": 0, "functions": 0, "complexity_score": 0, "cyclomatic": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                m["classes"] += 1
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        self.graph[node.name].add(base.id)
                    elif isinstance(base, ast.Attribute):
                        self.graph[node.name].add(base.attr)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                m["functions"] += 1
                m["complexity_score"] += len(node.body)  # v1 metric (statement count), kept for compatibility
                m["cyclomatic"].append(cyclomatic_complexity(node))
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            self.graph[node.name].add(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            self.graph[node.name].add(child.func.attr)
        self.metrics[key] = m

    def get_summary_metrics(self) -> dict[str, Any]:
        cc = [c for m in self.metrics.values() for c in m["cyclomatic"]]
        return {
            "total_files": len(self.metrics),
            "total_classes": sum(m["classes"] for m in self.metrics.values()),
            "total_functions": sum(m["functions"] for m in self.metrics.values()),
            "mean_complexity": sum(m["complexity_score"] for m in self.metrics.values()) / max(1, len(self.metrics)),
            "mean_cyclomatic": sum(cc) / max(1, len(cc)),
            "max_cyclomatic": max(cc, default=0),
            "graph_nodes": len(self.graph),
            "graph_edges": sum(len(v) for v in self.graph.values()),
            "parse_errors": len(self.errors),
        }
