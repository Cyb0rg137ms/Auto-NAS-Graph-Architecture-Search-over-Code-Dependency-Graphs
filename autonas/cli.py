"""Command line: ``autonas bench | mine | search``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _mine(a: argparse.Namespace) -> int:
    from nas_graph import ArchitectureSearchEngine, CodebaseASTParser, ModelPerformanceEvaluator

    root = Path(a.scan_dir)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 1
    p = CodebaseASTParser(str(root))
    p.scan()
    summary = p.get_summary_metrics()
    plan = ArchitectureSearchEngine(summary).search(a.max_latency, a.max_params)
    ev = ModelPerformanceEvaluator(plan["layers"], input_dim=a.input_dim)
    out = {"summary": summary, "plan": plan, "profile": ev.profile()}
    if a.output_code:
        Path(a.output_code).write_text(ev.generate_pytorch_stub(), encoding="utf-8")
        out["written"] = a.output_code
    print(json.dumps(out, indent=2))
    return 0


def _bench(a: argparse.Namespace) -> int:
    from autonas.benchmarks import run, to_markdown

    r = run(quick=a.quick, table_cache=a.table)
    print(to_markdown(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, default=str))
    if a.plot:
        from autonas.plots import plot_bench

        plot_bench(r, a.plot)
    return 0


def _search(a: argparse.Namespace) -> int:
    from autonas import DartsConfig, SearchSpace, darts_search, spirals, train_genotype

    data, space = spirals(), SearchSpace(a.nodes)
    out = darts_search(space, data, DartsConfig(steps=a.steps, seed=a.seed))
    g = out["genotype"]
    print(json.dumps({"genotype": space.describe(g), "retrained": train_genotype(space, g, data)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autonas")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="run the measured benchmark suite")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--json")
    b.add_argument("--plot", help="write the summary figure (PNG) here; needs matplotlib")
    b.add_argument("--table", help="cache file for the tabular benchmark")
    b.set_defaults(fn=_bench)
    m = sub.add_parser("mine", help="v1: scan a code base and emit a rule-based PyTorch stub")
    m.add_argument("scan_dir", nargs="?", default=".")
    m.add_argument("--max_latency", type=float, default=5.0)
    m.add_argument("--max_params", type=float, default=2.5)
    m.add_argument("--input_dim", type=int, default=128)
    m.add_argument("--output_code")
    m.set_defaults(fn=_mine)
    s = sub.add_parser("search", help="DARTS on the spirals task, then retrain the found cell")
    s.add_argument("--nodes", type=int, default=3)
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_search)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
