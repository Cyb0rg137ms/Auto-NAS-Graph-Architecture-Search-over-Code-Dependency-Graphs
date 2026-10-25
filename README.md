# Auto-NAS-Graph: neural architecture search with measured results

A compact, fully tested NAS toolkit in PyTorch. It can **train every architecture** in a cell search space, which
gives exact ground truth, and then measure how well each search strategy finds the best one.

- **`autonas.space`**: DAG cell search space (5 ops per edge), genotypes, neighbours, network builder
- **`autonas.table`**: exhaustive *tabular benchmark* in the style of NAS-Bench-201, with a JSON cache
- **`autonas.strategies`**: random search, regularised (aging) evolution (Real et al., 2019), and local search
- **`autonas.darts`**: DARTS supernet with first-order bilevel optimisation (Liu et al., 2019)
- **`autonas.surrogate`**: MLP performance predictor, evaluated by Spearman rank correlation
- **`nas_graph`** (v1 API, kept and fixed): AST code-base miner with McCabe complexity and a call/inheritance graph, a
  rule-based layer-stack generator, and a parameter/FLOP evaluator that emits runnable PyTorch stubs

```bash
pip install -e ".[dev]"        # CPU torch is sufficient
autonas bench                  # every table below (31 s on 1 CPU thread)
autonas search                 # DARTS on spirals, then retrain the found cell
autonas mine path/to/repo --output_code model.py   # v1 behaviour
pytest                         # 34 tests
```

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Limitations: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Measured results (`autonas bench`, CPU, seeded)

The task is 3-class interleaved spirals with noise 0.3 and 2 turns: 600 points, split 50/25/25 into train/val/test.
The space is a 3-node cell with 3 edges and 5 ops, so **125 architectures**. Each one was trained for 200 Adam steps at
width 32, and its accuracy is the mean of 2 seeds. Building the table took 26 s.

- **Validation accuracy across the space:** min 0.340, median 0.690, max 0.960. The choice of architecture matters.
- **Best-val architecture:** `0->1:tanh_linear, 0->2:relu_linear, 1->2:tanh_linear`, with test accuracy 0.9467. The
  best test accuracy in the space (oracle) is 0.9567.

**Search strategies.** Each cell shows mean test regret / probability of finding the best-val architecture, over 50
seeds:

| strategy | 5 evals | 10 evals | 20 evals | 40 evals |
|---|---|---|---|---|
| random | 0.0785 / 0.04 | 0.0262 / 0.08 | 0.0149 / 0.14 | 0.0081 / 0.26 |
| regularised evolution | 0.0785 / 0.04 | 0.0262 / 0.08 | 0.0106 / 0.14 | 0.0085 / 0.52 |
| local search | 0.0966 / 0.04 | 0.0520 / 0.18 | 0.0230 / 0.48 | 0.0079 / 0.68 |

With 40 evaluations (32% of the space), local search found the best-val architecture 68% of the time, against 26% for
random search. Evolution matches random search until its population (10) fills.

**DARTS, first-order: a negative result, reported as measured.**

| seed | chosen cell | val acc | test regret | val percentile |
|---|---|---|---|---|
| 0 | zero, zero, zero | 0.340 | 0.657 | 0.00 |
| 1 | linear, zero, skip | 0.597 | 0.323 | 0.32 |
| 2 | skip, zero, relu_linear | 0.720 | 0.210 | 0.58 |
| 3 | skip, skip, skip | 0.477 | 0.457 | 0.20 |
| 4 | linear, skip, zero | 0.433 | 0.557 | 0.10 |

Each supernet run takes about 0.6 s. Its argmax cells are dominated by parameter-free ops, the known DARTS collapse,
and are worse than 5 random evaluations. See [LIMITATIONS](docs/LIMITATIONS.md).

**Surrogate.** An MLP on one-hot genotypes, scored by Spearman correlation on held-out architectures and averaged over
3 splits:

| training archs | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| Spearman ρ | 0.358 | 0.669 | 0.759 | 0.894 |

## Bugs fixed from v1

1. `benchmark.py` called `NASGraph.add_node` and `NASGraph.add_edge`, which did not exist. Both are now implemented,
   and topological sort raises `CycleError` instead of returning an order for a cyclic graph.
2. **`BENCHMARK_RESULTS.md` was not measured.** It reported, for example, R² 0.96 for a predictor fit to *random
   targets*, which is impossible. It has been removed, and every number in this README comes from `autonas bench`.
3. The parser keyed files by basename, so `a/utils.py` and `b/utils.py` overwrote each other. It also ignored
   `async def` and method calls, silently swallowed syntax errors, and scanned `.git/` and `__pycache__/`.
4. The legacy `MixedOp` accepted `in_channels != out_channels` and then crashed in `forward`, because MaxPool and
   Identity cannot change channels. It now raises a `ValueError` at construction.
5. The README called the rule-based `ArchitectureSearchEngine` a "search" that "solves a bilevel optimisation". It is
   a fixed template, and it is now documented as one. The real searches live in `autonas.strategies` and
   `autonas.darts`.

All 11 original tests pass unchanged. 23 new tests were added; one is parametrised over the three strategies, so the
total is 34.

## License

MIT
