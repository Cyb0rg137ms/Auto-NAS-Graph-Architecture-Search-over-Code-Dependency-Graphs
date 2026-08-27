# Auto-NAS-Graph — Architecture & Technical Reference

> **Full Project Name:** Auto-NAS-Graph — AutoML Codebase Miner & Architecture Search
> **Category:** AutoML / Neural Architecture Search / Developer Tooling
> **Language:** Python 3.9+, Standard Library AST
> **Test Coverage:** 3/3 unit tests passing ✅

---

## 1. Architecture Overview

```
04_auto_nas_graph/
├── nas_graph/
│   ├── parser.py      # AST-based codebase crawler & complexity extractor
│   ├── search.py      # Constraint-guided neural architecture selector
│   ├── evaluator.py   # FLOPs/parameter profiler + PyTorch stub generator
│   └── __init__.py
├── tests/
│   └── test_nas.py
├── cli.py             # Command-line interface (scan → search → generate)
└── README.md
```

### Component Interaction

```
┌───────────────────────────────────────────────────────────────┐
│                  AUTO-NAS-GRAPH PIPELINE                     │
│                                                               │
│  Source Directory ──► CodebaseASTParser                      │
│                           │                                   │
│                   Walk *.py files                             │
│                   Extract classes, functions, calls           │
│                   Build dependency graph                      │
│                   Compute mean_complexity score               │
│                           │                                   │
│                           ▼                                   │
│                  ArchitectureSearchEngine                     │
│                   Layer budget loop                           │
│                   Constraint checks (FLOPs, params)          │
│                           │                                   │
│                           ▼                                   │
│                  ModelPerformanceEvaluator                    │
│                   FLOPs count                                 │
│                   Param count                                 │
│                   PyTorch stub code generation               │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Framework

### 2.1 Bi-Level Architecture Optimization

NAS is framed as a constrained two-level optimization:

```
Outer level:
  minimize  validation_loss( w*(α), α )
  over architecture parameters α

Inner level:
  w*(α) = argmin  training_loss( w, α )
  over model weights w

Subject to constraints:
  Latency(α)  ≤  L_max   (milliseconds)
  Params(α)   ≤  P_max   (millions of parameters)
```

`α` is the architecture vector (one entry per layer slot, encoding layer type selection).

### 2.2 AST Complexity Metric

For each Python file `f`, complexity is the total number of statements across all functions:

```
Complexity(f)  =  SUM over all functions in f:  len( function.body )
```

The codebase mean complexity drives the network depth heuristic:

```
depth  =  clip( mean_complexity / 2,  minimum=2,  maximum=24 )
```

### 2.3 Layer Selection Priority Rules

At each depth slot `i` (0-indexed), the layer type is chosen by this priority cascade:

```
i == 0  or  i == (depth - 1)  →  Linear      (input / output projection)
i mod 3 == 0                  →  Attention   (every 3rd layer gets attention)
i mod 2 == 0                  →  Residual    (every other gets residual skip)
otherwise                     →  Conv        (standard convolution)

Subject to: cumulative FLOPs ≤ L_max  AND  cumulative Params ≤ P_max
If adding the chosen layer would violate a budget → use LayerNorm instead (cheap)
```

### 2.4 FLOPs Estimation per Layer Type

Assuming input feature dimension `d`:

| Layer | FLOPs Estimate | Notes |
|-------|----------------|-------|
| Linear (d × d) | 2 × d² | Multiply-accumulate pairs |
| Conv 3×3 on 32×32 map | 9 × d² × 1024 | Spatial × kernel × channels |
| Attention (seq len = 64) | 3×d²×64 + 2×64²×d | QKV projection + softmax |
| Residual (2 × Linear) | 4 × d² | Two linear layers + skip |
| LayerNorm | 4 × d | Mean, variance, scale, shift |

---

## 3. Workflow

```
CLI: --scan_dir PATH --max_latency L --max_params P --output_code model.py
        │
        ▼
CodebaseASTParser.scan(path)
  Walk all *.py files in directory
  ast.parse() each file
  Count: classes, functions, function calls
  Compute: mean_complexity
        │
        ▼
ArchitectureSearchEngine.search(mean_complexity, L_max, P_max)
  depth = clip(mean_complexity / 2, 2, 24)
  For each depth slot i:
    Select layer type by priority rules
    Check FLOPs + Params budget
    Add to architecture stack (or fall back to LayerNorm)
        │
        ▼
ModelPerformanceEvaluator.profile(architecture)
  Count total parameters
  Estimate total FLOPs
  Generate PyTorch nn.Module code → output .py file
```

---

## 4. System Design

| Component | Module | Role |
|-----------|--------|------|
| **Crawler** | `parser.py` | File traversal, AST node walking, dependency graph |
| **Search** | `search.py` | Depth heuristic, layer selection, constraint verification |
| **Profiler** | `evaluator.py` | FLOPs/param counts, PyTorch `nn.Module` code generation |
| **CLI** | `cli.py` | End-to-end orchestration with `argparse` interface |
| **Tests** | `test_nas.py` | AST accuracy, constraint satisfaction, code generation |

---

## 5. Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Codebase-aligned models** | Network depth and type reflect actual code complexity |
| **Pre-deployment profiling** | Estimates FLOPs/params before any training begins |
| **Code generation** | Outputs runnable PyTorch `nn.Module` class automatically |
| **Zero training required** | Architecture search runs in milliseconds on CPU |
| **Extendable layer pool** | Add custom layer specs to `CANDIDATE_LAYERS` list |

---

## 6. Test Results

```
tests/test_nas.py::test_ast_parser_complexity           PASSED
tests/test_nas.py::test_architecture_search_constraints PASSED
tests/test_nas.py::test_performance_evaluator           PASSED
────────────────────────────────────────────────────────
3 passed in 0.04s
```

---

## 7. Quick Start

```bash
pip install -e .
pytest tests/
python cli.py --scan_dir . --max_latency 6.0 --max_params 3.5 --output_code model.py
```

<div align="center">
  <a href="https://q.com"><img src="../../assets/https_q_com.png" width="80" /></a>
</div>
