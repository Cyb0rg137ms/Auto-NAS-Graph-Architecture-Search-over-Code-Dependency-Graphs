"""
LIMITATIONS.md — AutoNAS-Graph
==============================

# Known Limitations and Future Work

## Search Space constraints

### 1. Small Cell Search Space
Our current DARTS cell contains 4 nodes and 4 operations:
- A real search space (like NAS-Bench-201 or DARTS CNN space) typically has 6–8 nodes and 5–8 operations (including dilated convolutions, depthwise-separable convolutions, etc.).
- The small cell size restricts the complexity of discovered architectures.

### 2. High Memory Requirement of DARTS
DARTS updates weights $w$ and architecture parameters $\alpha$ simultaneously using bi-level optimization.
- The exact gradient of the architecture loss requires calculating $d^2w / d\alpha dw$, which involves Hessian-vector products.
- This requires storing intermediate gradients for all nodes, making DARTS extremely memory-rigid.
- Our implementation uses the first-order approximation (ignoring second-order derivatives) to fit within consumer GPUs.

---

## Performance Predictor

### 1. Simple MLP Surrogate
Our `ArchPredictor` uses a simple Multi-Layer Perceptron:
- It treats the graph topology as a flat vector of operation choices.
- It does not understand **isomorphic graphs** (e.g. if node 1 and node 2 are swapped but structurally identical, the MLP sees them as different vectors).
- To solve this, a production-grade surrogate model must use a **Graph Neural Network (GNN)** or Graph Convolutional Network (GCN) that is invariant to node permutations.

### 2. Lack of Real Hardware Measurements
Our latency proxy is a simple lookup table. Real GPU latency depends heavily on execution context, CUDA stream scheduling, and operator fusion.
"""
