"""
benchmark.py
============
Performance benchmark script for Auto-NAS-Graph optimizer.
Measures DAG cycle checking and topological sorting.
"""

import time
from nas_graph.graph import NASGraph

def benchmark_graph_validation():
    # Build a graph with 20 nodes
    graph = NASGraph()
    for i in range(20):
        graph.add_node(f"node_{i}", op="conv3x3")
        
    for i in range(19):
        graph.add_edge(f"node_{i}", f"node_{i+1}")
        
    t0 = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        valid = graph.is_acyclic()
        assert valid is True
        order = graph.topological_sort()
        assert len(order) == 20
    elapsed = (time.perf_counter() - t0) / iterations
    print(f"Graph Validation Time Ms: {elapsed * 1000.0:.4f}")

if __name__ == "__main__":
    print("Running Auto-NAS-Graph benchmarks...")
    benchmark_graph_validation()
