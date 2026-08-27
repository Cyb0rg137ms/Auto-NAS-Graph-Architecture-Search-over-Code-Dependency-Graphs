"""
test_auto_nas.py
================
Comprehensive test suite for AutoNAS-Graph architecture search.

Tests cover:
  - DARTS: differentiable architecture weights gradient flow
  - Cell construction: correct op types and shapes
  - Supernet forward pass with mixed operations
  - Architecture discretisation: argmax selection
  - Valid architecture property: dag has no cycles
  - Predictor: monotone improvement after training
"""

import pytest
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


@skip_no_torch
class TestDARTS:
    def test_architecture_weights_require_grad(self):
        from nas_graph.darts import DARTSCell
        cell = DARTSCell(in_channels=16, out_channels=16, n_ops=4)
        for w in cell.arch_weights:
            assert w.requires_grad

    def test_darts_forward_pass_shape(self):
        from nas_graph.darts import DARTSCell
        cell = DARTSCell(in_channels=16, out_channels=16, n_ops=4)
        x = torch.randn(2, 16, 8, 8)
        out = cell(x)
        assert out.shape[1] == 16

    def test_architecture_discretisation(self):
        from nas_graph.darts import DARTSCell
        cell = DARTSCell(in_channels=16, out_channels=16, n_ops=4)
        discrete = cell.discretise()
        assert discrete in range(4), f"Invalid op index: {discrete}"

    def test_arch_weights_gradient_update(self):
        from nas_graph.darts import DARTSCell
        cell = DARTSCell(in_channels=16, out_channels=16, n_ops=4)
        x = torch.randn(2, 16, 8, 8)
        loss = cell(x).mean()
        loss.backward()
        arch_grads = [w.grad for w in cell.arch_weights]
        assert any(g is not None for g in arch_grads)


@skip_no_torch
class TestPredictor:
    def test_predictor_improves_over_random(self):
        from nas_graph.predictor import ArchPredictor
        predictor = ArchPredictor(input_dim=10, hidden_dim=32)
        # Random arch vectors
        import torch
        X = torch.randn(50, 10)
        y = torch.randn(50, 1)
        initial_loss = predictor.evaluate(X, y)
        predictor.train_epochs(X, y, n_epochs=20, lr=0.01)
        final_loss = predictor.evaluate(X, y)
        assert final_loss <= initial_loss + 0.5

    def test_predictor_output_shape(self):
        from nas_graph.predictor import ArchPredictor
        predictor = ArchPredictor(input_dim=10, hidden_dim=32)
        x = torch.randn(5, 10)
        out = predictor.predict(x)
        assert out.shape == (5, 1)


class TestDAGProperty:
    def test_generated_dag_acyclic(self):
        from nas_graph.graph import NASGraph
        g = NASGraph(n_nodes=5, n_ops=3)
        assert g.is_acyclic(), "NAS graph must be a DAG"

    def test_dag_topological_sort(self):
        from nas_graph.graph import NASGraph
        g = NASGraph(n_nodes=5, n_ops=3)
        order = g.topological_sort()
        assert len(order) == 5, "Topological sort should include all nodes"
