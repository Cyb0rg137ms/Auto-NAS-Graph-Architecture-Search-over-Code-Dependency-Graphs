import tempfile
from pathlib import Path

import pytest
import torch

from autonas import (
    Genotype,
    SearchSpace,
    TabularBenchmark,
    TrainConfig,
    build_network,
    darts_search,
    DartsConfig,
    local_search,
    random_search,
    regularized_evolution,
    spearman,
    spirals,
    train_genotype,
)
from autonas.darts import Supernet
from autonas.surrogate import surrogate_rank_correlation
from nas_graph.evaluator import ModelPerformanceEvaluator
from nas_graph.graph import CycleError, NASGraph
from nas_graph.parser import CodebaseASTParser, cyclomatic_complexity
import ast

SPACE = SearchSpace(3)


@pytest.fixture(scope="module")
def data():
    return spirals(n_per_class=60, seed=0)


@pytest.fixture(scope="module")
def table(data):
    return TabularBenchmark.build(SPACE, data, TrainConfig(width=8, steps=15))


def test_space_size_and_enumeration():
    assert SPACE.edges == [(0, 1), (0, 2), (1, 2)]
    gs = list(SPACE.all_genotypes())
    assert len(gs) == SPACE.size == 125 and len(set(gs)) == 125


def test_neighbours_differ_in_one_edge():
    g = Genotype((0, 1, 2))
    nb = SPACE.neighbours(g)
    assert len(nb) == 3 * 4
    assert all(sum(a != b for a, b in zip(g.ops, n.ops)) == 1 for n in nb)


def test_zero_cell_output_is_constant():
    net = build_network(SPACE, Genotype((0, 0, 0)), 2, 3, 8)
    out = net(torch.randn(5, 2))
    assert torch.allclose(out, out[0].expand_as(out))  # zero cell -> input ignored


def test_skip_cell_math():
    from autonas.space import Cell

    c = Cell(SPACE, Genotype((1, 1, 1)), 4)
    x = torch.randn(3, 4)
    assert torch.allclose(c(x), 2 * x)  # n2 = x + n1 = x + x


def test_bad_genotype_length():
    with pytest.raises(ValueError):
        build_network(SPACE, Genotype((1, 1)), 2, 3)


def test_spirals_split_deterministic():
    a, b = spirals(seed=3), spirals(seed=3)
    assert torch.equal(a.x_train, b.x_train)
    assert len(a.y_train) + len(a.y_val) + len(a.y_test) == 900 and a.n_classes == 3


def test_training_beats_chance_and_is_seeded(data):
    g = Genotype((3, 3, 3))
    r1 = train_genotype(SPACE, g, data, TrainConfig(steps=150))
    r2 = train_genotype(SPACE, g, data, TrainConfig(steps=150))
    assert r1 == r2
    assert r1["val_acc"] > 0.6


def test_table_roundtrip_and_metrics(table):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        table.save(p)
        t2 = TabularBenchmark.load(p)
    assert t2.val == table.val and t2.params == table.params
    assert table.test_regret(max(table.test, key=table.test.get)) == 0
    assert 0 <= table.val_percentile(table.best_val) <= 1


@pytest.mark.parametrize("fn", [random_search, regularized_evolution, local_search])
def test_strategies_monotone_and_budget(table, fn):
    tr = fn(SPACE, table.query, 30, seed=1)
    assert len(tr) == 30
    v = [table.val[g] for g in tr]
    assert v == sorted(v)


def test_local_search_finds_optimum_with_full_budget(table):
    tr = local_search(SPACE, table.query, 400, seed=0)
    assert table.val[tr[-1]] == max(table.val.values())


def test_evolution_rejects_bad_sample(table):
    with pytest.raises(ValueError):
        regularized_evolution(SPACE, table.query, 5, population=2, sample=3)


def test_supernet_alpha_gets_gradient(data):
    net = Supernet(SPACE, 2, 3, 8)
    torch.nn.functional.cross_entropy(net(data.x_val), data.y_val).backward()
    assert net.alpha.grad is not None and net.alpha.grad.abs().sum() > 0
    assert all(p is not net.alpha for p in net.weights())


def test_darts_returns_valid_genotype(data):
    out = darts_search(SPACE, data, DartsConfig(width=8, steps=20))
    assert len(out["genotype"].ops) == 3
    assert torch.allclose(out["alpha_softmax"].sum(-1), torch.ones(3))


def test_spearman_properties():
    a = torch.tensor([1.0, 2, 3, 4])
    assert spearman(a, a * 10) == pytest.approx(1)
    assert spearman(a, -a) == pytest.approx(-1)
    assert spearman(torch.tensor([1.0, 1, 2]), torch.tensor([1.0, 1, 2])) == pytest.approx(1)


def test_surrogate_runs(table):
    r = surrogate_rank_correlation(table, 50, epochs=20)
    assert -1 <= r <= 1


def test_cyclomatic():
    src = "def f(x):\n    if x and y or z:\n        for i in x:\n            pass\n    return [a for a in x if a]\n"
    fn = ast.parse(src).body[0]
    assert cyclomatic_complexity(fn) == 1 + 1 + 2 + 1 + 1  # if, boolop(3 vals), for, comprehension


def test_parser_same_basename_no_collision_and_errors(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "a" / "utils.py").write_text("def f():\n    pass\n")
    (tmp_path / "b" / "utils.py").write_text("async def g():\n    self.h()\n")
    (tmp_path / "bad.py").write_text("def (:\n")
    (tmp_path / ".git" / "x.py").write_text("def z(): pass\n")
    p = CodebaseASTParser(str(tmp_path))
    g = p.scan()
    s = p.get_summary_metrics()
    assert s["total_files"] == 2 and s["total_functions"] == 2 and s["parse_errors"] == 1
    assert "h" in g["g"]


def test_graph_add_node_edge_and_cycle():
    g = NASGraph(n_nodes=0)
    for i in range(4):
        g.add_node(f"n{i}", op="conv")
    for i in range(3):
        g.add_edge(f"n{i}", f"n{i + 1}")
    assert g.topological_sort() == ["n0", "n1", "n2", "n3"] and g.is_acyclic()
    g.add_edge("n3", "n0")
    assert not g.is_acyclic()
    with pytest.raises(CycleError):
        g.topological_sort()
    with pytest.raises(KeyError):
        g.add_edge("n0", "nope")


def test_legacy_mixedop_channel_mismatch():
    from nas_graph.darts import MixedOp

    with pytest.raises(ValueError):
        MixedOp(8, 16, 4)


def test_generated_stub_runs_and_param_count_matches():
    layers = ["LinearProjection", "ConvolutionalBlock", "AttentionBlock", "ResidualBlock", "LayerNorm"]
    ev = ModelPerformanceEvaluator(layers, input_dim=32)
    ns: dict = {}
    exec(ev.generate_pytorch_stub(), ns)
    m = ns["AutoGeneratedModel"]()
    assert m(torch.randn(4, 32)).shape == (4, 32)
    assert sum(p.numel() for p in m.parameters()) == ev.profile()["parameter_count"]


def test_cli_mine(tmp_path, capsys):
    from autonas.cli import main

    (tmp_path / "m.py").write_text("def f(x):\n    if x:\n        return 1\n    return 2\n")
    out = tmp_path / "model.py"
    assert main(["mine", str(tmp_path), "--output_code", str(out)]) == 0
    assert out.exists() and '"total_files": 1' in capsys.readouterr().out
