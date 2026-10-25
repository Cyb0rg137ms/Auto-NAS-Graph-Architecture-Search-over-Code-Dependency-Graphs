import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from autonas.plots import plot_bench


def test_plot_from_committed_bench(tmp_path):
    r = json.loads((Path(__file__).parent.parent / "docs/assets/bench.json").read_text())
    out = plot_bench(r, tmp_path / "x.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_cli_plot_quick(tmp_path):
    from autonas.cli import main

    assert main(["bench", "--quick", "--plot", str(tmp_path / "q.png"), "--json", str(tmp_path / "q.json")]) == 0
    assert (tmp_path / "q.png").exists()
