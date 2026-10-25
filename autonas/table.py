"""Exhaustive tabular benchmark: train every genotype once per seed.

Search strategies then *query* the table, so their cost is counted in
architecture evaluations and results are exactly reproducible
(the approach of NAS-Bench-101/201, at toy scale).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from autonas.data import Split
from autonas.space import Genotype, SearchSpace
from autonas.train import TrainConfig, train_genotype

log = logging.getLogger(__name__)


@dataclass
class TabularBenchmark:
    space: SearchSpace
    val: dict[Genotype, float] = field(default_factory=dict)
    test: dict[Genotype, float] = field(default_factory=dict)
    params: dict[Genotype, int] = field(default_factory=dict)
    build_seconds: float = 0.0

    @classmethod
    def build(
        cls, space: SearchSpace, data: Split, cfg: TrainConfig = TrainConfig(), seeds: tuple[int, ...] = (0,)
    ) -> TabularBenchmark:
        t0 = time.perf_counter()
        tb = cls(space)
        for k, g in enumerate(space.all_genotypes()):
            runs = [train_genotype(space, g, data, TrainConfig(cfg.width, cfg.steps, cfg.lr, s)) for s in seeds]
            tb.val[g] = sum(r["val_acc"] for r in runs) / len(runs)
            tb.test[g] = sum(r["test_acc"] for r in runs) / len(runs)
            tb.params[g] = int(runs[0]["params"])
            if k % 25 == 0:
                log.info("table %d/%d", k, space.size)
        tb.build_seconds = time.perf_counter() - t0
        return tb

    def query(self, g: Genotype) -> float:
        return self.val[g]

    @property
    def best_val(self) -> Genotype:
        return max(self.val, key=lambda g: (self.val[g], -self.params[g]))

    def test_regret(self, g: Genotype) -> float:
        return max(self.test.values()) - self.test[g]

    def val_percentile(self, g: Genotype) -> float:
        """Fraction of the space with validation accuracy strictly below g's."""
        v = self.val[g]
        return sum(1 for x in self.val.values() if x < v) / len(self.val)

    def save(self, path: str | Path) -> None:
        rows = [
            {"ops": list(g.ops), "val": self.val[g], "test": self.test[g], "params": self.params[g]} for g in self.val
        ]
        Path(path).write_text(json.dumps({"n_nodes": self.space.n_nodes, "ops": self.space.ops, "rows": rows}))

    @classmethod
    def load(cls, path: str | Path) -> TabularBenchmark:
        d = json.loads(Path(path).read_text())
        tb = cls(SearchSpace(d["n_nodes"], tuple(d["ops"])))
        for r in d["rows"]:
            g = Genotype(tuple(r["ops"]))
            tb.val[g], tb.test[g], tb.params[g] = r["val"], r["test"], r["params"]
        return tb
