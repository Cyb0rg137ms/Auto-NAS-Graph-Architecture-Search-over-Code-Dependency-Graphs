"""autonas: a small, fully measured neural-architecture-search toolkit.

* :mod:`autonas.space`     - cell search space, genotypes, network construction
* :mod:`autonas.data`      - deterministic synthetic tasks (spirals)
* :mod:`autonas.train`     - seeded training / evaluation of one genotype
* :mod:`autonas.table`     - exhaustive "tabular benchmark" of the whole space
* :mod:`autonas.strategies` - random search, regularised evolution, local search
* :mod:`autonas.darts`     - DARTS supernet with first-order bilevel optimisation
* :mod:`autonas.surrogate` - MLP performance predictor + rank correlation
* :mod:`autonas.mining`    - AST code-base miner (cyclomatic complexity, call graph)
"""

from autonas.darts import DartsConfig, darts_search
from autonas.data import spirals
from autonas.space import OPS, Genotype, SearchSpace, build_network
from autonas.strategies import local_search, random_search, regularized_evolution
from autonas.surrogate import spearman, surrogate_rank_correlation
from autonas.table import TabularBenchmark
from autonas.train import TrainConfig, train_genotype

__all__ = [
    "OPS",
    "DartsConfig",
    "Genotype",
    "SearchSpace",
    "TabularBenchmark",
    "TrainConfig",
    "build_network",
    "darts_search",
    "local_search",
    "random_search",
    "regularized_evolution",
    "spearman",
    "spirals",
    "surrogate_rank_correlation",
    "train_genotype",
]
__version__ = "2.0.0"
