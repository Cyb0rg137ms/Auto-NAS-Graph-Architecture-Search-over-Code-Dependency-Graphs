# Limitations

- **The scale is a toy.** The space has 125 MLP cells and the task is a 2-D synthetic spiral problem; there is no
  CIFAR or ImageNet. Each architecture's accuracy is the mean of 2 seeds at 200 full-batch Adam steps, so noise of a
  few points remains, and test regrets below about 0.01 are within seed noise.
- **DARTS collapses here.** In the 5 seeded runs, first-order DARTS chose cells at the 0th–58th validation percentile,
  and one run chose an all-`zero` cell. It was worse than 5 random evaluations. This matches documented DARTS failure
  modes on tiny spaces (skip/zero dominance and the first-order approximation). No regularisers are implemented: no
  DARTS-PT, no early stopping, no second-order term.
- **The surrogate is a plain MLP over one-hot edges.** It is not permutation-invariant and has no GNN. Its Spearman
  correlation is about 0.89 with 100 of the 125 archs as training data, which is most of the space.
- **There is no hardware-aware search.** Latency is not measured. The v1 `ArchitectureSearchEngine` latency/params
  are hand-set proxies, and its "codebase complexity → network depth" mapping is a heuristic with no empirical
  justification. It is kept only for API compatibility.
- **The code miner is name-based.** Calls and bases are resolved by bare name, not by import resolution, so
  same-named functions in different modules merge into one node.
