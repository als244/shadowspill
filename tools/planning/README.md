# Planner experiments

These tools measure the planner across a reproducible corpus of workloads.

- [Corpus collection](corpus/README.md) builds and profiles programs on a GPU.
- [Evaluation](evaluation/README.md) reuses that corpus to compare planner settings,
  memory budgets and transfer assumptions on CPU.
- `cases/` contains the model specifications shared with performance qualification.

Each run records source identities, configuration and measurements. Existing
corpora and result archives remain unchanged when the tools move.

For a single training preparation, the smaller public
[offline sweep API](../../docs/python/advanced/offline-sweeps.md) exports measurements
and produces comparison tables and HTML reports on a laptop.
