# PressureFit frontier collection

This package consumes the immutable `StepProgram` artifacts produced by
`corpus_collection/`. It does not rebuild models, capture graphs, compile, or
profile. One Program worker loads its Program once and evaluates every
configured budget/bandwidth point.

The versioned JSON config defines a Cartesian collection of execution budgets,
spill budgets, and exact rational transfer-bandwidth scales. The full v1 grid
contains 15 points per Program and 2,520 points over the 168-Program corpus.
One globally frozen bidirectional-concurrent fetch/evict pair is used across all
Programs, yielding exactly three bandwidth combinations at 1/2x, 1x, and 2x.

Each point is journaled before PressureFit starts and atomically publishes one
of `succeeded`, `infeasible`, `search_exhausted`, or `error`. A native worker
exit or timeout is attributed to the active point; the controller restarts the
Program worker, which reloads the Program and skips all completed points.

Successful points retain:

- a compact `point.json` for comparisons;
- the complete canonical `AnnotatedProgramPlan`, including source Program,
  schedule, recomputation vector, simulator intervals/timeline, candidate
  diagnostics, and physical-admission certificate;
- aggregate `frontier.csv` and `frontier.jsonl` indexes.

The annotated plan records the complete `PressureFitDiagnostics`, every
physical-admission refinement, and simulator evidence. Its timing section
separately reports cumulative PressureFit/cache-resolution time, physical
admission time, orchestration remainder, and total selection wall time, plus
the PressureFit/admission split for each refinement attempt. The compact
frontier indexes expose the aggregate timing fields and link to that complete
artifact.

`plan_digest` is a semantic identity: it deliberately excludes cache-hit and
wall-clock observations so identical planner decisions compare equal across
runs. `artifact_sha256` covers the complete serialized artifact, including all
timings, and uniquely identifies one measured run.

The baseline directory also freezes the config, corpus manifest, git revision,
tracked dirty patch, Python environment, logs, and all attempt journals.

Example:

```bash
python -m qualification.planner.collect_frontier \
  --config qualification/planner/configs/full_pressurefit_frontier_v1.json \
  --corpus-dir ~/.cache/shadowspill/corpora/full_model_program_corpus_v1 \
  --output-dir ~/.cache/shadowspill/corpora/full_model_program_corpus_v1/_frontiers \
  --planning-cache ~/.cache/shadowspill/planning/frontier_v1
```

Use `--resume` after interruption. `--case`, `--start-at`, and `--limit` select
subsets for smoke tests without changing point identity. A later planner
revision run with the same config creates a new implementation-derived baseline
ID and can be compared row-for-row through `frontier.csv`.
