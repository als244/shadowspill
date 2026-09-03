# Qualification

`qualification/` is ShadowSpill's thin release-acceptance surface. It owns the
protocol descriptions and four launchers, but no alternate implementation of
planning, execution, diagnostics, serialization, or model state.

```text
qualification/
├── numerical/
│   ├── README.md
│   ├── run.py       one reference/planned correctness cell
│   └── matrix.py    the five approximately-1B cells
└── performance/
    ├── README.md
    ├── run.py       one full-model throughput cell
    └── matrix.py    the retained full-model matrix
```

The launchers delegate to `src/tools/qualification/`, which in turn uses the
public `src/shadowspill/` APIs and workload definitions under `workloads/`.
Generated reference states, compact result summaries, and optional detailed
reports are written beneath `qualification/results/`, which is ignored by Git.
The numerical matrix reuses one identity-checked compiled reference under
`qualification/results/references/approximately_1b/<model>/<provider>/reference.pt`.
Its neighboring `inputs.pt` contains the exact input microbatches, while the
reference contains only the final model and optimizer state; repeated matrix
runs do not create duplicate checkpoints.

Run the numerical matrix:

```bash
python -m qualification.numerical.matrix \
  --keep-going
```

Compact correctness evidence is the default. A `--cold` run uses temporary
compiler and planning caches and removes them afterward. Use
`--detailed-artifacts` only for an investigation that needs full PlanReports,
PressureFit fixtures, and per-task runtime traces. Use
`--regenerate-reference` only when intentionally replacing the canonical
compiled references and input sidecars.

Run the full-model matrix:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model \
  --force-fresh \
  --keep-going
```

Both matrix launchers follow the planning-evaluation logging protocol: every
cell opens with a labeled START block (model, data geometry, budgets), the
cell subprocess streams live under a `[cell/N]` prefix, and every cell closes
with a PASS/FAIL block carrying per-gate status and UTC START, STOP, and
DURATION records. The console stream is duplicated with timestamps into
`matrix.log` beside `summary.json`, and each cell keeps one timestamped log.

The performance matrix judges throughput against floors measured on one
machine. Its `--measure-only` reports the measurement without those floors,
closing cells as MEASURED rather than PASS, which is the mode to run on a
machine the floors did not come from.

Framework-free PressureFit benchmarking belongs in
`benchmarking/planning_eval/fixture_benchmark.py`. Step inspection reads the
saved step diagnostics (`docs/python/step-diagnostics.md`), and the gap report
below summarizes them across a matrix.

`--profiler-annotations` on the performance launcher emits profiler ranges around
task boundaries and compiled calls, so an external profiler can attribute time
to the task that spent it. It is off by default because the ranges cost
something to emit and a gate run should not pay for them.

The real-versus-simulated gap report reads the traced warm steps a
performance matrix saved and prints, per model, where the step's time went
against the simulation: span, task-duration, and idle deltas; task-duration
error by phase; task start drift along the compute lane; each transfer lane's
assumed versus effective bandwidth; and every measured transfer's achieved
rate classified by its overlap with the opposite lane and bucketed by size.
It is the acceptance experiment for simulator changes:

```bash
python -m tools.qualification.gap_report qualification/results/full_model
```
