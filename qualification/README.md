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

Framework-free PressureFit benchmarking belongs in
`benchmarking/planning_eval/fixture_benchmark.py`. Step and NSYS inspection
belong in `src/tools/diagnostics/`.
