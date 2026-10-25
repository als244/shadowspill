# Bandwidth frontier

Compare saved, profiled candidates under different host-transfer rates and
memory budgets. Planning and plotting are CPU-only: this tool never builds a
model, profiles a task, opens a pool, or installs an allocator.

Build/profile once through the public APIs or quickstart, then pass one or more
`ProfiledProgram` archives. Quickstart writes these under `profiled/`; its
`programs/index.json` associates their filenames with candidate names.

```bash
python -m benchmarking.bandwidth_frontier \
    --program batch1=/path/to/profiled/candidate-0000.ssp \
    --program batch2=/path/to/profiled/candidate-0001.ssp \
    --budget-gib 12 16 20 24 \
    --spill-gib 112 \
    --sim-transfer-bandwidths 2.75/2.75,25/25,100/100 \
    --units-per-step 65536 --unit-label tokens \
    --title 'Qwen3 30B-A3B, 65536 tokens per update' \
    --output-dir results/bandwidth
```

Each supplied candidate must represent the same amount of work per update.
`--units-per-step` defaults to one and `--unit-label` to `steps`. Rates use decimal
GB/s; memory uses GiB. `--latency-us FETCH EVICT` overrides per-transfer latency.
An optional `--communication-info` archive supplies topology, collective
measurements, and other communication assumptions. Only host rates/latencies
change across the sweep; concurrent fetch/evict rates equal their solo rates.

For training Programs, `--orderings factors` tries depth/breadth factor pairs;
`depth-first` restricts the traversal. Programs with ordering constraints retain
their legal default. `--resolution-options` and `--search-workers` configure the
existing planner. Each budget is planned independently; the old frontier's
cross-budget incumbent sharing is not provided by this phase runner.

`raw_data/` holds the request, one full `SweepResult` archive per calibration,
and JSON/CSV rows including failures and winners. `--resume` reuses completed
calibrations only when input hashes and planning settings match. Unexpected
implementation errors stop the run; infeasible budgets remain recorded cases.
No compiler caches are copied into these results.

Figures can be redrawn from numeric evidence alone:

```bash
python -m benchmarking.bandwidth_frontier \
    --replot results/bandwidth --output-dir results/redrawn
```

Optional `--measured observations.json` overlays supplied observations:

```json
{"measured": [{"label": "Measured H100", "spill_gib": 112,
               "units_per_second": {"16": 2500, "24": 3100}}]}
```

The former model/build-store/runtime options are removed. Choose the model and
input shapes during build, then reuse the resulting profiles here. Estimated
memory remains simulation-only; this runner does not make a Plan executable.
