# Full-model performance qualification

The runner builds, profiles, plans and executes in separate processes. The last
worker loads an executable Plan and exact initialization data; it never constructs
the model, compiles tasks, profiles them, or calibrates communication.

The retained geometries use sequence length 1,024 and 65,536 tokens per update:

| Family | Tokens per example | Examples per update |
|---|---:|---:|
| Llama 3 8B | 8,192 | 8 |
| Qwen 3.5 9B | 16,384 | 4 |
| OLMoE 7B | 32,768 | 2 |

```bash
python -m qualification.performance.run llama3 mlops \
  qualification/results/full_model/mlops_llama3.json

python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model --keep-going
```

The matrix defaults to the three MLOps cells with recorded throughput floors.
The two PyTorch cells are also available through `--cells`. Cases run serially so
GPU activity from another case cannot contaminate timings. Pinned-host and remote
spill use the same phase runner; remote configuration reaches calibration and
Executor. Full-size remote qualification is separate from the default gate.

## Runner and artifacts

`matrix/` separates argument/manifest selection, worker command construction,
case execution, failure evidence and console reporting. Remote qualification
calls the same `matrix.run.main` entry; there is no second matrix implementation.

`run.py` prepares the request and launches four short phases:

| Process | Input | Output |
|---|---|---|
| `prepare.py` | Manifest and seed | `program.ssp`, exact CPU `inputs.pt` |
| `qualification.pipeline.profile` | Program | `profiles.ssp`, `profiled.ssp`, `communication.ssp` |
| `qualification.pipeline.plan` | ProfiledProgram, communication information, budgets | `plan.ssp`, `summary.json` |
| `execute.py` | Plan and initialization values | Timing samples, runtime evidence, verdict |

`<result>.phases/` retains the archives, request, settings, and `phases.json`.
The shared `qualification/pipeline/` package owns subprocess launching, bounded
initialization/export and timing/trace observations. `phases.py` contains only
warmup/checkpoint and repeated measurement; `verdict.py` applies the gate rules.
Planning hides devices and asserts that PyTorch was not imported. Execution uses
a fresh interpreter so Executor can install its allocator before device use.

`--plan-only` stops after saving the executable Plan. Replanning requires only
editing the saved `settings.json` and running:

```bash
python -m qualification.pipeline.plan /path/to/case.phases
```

Compiler caches remain separate from these archives. The matrix's `--empty-caches`
creates temporary Inductor/Triton caches per cell, then removes those caches;
phase archives remain available for inspection and execution. ArtifactStore
mode and export-bypass flags are no longer part of these runners.

## Measurement protocol and gates

With checkpointing, the worker streams an initial checkpoint, traces one warm
update, restores, and measures three groups of four updates. `--skip-checkpoint`
keeps the warm update instead. The matrix defaults to skipping checkpoint I/O;
`--checkpoint` enables it. Checkpoint/replay correctness is also tested by the
numerical matrix. Neither mode allocates a full anonymous checkpoint copy.

Each device-clock cycle starts immediately before dispatch and ends where the
next cycle starts, after the previous invocation and its terminal transfers
finish. The final event closes the last cycle. The cycle includes opening fetches,
required writeback and dispatch gaps; it does not subtract those costs. Scalar
loss reads and group logging occur after timing. `host_group_seconds` independently
records wall time, and `dispatch_seconds` records the asynchronous call duration.

Group lines report mean seconds per update and group tokens divided by group
time. The headline `median_step_seconds` is the median of all twelve cycles;
throughput is tokens per update divided by that median. Raw `cycle_seconds` and
group timings are saved. There are no extra cooling pauses between groups.

The existing gates remain:

- Exactly three groups of four measured updates and the expected optimizer counters.
- Finite objectives and successful checkpoint restore when enabled.
- Admitted physical/pool limits; no steady-state device allocations, pinned
  registrations or event growth, and no outstanding runtime work/errors.
- Measured cycle duration within 10% of simulation.
- Throughput at least 95% of a recorded floor when its hardware/configuration match.

`--measure-only` records the same verdict fields without failing the command on a
diagnostic gate. It does not redefine a failed gate as a pass. New measurements
never become their own throughput baseline. `--profiler-annotations` enables
runtime task ranges for an external profiler.

## Precision and memory

On SM80+, defaults are BF16 weights, gradients and AdamW moments without masters.
Below SM80, weights/gradients are FP16 and moments are FP32. The optimizer is
`mlops.optim.AdamW`. Independent `--model-dtype`, `--master-dtype`, `--grad-dtype`
and `--opt-state-dtype` overrides are saved in the manifest.

The default total device allowance is 16 GiB on SM80+ and 10 GiB below it;
`--execution-budget-gib` overrides it. Spill defaults to 112 GiB;
`--spill-budget-gib` changes it. `--planning-spill-budget-gib` can impose a smaller
planning limit. Executor allocates the capacity saved in the resulting Plan.
`--external-headroom-mib` reserves 512 MiB by default inside the total allowance.
Physical budget enforcement is always enabled; the old enforcement switches
are removed.

Recorded floors belong to the RTX 5090, the original geometry, 16/112 GiB budgets
and BF16 settings. A mismatch marks the regression comparison not applicable;
runtime, memory and simulator checks still apply. The result explains which
fields differ. Building needs CPU memory for the model and saved initial values;
execution memory-maps those values and fills Executor-owned storage in bounded
chunks. The host must also accommodate the configured pinned spill pool.

For remote runs, the Plan spill budget is the minimum of the requested peer
capacity and the planning allowance. Calibration and execution receive that same
capacity. The saved settings record it explicitly.
