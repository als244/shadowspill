# Measure a saved Plan

`run` measures an existing `build_step` Plan through `Executor`. It does not
build, profile or search. For training applications, use the
[public client](../../docs/python/client.md).

Supply a bounded initializer and an input source from your workload:

```python
from contextlib import nullcontext
from tools.measure_plan import run


def initialize(context):
    # Open weights here, after the execution allocator is installed.
    return nullcontext(fill_tensor)


def inputs(context):
    return make_input_source(rank=context.rank, world_size=context.world_size)


run(
    "training.ssp",
    initializer=initialize,
    inputs=inputs,
    output_dir="runs/measurement",
    steps=7,
    warmup_steps=1,
    units_per_step=65536,
    unit_label="tokens",
)
```

Each input is a mapping of `Executor.run` arguments, such as `args`, `kwargs`
and `hyperparams`. `Context` contains the Plan, logical `rank`, process
`worker_rank` and `world_size`. `execution_options` accepts a mapping or a
function of that context. Optional `schedules`, `metric_reducer` and
`parameter_norms_every` control per-update observations; input hyperparameters
override scheduled values.

Run in a fresh process. For distributed Plans, start one worker per assigned GPU
using the normal torchrun/rendezvous options. Each worker exposes its device as
`cuda:0` and shares the output directory. `rank_map` maps physical workers to
logical ranks. Rank zero returns the aggregate result; other ranks return `None`.

## Timing and reports

The default final update records a runtime trace. The steady median excludes
warmup updates and the traced update, so at least one untraced measured update
must remain. Set `trace_final_step=False` for throughput-only measurement.

Per-update timing covers execution, completion, owned host-result copies and
synchronization. It excludes input production, schedules, metric reduction and
logging. The aggregate uses the slowest rank per update; throughput divides
**global** useful units by that time. Losses remain separate by rank.

`result.json` records Plan identity, prediction, budgets, times, losses and trace
paths. Per-rank phase journals cover initialization, execution and cleanup.
Runtime traces use the standard [step diagnostics](../../docs/python/advanced/step-diagnostics.md).
Overflow flags remain explicit. Trace files are written after runtime cleanup.

An optional `nvidia-smi` child samples device state; `monitor=False` disables it.
Missing observations are recorded in `monitor.json`. Analyze the journals after
execution:

```bash
python -m tools.measure_plan.report runs/measurement
```

This writes `phase-report.json`, including partial journals from interrupted runs.

## Nsight

Pass `NsysCapture(start_step=2, steps=2)` as `nsys` to mark a capture window and
exclude those updates from steady timing. The caller launches the collector using
[`NsysCapture.rank_command`](../../packages/engine/src/shadowspill/orchestration/capture/nsys.py)
and verifies native reports after workers exit. Runtime tracing and Nsight can
be enabled together.

The collector buffers CUDA activity with a 60,000 ms flush interval by default.
Pass `NsysCapture(start_step=2, steps=2, cuda_flush_interval_ms=10000)` to use
a 10-second interval for that run. On CUDA 11 and later a value of `0` saves
buffers when they fill. Longer intervals can use more profiler memory.
