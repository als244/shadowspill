# Saved-Plan benchmarks

Run a selected training Plan repeatedly without rebuilding, reprofiling or
replanning it. This harness uses `Trainer`, so the saved execution, chunking,
checkpoint-region, retention and parallelism choices remain the ones in the
Plan. Normal device compatibility, physical memory admission and spill sizing
checks still apply. A Plan does not become portable to an incompatible GPU
architecture by going through this runner.

This package owns benchmark execution and reporting. Shared host journals and
worker coordination live in [`orchestration`](../../orchestration/README.md).
The generic `run` entry point takes workload hooks; only the separate `recipe`
bridge knows how to read an existing recipe run. Training recipes never import
this benchmark package.

## Existing public training run

Use a short script like [examples/saved_run.py](examples/saved_run.py):

```python
from benchmarking.plan_runner import benchmark

if __name__ == "__main__":
    result = benchmark(
        "runs/selected",  # config/model/selected.json + plans/
        output_dir="benchmarks/attempt-1",
        steps=7,
        warmup_steps=1,
        trace_final_step=True,  # Default: trace update 7; time updates 2–6.
        units_per_step=2_097_152,  # Optional GLOBAL units per update.
        unit_label="tokens",
    )
    print(result["median_seconds"], result["predicted_seconds"])
```

`benchmark` starts fresh workers on the current node, using the saved rank
count. It reuses the selected recipe's initializer, real input source, metric
reducer and Trainer settings. It does not import a particular architecture or
choose sequence lengths, batch geometry, precisions or budgets. Those belong
to the saved run. Use a separately selected run directory for each budget or
geometry. Source artifacts are read-only; each attempt needs a new output path.

Random host initialization versus checkpoint loading comes from the saved
configuration. Benchmark runs start at update zero and disable evaluation,
checkpoint writes, resume, startup diagnostics and metric tables. They retain
normal Trainer scalar logging and workload metric reducers. Use the public
training recipe for evaluation/checkpoint/resume qualification. Initialization
is still bounded by Trainer's host buffers; the harness never creates a second
full copy of the model.

The optional command is `python -m benchmarking.plan_runner RUN --output-dir OUT`.
The Python function has the same controls; a large command line is not needed.

## Nsight Systems

Nsight capture is **off by default**. Enable it for any saved workload:

```python
from benchmarking.plan_runner import NsysCapture, benchmark

if __name__ == "__main__":
    result = benchmark(
        "runs/selected",
        output_dir="benchmarks/capture",
        steps=7,
        nsys=NsysCapture(
            start_step=5, steps=2,
            # output_dir="/shared/captures/experiment",  # Optional override.
        ),
    )
    for capture in result["nsys"]["reports"]:
        print(capture["rank"], capture["report_path"])
```

This runs updates 1–7 normally: warmup 1, timing 2–4, Nsight 5–6, and the default
ShadowSpill runtime trace on 7. Captured updates are excluded from the steady
median. The capture interval is one-based, must fit within the run, and must
leave at least one measured update outside all tracing. A requested overlap
with the final runtime trace is allowed, but combines both kinds of overhead.
Even updates outside the capture window run under the attached collector;
compare a separate capture-disabled run for an uninstrumented baseline.

`NsysCapture` defaults to CUDA/NVTX/OS runtime collection, no CPU sampling,
and device metrics on CUDA-visible GPUs at 1,000 Hz. It accepts `gpu_metrics`,
`gpu_metrics_frequency` and `executable` overrides. GPU metrics require the
appropriate host permissions; this API does not elevate privileges. One Nsight
collector runs **inside the distributed launcher for each global rank**. Before
starting that collector, the wrapper restricts `CUDA_VISIBLE_DEVICES` to the
rank's assigned GPU. `--gpu-metrics-devices=cuda-visible` therefore samples only
that device. The shared Trainer adapter uses rank barriers and **each rank's**
cudaProfilerApi start/stop,
labels `train/step=N/rank=R`, and enables the existing
`profiler_annotations=True` task/transfer ranges only during captured updates.
Collection stops without terminating the remaining updates.

The launcher waits for every worker and collector to exit and requires a
nonempty report from **every rank**. By default the common capture directory is
`OUTPUT_DIR/profiling`, where `OUTPUT_DIR` is this benchmark run's directory.
Set `NsysCapture(output_dir=...)` to choose a different common directory (relative
paths resolve from the launcher's working directory). Each collector creates
`rank-NNNNN/trace.nsys-rep`, `capture.json` and `collector.log` there. Global rank
names avoid collisions between nodes, and existing rank folders are never
overwritten. The manifest records hostname, global/local rank, device mask,
collector command, completion status and report size.

`command.json` records the launch, and `result.json` includes the per-rank
`nsys.reports` list. Report paths are relative to the benchmark run when inside
it, absolute otherwise. `content_verified=False`
means the report's ranges/kernels/metric samples still need offline inspection;
a requested metric flag alone is not evidence of successful collection. Export
or inspect the report after releasing the GPU allocation. The default runner
does not add SQLite export or report processing to execution.

Custom `run(...)` users pass the same `nsys=NsysCapture(...)` in each worker and
launch their script with `capture.rank_command(worker_command, output_dir=...)`
**inside** torchrun, using its `--no-python` option. For example, the node launcher
can construct this command in Python:

```python
import subprocess
import sys
from orchestration.nsys import NsysCapture

capture = NsysCapture(start_step=5, steps=2, output_dir="/shared/captures/experiment")
worker = capture.rank_command([sys.executable, "worker.py"])
# torchrun_options contains this job's nnodes, node-rank, rendezvous and nproc settings.
subprocess.run(["torchrun", *torchrun_options, "--no-python", *worker], check=True)
```

This wrapper preserves global/local rank identities and needs no cross-node
collector. Use a shared capture directory, or the same node-local path and gather
the globally named rank folders after the job. The capture interval must agree
with the worker's `run(..., nsys=capture)` settings. Barriers align the selected
updates; they do not synchronize the hosts' clocks for cross-node timeline
analysis. Actual multi-node capture still requires qualification on that setup.

The low-level worker API cannot relaunch arbitrary Python hooks; it controls
capture boundaries only. It does not claim to have saved an Nsight report.
The saved-run `benchmark(...)` handles that launch and report check automatically.
It auto-launches one node; use the worker API under your external launcher for
multiple nodes. The optional CLI equivalents are `--nsys-steps 2
--nsys-start-step 5` and `--nsys-output-dir /shared/captures/experiment`.

## Any saved training Plan

`run` is the lower-level worker API. Supply Python hooks instead of a saved
recipe directory. This works with other model families and custom workloads:

```python
from contextlib import nullcontext
from benchmarking.plan_runner import run


def initializer(context):
    # Open your tensor source here, after the execution allocator is installed.
    # fill_tensor(request, destination) fills the requested bounded CPU slice.
    return nullcontext(fill_tensor)


def inputs(context):
    # Each item contains kwargs for Trainer.step: args/kwargs or input_loader,
    # and optionally hyperparams. Use context.rank for logical data sharding.
    return make_input_source(rank=context.rank, world_size=context.world_size)


if __name__ == "__main__":
    run(
        "training.ssp",
        initializer=initializer,
        inputs=inputs,
        output_dir="benchmark",
        steps=7,
        warmup_steps=1,
    )
```

`Context` holds the loaded Plan, logical `rank`, process `worker_rank` and
`world_size`. Hooks are invoked after runtime creation. `trainer_options` can
be a mapping or a `context -> mapping` callable for schedules, metric reducers,
spill configuration or other normal Trainer options. The runner owns the
Executor, rank mapping and device; an optional `trainer_type` supports existing
Trainer subclasses. This API accepts **training Plans from `build_step`**.

Run the custom script directly on one GPU. For multiple GPUs use the normal
launcher, e.g. `torchrun --standalone --nproc-per-node=8 -m orchestration.rank_device script.py`.
For multiple nodes, use the equivalent torchrun rendezvous options on each
node. Each worker must expose its assigned GPU as `cuda:0`, run in a fresh
process and participate at the saved Plan's world size. `rank_map` supports a
process-to-logical-rank permutation. The output directory must be shared across
nodes. `run` returns the aggregate result on process rank zero and `None` on
other ranks. A failed worker must be terminated by the launcher, as with normal
distributed training.

## Timing and device observations

All updates execute normally. By default, `trace_final_step=True` collects a
runtime trace of the final training update on every rank. It does not add a
diagnostic update or change the learning rate. The steady median excludes both
the initial `warmup_steps` and that final traced update. At least one untraced
measured update must remain. Set `trace_final_step=False` (CLI `--no-final-trace`)
for a throughput-only run. The runner owns the per-input `runtime_trace` setting.

For each update, take the maximum of all rank `Trainer.step` times, then take the
median of the untraced measured updates. Throughput
uses **global** units divided by this median. Losses and extra scalars remain
per rank; the runner does not assume arbitrary workload losses can be summed.

`result.json` records the saved Program/schedule identity, prediction, budgets,
actual per-rank spill capacities, losses, all update times, the measured update
indices and each rank's trace paths. Each rank writes `trace/step.json` and
`trace/step.events.npz` after closing its runtime. These are the standard
[step diagnostics](../../docs/python/step-diagnostics.md): measured versus
simulated task times, input/reuse waits, transfers and collective records.
Overflow flags remain explicit in the result; an incomplete trace must not be
used for complete attribution. Event packing and file writes happen outside
the training loop; trace collection itself still adds overhead to the final
update. Detailed trace analysis belongs in CPU work after releasing the GPUs.

Per-rank journals
separately time imports, Plan loading, communicator opening, runtime creation,
initialization, input creation, fit, runtime cleanup and result gathering. The
parent launcher also times fresh-process startup and final process exit.
`Trainer.step` excludes input production, metric reducers and logging; the
`training.fit` span includes them. Phase journals do not synchronize CUDA.

An external `nvidia-smi` process samples utilization, memory, power, clocks and
temperature every second through startup, training and cleanup. The saved-run
launcher uses one monitor for its selected devices; custom distributed `run`
calls monitor their individual GPUs. `monitor=False` disables this observation.
Missing/unavailable metrics are recorded in `monitor.json`, not reported as zero.
Sampled utilization is not achieved FLOPs, and sampling can miss memory peaks.

After releasing the GPUs, run:

```bash
python -m benchmarking.plan_runner.report benchmarks/attempt-1
```

This writes `phase-report.json` by joining host timestamps and device samples
on CPU. No CSV decoding or report generation is added to the training critical
path. Interrupted runs keep their partial journals for this report too.

## Layout

| File | Responsibility |
| --- | --- |
| `api.py` | Model-independent, rank-local saved-Plan execution and aggregation |
| `recipe.py` | Fresh-worker launcher and bridge to existing saved recipe hooks |
| [`orchestration/nsys.py`](../../orchestration/nsys.py) | Shared external-profiler launch settings and capture adapter |
| [`orchestration/nsys_worker.py`](../../orchestration/nsys_worker.py) | One isolated collector and report per global rank |
| [`orchestration/lifecycle.py`](../../orchestration/lifecycle.py) | Shared host phase journal used by recipes and benchmarks |
| `monitor.py` | External GPU sampling with explicit process cleanup |
| `report.py` | Offline CPU phase/device analysis |
| `examples/saved_run.py` | Short Python entry point to edit for a campaign |
