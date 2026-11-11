# Saved-Plan benchmarks

Run a selected training Plan repeatedly without rebuilding, reprofiling or
replanning it. This harness uses `Trainer`, so the saved execution, chunking,
checkpoint-region, retention and parallelism choices remain the ones in the
Plan. Normal device compatibility, physical memory admission and spill sizing
checks still apply. A Plan does not become portable to an incompatible GPU
architecture by going through this runner.

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
launcher, e.g. `torchrun --standalone --nproc-per-node=8 -m benchmarking.rank_device script.py`.
For multiple nodes, use the equivalent torchrun rendezvous options on each
node. Each worker must expose its assigned GPU as `cuda:0`, run in a fresh
process and participate at the saved Plan's world size. `rank_map` supports a
process-to-logical-rank permutation. The output directory must be shared across
nodes. `run` returns the aggregate result on process rank zero and `None` on
other ranks. A failed worker must be terminated by the launcher, as with normal
distributed training.

## Timing and device observations

All updates execute normally. `warmup_steps` only excludes initial updates from
the reported steady median. For each update, take the maximum of all rank
`Trainer.step` times, then take the median of the measured updates. Throughput
uses **global** units divided by this median. Losses and extra scalars remain
per rank; the runner does not assume arbitrary workload losses can be summed.

`result.json` records the saved Program/schedule identity, prediction, budgets,
actual per-rank spill capacities, losses and all update times. Per-rank journals
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
| `lifecycle.py` | Shared host phase journal used by recipes and benchmarks |
| `monitor.py` | External GPU sampling with explicit process cleanup |
| `report.py` | Offline CPU phase/device analysis |
| `examples/saved_run.py` | Short Python entry point to edit for a campaign |
