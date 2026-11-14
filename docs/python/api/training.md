# Trainer

For end-to-end training of supported models, start with the
[Python `train(model, config)` runner](../../../packages/workloads/src/shadowspill_workloads/recipes/README.md).
It handles preparation, search and execution. This reference covers the lower
level used to run an already selected Plan or write a custom training loop.

`Trainer` runs a saved training `Plan`, manages update counts, and provides an
optional loop, logging and checkpoints. Build, profile and plan first; Trainer
never repeats those phases. Start with the [complete example](../../examples/generic-training.md)
or [phase guide](../phases.md).

## Start or resume

Construct Trainer in a fresh execution process, before another device allocator
initializes. The Plan must come from `build_step`.

```python
from shadowspill import Plan, Trainer

with Trainer(Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)
    trainer.fit(training_batches(), steps=100, run_dir="run")
    trainer.save("run/checkpoint.ssp")
```

`fill_tensor(chunk, destination)` fills bounded CPU tensor views. `chunk` gives
the tensor name, dtype, owned geometry and flat element offset. The
[example initializer](../../examples/generic-training.md#execute-in-a-fresh-worker)
fills ordinary model tensors; Trainer initializes captured optimizer defaults.
`initialize(..., chunk_bytes=64 << 20)` sets the maximum chunk size and returns
Trainer. Use it before the first update. For eager model initialization outside
saved execution, `shadowspill.training.reset_parameters(model)` calls each
submodule's available `reset_parameters` method.

To resume, replace `initialize` with `trainer.load("run/checkpoint.ssp")`.
`fit(..., steps=100)` means **100 total completed updates**, including restored
updates. A checkpoint at update 60 therefore runs 40 more. Loading a checkpoint
and loading a Plan are separate: the former restores values and progress; the
latter describes computation and memory placement.

## Constructor options

<!-- api-options: packages/engine/src/shadowspill/training/trainer.py:Trainer.__init__ -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `plan` | Required | Executable training Plan from `build_step`. |
| `evaluation_plan` | `None` | Separately prepared forward Plan sharing current model state. |
| `schedules` | `None` | Captured optimizer control name → `callable(completed_steps)`. |
| `metric_reducer` | `None` | Convert completed CPU observations into scalar metrics or `MetricSummary`. |
| `frozen_state_id` | `None` | Stable base-checkpoint identity; permits omitting unchanged frozen weights from checkpoints. |
| `parameter_norms_every` | `0` | Collect weight/gradient norms every N updates; zero disables collection. |
| `device` | `0` | Process-local device ordinal; must match the compiled device binding. |
| `rank_map` | `None` | Worker → logical-rank mapping for distributed execution. |
| `spill` | `None` | Default pinned-host pool, or an explicit spill backend. |
| `external_headroom_bytes` | `None` | Inherit the Plan's external-memory reservation; an override cannot increase it. |
| `reject_overbudget` | `False` | Fail on observed external/whole-process budget overruns instead of reporting them. Pool bounds remain enforced either way. |
| `executor` | `None` | Borrow an Executor for this exact Plan instead of creating one. |

The default pinned-host capacity is the worker's required spill peak plus 5%,
capped by the Plan's spill budget. The peak includes shared evaluation and
accumulation paths. `trainer.executor.spill_capacity_bytes` reports the selected
capacity. An explicit `spill` keeps its requested capacity and must fit both the
requirement and budget, with matching communication measurements. See
[pool sizing](pool-memory.md) and [external storage](ssd.md).

A borrowed Executor owns device, rank, pool and headroom configuration. Set
`spill` and `reject_overbudget` on that Executor; passing them alongside
`executor=` is rejected. It must already contain any requested evaluation Plan
and enable parameter norms if needed. `close()` or context-manager exit waits
for work and closes only an Executor Trainer created.

## Inputs and one update

One `step` completes **one optimizer update** over all microbatch calls captured
in the Plan. The build argument `num_examples` counts these complete calls,
not individual samples. A Plan built with `num_examples=4` and input batch size
16 processes 64 samples per rank per update.

```python
# Exactly four calls, matching the shapes captured at build time.
result = trainer.step(
    args=[(features0,), (features1,), (features2,), (features3,)],
    kwargs=[{"targets": y0}, {"targets": y1}, {"targets": y2}, {"targets": y3}],
)
print(result.step, result.loss, result.seconds)
```

<!-- api-options: packages/engine/src/shadowspill/training/trainer.py:Trainer.step -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `args` | `None` | Sequence of positional tuples, one per captured call. |
| `kwargs` | `None` | Sequence of keyword mappings, one per captured call. |
| `input_loader` | `None` | Alternative to `args`/`kwargs`: repeatable `index -> (args, kwargs)` reader. |
| `hyperparams` | `None` | Runtime optimizer controls or scalar fills of named persistent model buffers. |
| `runtime_trace` | `False` | Collect this real update's task/allocation/transfer diagnostics. |
| `profiler_annotations` | `False` | Emit task/transfer ranges for an external profiler. |

The objective owns loss normalization across microbatches and ranks. Trainer
neither divides by batch size nor performs an extra distributed loss reduction.
A mean over four equally sized microbatches, for example, needs each call's mean
loss divided by four. Input values may change; tensor signatures and captured
non-tensor constants must match the Plan. Inputs are validated and staged in the
spill pool before GPU execution.

`StepResult` contains CPU `losses`, `metrics`, `parameter_metrics`, `step`,
`seconds`, applied `hyperparams`, reduced `summary`, and optional `diagnostics`.
`loss` sums the per-call losses. `seconds` includes input staging, execution and
observation/trace collection; Python metric reducers, logging and checkpointing
run outside that timer. Simulation describes the scheduled GPU work, so its
duration is not an end-to-end loop timer. Results survive later updates and
Trainer shutdown. Use [Executor](executor.md) for asynchronous calls.

## Schedules and the optional loop

`schedules={"optimizer_group.0.lr": schedule}` calls `schedule(step_count)` before
each update, starting at zero. Only optimizer controls captured during build are
valid schedule names. Explicit `step(hyperparams=...)` values override schedules;
omitted controls use captured defaults. Python constants baked into an optimizer
cannot become runtime controls later. `hyperparams` can also fill persistent
model buffers by qualified name; omitted buffers keep their current values.
Parameters and nonpersistent buffers cannot be filled this way.

`fit` consumes **one mapping of `step` arguments per update**. For example, for a
Plan with four microbatches:

```python
def training_batches():
    for microbatches in data_loader:  # Each item contains four (features, targets) pairs.
        yield {
            "args": [(x,) for x, y in microbatches],
            "kwargs": [{"targets": y} for x, y in microbatches],
        }

trainer.fit(training_batches(), steps=100, run_dir="run", log_every=10)
```

<!-- api-options: packages/engine/src/shadowspill/training/trainer.py:Trainer.fit -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | Required | Iterable of mappings passed to `step`. |
| `steps` | Required | Target total completed updates, not additional updates. |
| `run_dir` | `None` | Optional directory for JSONL metrics, tables and default checkpoint paths. |
| `log_every` | `1` | Scalar logging cadence; zero disables it. Console logging works without `run_dir`. |
| `tables_every` | `100` | Reduced-metric table cadence; zero disables it. |
| `logger` | `None` | Additional `logger(record)` destination; optional `.table(...)` receives tables. |
| `callbacks` | `()` | `(trainer, result)` callbacks after logging and before checkpointing. |
| `checkpoint_every` | `0` | Save cadence; zero disables automatic saves. |
| `checkpoint_dir` | `None` | Defaults to `run_dir/checkpoints` when checkpointing is enabled. |
| `keep_last` | `3` | Positive number of automatic checkpoints to retain. |
| `startup_diagnostics` | `False` | Diagnose the first input before training; requires `run_dir` and a supported optimizer. |

Enabled cadences also run on the final update. Automatic saves use the default
master-weight representation. `fit` returns the last `StepResult`, or `None` if
already at the requested count; an exhausted source before that count is an
error. A resumable source supplies `state_dict`/`load_state_dict`. If the loaded
checkpoint contains source progress, the next `fit` restores it before iteration
and requires such a source. Use a custom `step` loop for application-specific
validation or checkpoint policies.

## Evaluation

Build and plan `build_forward` separately, then supply it as `evaluation_plan`.
`evaluate(args=None, kwargs=None, input_loader=None)` uses that Plan's captured
call structure and returns ordinary outputs on CPU. It shares the current model
tensors, without an optimizer update, schedule advance or update-count change.
Use `forward_fn` during build to capture a loss or another evaluation entry point.

Training and evaluation are admitted together against one budget. Their tensor
names/layouts must agree; evaluation can require additional workspace even
though its weights are shared. The [Executor reference](executor.md) describes
shared state and named Plans.

## Checkpoints

| Method | Options | Result |
| --- | --- | --- |
| `save(path)` | `source=None`, `chunk_bytes=64 << 20`, `weights="master"` | Saved archive `Path`. `source` defaults to the most recent `fit` source. |
| `load(path)` | `chunk_bytes=64 << 20`, `fill_tensor=None` | Trainer, with restored tensors and progress. `fill_tensor` supplies an omitted frozen base. |

Checkpoints include persistent training state, Trainer count/elapsed time,
default RNG state, and source/schedule state when their callbacks expose it.
Saving waits for outstanding work and streams bounded CPU chunks; it does not
assemble a full-device snapshot.

- `weights="master"` stores ordinary master weights once and reconstructs their
  compute copies on load. This is the default.
- `weights="compute"` stores compute weights and reconstructs masters from those
  rounded values, losing the master's additional precision.
- Parameters without masters and optimizer/buffer state are saved independently
  of that choice. Encoded weights or incompatible layouts retain required copies;
  the manifest explains exceptions.

A compatible Plan can use different kernels, microbatch geometry or budgets if
the checkpoint's tensor names, shapes and owned regions still match. Distributed
workers save one archive per logical rank. Sharded compute weights are derived
locally; replicated compute weights with sharded masters are reconstructed
collectively during load. All ranks participate. Restore does not reshard to a
different world size. See [distributed execution](distributed.md) and
[checkpoint reconstruction](../../architecture/state-import.md#export-and-checkpoints).

`frozen_state_id` enables checkpoints that omit unchanged frozen parameters.
Use a stable base revision/content identity that also fixes model configuration.
Restore requires that same identity and an explicit `fill_tensor` loader, even
in an already initialized worker:

```python
with Trainer(plan, frozen_state_id="base-checkpoint-revision") as trainer:
    trainer.load("adapter-update.ssp", fill_tensor=fill_base_tensor)
    trainer.fit(resumable_source, steps=100)
```

Trainable tensors, persistent buffers and mutated frozen parameters remain in the
archive; a missing retained tensor is an error.

## Metrics and logger helpers

`metric_reducer(observations)` returns a scalar mapping or
`shadowspill.training.observations.MetricSummary` with scalars and tables.
`parameter_norms_every=N` additionally exposes CPU `param_norm`/`grad_norm` scalars
under `result.parameter_metrics[name]` every Nth update, including after resume.
It measures compute weights and final gradients, counts unique sharded elements,
and needs small buffers within the existing device budget. Zero disables the
collector entirely. Norms and tracing add measured runtime cost that the saved
training simulation omits. See [norm observation details](../../architecture/optimizer.md#observing-parameters-and-gradients).

`shadowspill.training.logging` provides optional helpers; importing it does not
import W&B:

| Helper | Use |
| --- | --- |
| `Wandb(project=..., run_dir=..., **options)` | Context manager forwarding options to `wandb.init`; receives scalar records and tables. Writes `wandb_run.json`. |
| `DistributedLogger(control_group, run_dir=..., wandb=None, device=None, reduce=sum_rank_records, max_pending=256, timeout=120)` | Asynchronously aggregates CPU records over a Gloo control group. All ranks construct, log the same steps and close in matching order. |
| `sum_rank_records(records)` | Sums training loss/work units, takes maximum step/elapsed/eval times, and derives throughput. Other metrics, including evaluation loss, remain rank-local. |

The default distributed reducer assumes each rank's loss is already its share of
the global objective. Supply a different `reduce` function for other conventions.
Distributed output lives in `run_dir/aggregate/metrics.jsonl`; an optional `wandb`
mapping requires `project` and adds per-rank plus aggregate runs.

## Startup diagnostics

`diagnose(directory=..., warmup=1, args=None, kwargs=None, input_loader=None,
hyperparams=None)` writes measured/simulated timelines and returns their summary.
It runs at zero learning rate without advancing Trainer progress or schedules,
and restores affected model buffers, controls and default RNG. Its time is
excluded from `elapsed_seconds`; indexed inputs must be repeatable.

This requires the captured optimizer's explicit `zero_lr_preserves_state=True`
guarantee and a CPU tensor learning rate per group. Ordinary PyTorch AdamW does
not provide that guarantee. Unsupported Plans fail before a diagnostic update;
a GPU execution failure still makes the Executor unusable.

For a trace of an ordinary training update, use `step(runtime_trace=True)` and
read `result.diagnostics`. It changes weights/counts normally. External profiler
ranges use the independent `profiler_annotations=True` flag. The
[saved-Plan runner](../../../benchmarks/plan_runner/README.md#nsight-systems)
can collect bounded Nsight captures with per-rank output files.
