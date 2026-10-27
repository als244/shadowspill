# Training and forward execution over saved plans

`Trainer` manages training through an Executor. Preparation is separate:
`build_step` → `measure_program` → `apply_profiles` → `plan_program`. Construct
Trainer in a fresh worker with the resulting Plan; it never builds, profiles,
or searches. See the [complete example](../../examples/generic-training.md) and
[phase API guide](../../architecture/offline-pipeline.md).

## Trainer

```python
from shadowspill import Plan, Trainer

with Trainer(Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)
    result = trainer.step(args=[(features,)], kwargs=[{"targets": targets}])
    print(result.step, result.loss)
```

The Plan must come from `build_step`. Its example count and input signatures fix
the per-call inputs. `fill_tensor(chunk, destination)` fills Executor-provided CPU
buffers; `chunk` identifies a tensor name, dtype, local storage geometry and flat
chunk offset. Initialization streams bounded chunks and creates optimizer defaults
without assembling a full model on the device.

For ordinary eager models, `shadowspill.training.reset_parameters(model)` calls
each submodule's `reset_parameters` method when present. Saved execution uses
the bounded `fill_tensor` interface above.

An optional existing `executor=` must belong to the same Plan. Trainer closes
only an Executor it created. `close()` and context-manager exit wait for work and
release owned resources. By default, Executor inherits the external-memory
headroom recorded during profiling and included in the Plan's total GPU budget.
An `external_headroom_bytes` override cannot exceed that planned reservation.

`spill=SSDPool(...)` or `spill=RemotePool(...)` selects storage when Trainer creates
its Executor; pass it directly to an existing Executor instead. Pinned-host storage
is the default. Capacity must equal the Plan's spill budget, and its communication
costs must describe the selected backend. Build/profile/planning open no spill pool.
See [SSD storage](ssd.md) for setup, bounded I/O and current sharded-transfer limits.

## Inputs, objectives and results

`step(args=..., kwargs=...)` takes sequences of positional tuples and keyword
mappings, one entry per example. Alternatively,
`input_loader(index) -> (args, kwargs)` supplies inputs in the plan's chosen order.
The objective controls normalization across examples; Trainer does not infer
weights from batch size or sequence length. Input trees support arbitrary model
arguments, not only text tokens.

`StepResult` holds CPU observations: per-example losses/metrics, update number,
elapsed seconds, applied hyperparameters and optional reduced metrics. Its `loss`
property sums the example losses. `Trainer.step` waits for completion. For direct
asynchronous control use `Executor.run`, whose result handle waits only when
`.result()` is called; the next run automatically waits for its predecessor.

For an external profiler, `Executor.run(..., profiler_annotations=True)` adds
semantic task and runtime-transfer ranges to that invocation. The default is off.
Annotations remain enabled until the result handle waits for task and transfer
completion, then disable before the next invocation. They do not change the saved
Plan or its admission. The [quickstart capture tool](../../../benchmarking/quickstart.md#nsight-captures-of-saved-winners)
uses this option in fresh workers.

## Evaluation

Build and plan a forward computation separately, with the same parameter names
and layouts, then pass `evaluation_plan=validation_plan`. `evaluate` accepts the
same input-call syntax and returns the model's ordinary output on CPU. Both
plans share current model tensors and are admitted together against one budget.
Evaluation does not update optimizer state or advance schedules.

`build_forward(..., forward_fn=...)` can capture a loss method or other evaluation
entry point. The default calls `model.forward`. The callback is needed only at
build time. Direct Executor users can admit named plans with
`Executor(training_plan, plans={"evaluation": validation_plan})` and select one
with `run(plan="evaluation", ...)`.

## Updates, schedules and loop policy

`hyperparams` supplies per-call optimizer controls declared during build and scalar
fills of persistent model buffers by their qualified names. For example,
`hyperparams={"lr": 0.001, "loss_scale": 2.0}` changes the learning rate and fills
the model's `loss_scale` buffer before the call. Omitted optimizer controls use
their captured defaults; omitted model buffers keep their current values. Model
parameters, nonpersistent buffers and tensor-wrapper components cannot be set
this way. Sharded buffer fills affect only owned elements and require a dense
full-storage view.

A `schedules` mapping passed to Trainer maps control names to
`callable(completed_steps)`. Explicit per-call values override schedule outputs.
Only captured runtime controls can change; a Python constant baked into an
optimizer graph cannot become a runtime control later.

`fit(source, steps=...)` consumes mappings of `step` keyword arguments. Optional
arguments configure logging, callbacks and checkpoint cadence. An input source
can implement `state_dict`/`load_state_dict` for checkpointed progress. Custom
loops can call `step`, `evaluate`, `save` and `load` directly. A `metric_reducer`
can aggregate completed `StepObservations` for application-specific reporting.

## Parameter norms and custom logging

`Trainer(plan, parameter_norms_every=N)` optionally collects compute-weight and
final-gradient L2 norms on every Nth completed update. Zero (the default) creates
no collector, norm buffers, diagnostic kernels or norm-summary transfers.
The result is `result.parameter_metrics[name]`, containing CPU scalar
tensors named `param_norm` and `grad_norm`; unobserved updates return an empty map.
The frequency uses the restored step count after resume.

```python
def summarize(observations):
    return {
        f"gradient_norm/{name}": values["grad_norm"].item()
        for name, values in observations.parameter_metrics.items()
    }

trainer = Trainer(plan, parameter_norms_every=100, metric_reducer=summarize)
trainer.fit(source, steps=1000, logger=write_record, callbacks=[after_step])
```

`metric_reducer` receives host observations; `logger(record)` receives log records;
each callback receives `(trainer, result)` after normal logging and before
checkpointing. These are ordinary Python functions and can customize aggregation,
destinations and loop behavior. They are not compiled tasks. Parameter norm
collection itself is a bounded built-in operation, not an arbitrary tensor callback.

Executor observes each compute weight at its first task read and the final
gradient before its local optimizer update. It does not measure the FP32 master
in place of the compute weight. Sharded norms sum squared contributions over
unique owned elements, excluding padding and duplicate replicas. The collector
uses one ordered task stream and retains no parameter or gradient references.

Reduction buffers occupy 1 KiB plus eight bytes per observed parameter, subject
to allocator alignment, within the existing execution pool and total GPU budget.
If that spare space is unavailable, worker setup fails rather than exceeding the
budget. There are no additional Program tasks or per-task allocations. Diagnostic
kernels are prepared during worker setup; scalar summaries are copied at result
completion. `StepResult.seconds` includes norm collection only on updates where
it runs. Skipped updates do no norm computation or summary transfer; an enabled
collector keeps its small buffers reserved between updates. Simulated timelines
describe the saved training tasks and omit diagnostics. Python reducers and loggers
run outside the step timer; total loop elapsed time includes them when enabled.
Caller-owned CPU summaries survive later calls and executor shutdown.

Direct users enable `Executor(plan, parameter_norms=True)` and can skip collection
per call with `run(collect_parameter_norms=False)`. Collection defaults to enabled
for its training Plan and disabled for named evaluation Plans. A Trainer supplied
with an existing Executor requires that Executor to enable norms when N is nonzero.

## Checkpoints

`save(path, source=None, chunk_bytes=...)` streams model tensors, masters,
optimizer state and stored gradients through the Executor export interface.
It also saves Trainer progress and optional input-source state. `load` restores
those tensors and counters into an already admitted compatible worker. File
formatting lives in checkpoint helpers rather than the Executor itself.

Saving waits for the pending invocation. Export is chunked and does not create a
full-device snapshot. A checkpoint is separate from a Plan: the Plan describes
computation/layout; the checkpoint supplies mutable values. Numerical qualification
checks bitwise replay after restoring a checkpoint.

For an immutable frozen base, construct `Trainer(plan, frozen_state_id="...")`.
Checkpoints then omit unchanged frozen parameters. Trainable tensors, their
compute weights and masters, optimizer state, persistent buffers and any mutated
frozen parameters remain in the archive. The identity is supplied by the caller;
use a checkpoint revision or content identity that also fixes model configuration.

```python
trainer = Trainer(plan, frozen_state_id="my-base-checkpoint-revision")
trainer.initialize(fill_tensor)
# ... train and save ...
trainer.load("update.ssp", fill_tensor=fill_tensor)
```

The loader fills Executor-provided CPU chunks for omitted base parameters only.
Restore requires the same identity and explicit loader, even in an initialized
worker. Every retained tensor must come from the checkpoint; a missing adapter,
buffer or optimizer tensor is an error. Older Programs without trainability
metadata can still save full checkpoints, but must be rebuilt to omit frozen
parameters. Named evaluation Plans remain read-only with respect to shared model
tensors.

## Reuse preparation across budgets and processes

Save Program and Profiles independently, or save a self-contained ProfiledProgram.
Reload and replan it with different `MemoryBudgets` or `CommunicationInfo` on CPU.
The resulting Plan is an executable archive that a fresh Executor can load with
matching runtime/provider builds. See [budget sweeps](../../examples/reusable-planning.md).

## Data parallel training and resource ownership

All builders take one `parallelism` argument. `Parallelism.dp(...)` describes
logical model replicas and weight/gradient/optimizer storage sharding; actual
process groups are bound later from the saved declarations and a complete rank
map. Simple DP compute profiling uses one representative device. Distributed
execution is qualified for the common sharded-DP path; broader combinations
remain separate qualification work. See [distributed status](distributed.md).

## Startup diagnostics

`Trainer.diagnose(args=..., kwargs=..., directory="startup", warmup=1)` runs the
saved Plan at zero learning rate and writes measured and simulated timelines.
It requires the optimizer used by `build_step` to explicitly guarantee
`zero_lr_preserves_state=True`, with a captured CPU tensor learning rate for every
parameter group. MLOps AdamW provides this guarantee; ordinary PyTorch AdamW
does not. Unsupported Plans fail before dispatching a diagnostic update.

Only affected model buffers are saved in a temporary file, using bounded CPU
spill views. Parameters, master weights, moments and counters stay in place.
Diagnostics restore model buffers, runtime controls and default RNG, and do not
advance Trainer counters, schedules or input position. Their time is excluded
from Trainer's elapsed training time. Indexed input loaders must be repeatable.
GPU execution failures still make Executor unusable, as for any invocation.

`Trainer.fit(..., startup_diagnostics=True, run_dir=...)` diagnoses its first
input and then reuses it for training. The option defaults off. Each rank writes
its own directory in distributed execution. The JSON text runner accepts the
same `startup_diagnostics` option.

For a trace of an actual invocation, use
`pending = executor.run(..., runtime_trace=True)` and `pending.diagnostics`.
Reading diagnostics waits for completion and returns `StepDiagnostics` with
task, allocation and transfer records; `.result()` retains its normal output.
Untraced calls return `None` for diagnostics and allocate no trace resources.
Unlike `Trainer.diagnose`, tracing alone does not preserve training state.

## Custom optimizers

Custom optimizers must expose a tensor-only, capturable update. Python control
flow that reads tensor values and Python step hooks are rejected during build;
there is no eager optimizer fallback in a saved Plan. Capture failures leave the
caller's optimizer and parameters unchanged.
