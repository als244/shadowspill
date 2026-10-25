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

An optional existing `executor=` must belong to the same Plan. Trainer closes
only an Executor it created. `close()` and context-manager exit wait for work and
release owned resources. By default, Executor inherits the external-memory
headroom recorded during profiling and included in the Plan's total GPU budget.
An `external_headroom_bytes` override cannot exceed that planned reservation.

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

`hyperparams` supplies per-call optimizer controls declared during build. A
`schedules` mapping passed to Trainer maps those same names to
`callable(completed_steps)`. Explicit per-call values override schedule outputs.
Only captured runtime controls can change; a Python constant baked into an
optimizer graph cannot become a runtime control later.

`fit(source, steps=...)` consumes mappings of `step` keyword arguments. Optional
arguments configure logging, callbacks and checkpoint cadence. An input source
can implement `state_dict`/`load_state_dict` for checkpointed progress. Custom
loops can call `step`, `evaluate`, `save` and `load` directly. A `metric_reducer`
can aggregate completed `StepObservations` for application-specific reporting.

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
execution remains guarded pending the separate memory/completion qualification;
see [distributed status](distributed.md).

## Startup diagnostics

The phase runners save build/profile/plan archives and execute in fresh workers.
Runtime timing/trace collection observes an already admitted Executor. It does
not trigger preparation or use a zero-learning-rate update as an implicit no-op.
A warmup that must leave state unchanged needs checkpoint restore or a backend
with an explicit no-update contract. Qualification checks physical budget sealing
and steady-state allocation/event growth separately from numerical comparisons.

## Callers still being migrated

`DataParallel`, `Distributed`, `TransferSharding`, `Forward`, `PreparedTraining`
and `reset_parameters` remain importable for repository callers using the older
preparation/backend path. They are not arguments to the saved-plan Trainer.
`Forward` and `PreparedTraining` will be removed after those callers migrate;
new code should use `build_forward`, saved artifacts and Executor. The private
compatibility Trainer is not the public Trainer documented above.
