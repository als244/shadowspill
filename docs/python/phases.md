# Build, profile, plan and execute

For a supported model's complete training workflow, use the
[shared Python `train` runner](../../workloads/recipes/README.md). This guide is
for callers who need separate phases or their own model/data integration.

ShadowSpill separates preparing a workload from running it. The phase functions
are available from `shadowspill`; `Trainer` runs the selected Plan. Start with the
[complete Python example](../examples/generic-training.md), which defines the
model, objective, initializer and input loop.

| Phase | Public API | Result | Needs GPU? |
| --- | --- | --- | --- |
| Build | `build_step` (or `build_forward` / `build_forward_backward`) | `Program`: captured calls and finalized task code | Yes |
| Profile tasks | `measure_program` → `apply_profiles` | `Profiles` → `ProfiledProgram`: measured tasks and memory | Measurement does; applying profiles is CPU-only |
| Measure communication | `measure_communication`, optionally `measure_overlap` | `CommunicationInfo` and overlap profiles | Yes, with the required participants |
| Plan | `plan_program` or `sweep_programs` | `Plan`: selected variants, schedule and memory layout | No |
| Execute | `Trainer` or `Executor` | Results and optional checkpoints | Yes, in a fresh worker |

Each artifact can be saved and loaded independently. A Plan includes task code
and planning evidence; model values come from initialization or a checkpoint.
Compatible runtime/operator libraries must be installed in the execution worker.

## Minimal phase calls

Prepare with a representative GPU. The application defines the model, objective,
optimizer and example inputs:

```python
import shadowspill as ss

program = ss.build_step(
    model, optimizer=optimizer, objective=objective,
    example_args=(inputs, targets), num_examples=4,
)
profiles = ss.measure_program(program)
ss.apply_profiles(program, profiles).save("profiled.ssp")
ss.measure_communication(program=program).save("communication.ssp")
```

`num_examples=4` means four complete rank-local microbatch calls per update.
If each input batch has 16 samples, the update has 64 samples per rank. The
objective supplies the corresponding loss normalization. Build accepts CPU/meta
model state but still uses a GPU to finalize code; there is no separate CPU-only
prepare phase. Neither build nor compute profiling opens execution/spill pools.

Plan on CPU, reusing those artifacts:

```python
plan = ss.plan_program(
    ss.ProfiledProgram.load("profiled.ssp"),
    memory_budgets=ss.MemoryBudgets(execution_bytes=24 << 30, spill_bytes=128 << 30),
    communication_info=ss.CommunicationInfo.load("communication.ssp"),
)
plan.save("training.ssp")
ss.write_report(plan, "report")
```

Execute in a **fresh process**, before another device allocator initializes:

```python
import shadowspill as ss

with ss.Trainer(ss.Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)  # Or trainer.load("checkpoint.ssp") to resume.
    trainer.fit(training_batches(), steps=100)
```

`fill_tensor` fills bounded CPU views. Each item yielded by `training_batches()`
supplies one update's inputs, matching the Plan's fixed calls. `steps=100` is the total target
count, including restored updates. Trainer never builds, profiles or searches.
Its default pinned-host pool uses the required spill peak plus 5%, capped by the
planned budget.

## Where options belong

| Change | Control | Work needed |
| --- | --- | --- |
| Model/objective/optimizer, precision, input shapes or microbatch count | [Build arguments](api/build.md) | Build and profile affected implementations, then plan |
| Checkpoint boundaries, split axes/chunk sizes or retained intermediates | `partition`, `chunking`, `graph_pair_policy` in [frontend policies](api/frontend.md) | Build/profile each candidate, then compare Plans |
| Timing windows, sample caps or external-memory headroom | [Profiling options](api/profiling.md) | Measure with the requested settings, apply profiles, then plan |
| Transfer backend, collective layout or communication costs | [Communication measurement](api/communication.md) | Obtain compatible measurements, then plan; changed ownership can also require build |
| Device/spill budget, microbatch traversal, save-tail choices or search effort | [Planning options](api/planning.md) | Replan existing compatible profiles on CPU |
| Input values, captured learning-rate controls, logging or checkpoints | [Trainer options](api/training.md) | Execute the chosen Plan |

Changing a retention policy changes **what each graph saves**. Planning separately
chooses **which graph runs** and **where saved values live**. Selecting a save
variant does not force GPU residency.

`sweep_programs` compares supplied ProfiledPrograms and budgets; it does not
create or profile candidates. The source-tree [recipe runner](../../workloads/recipes/README.md)
and [quickstart](../../benchmarking/quickstart.md) can construct candidate
microbatch sizes, chunk sizes and retention policies, then search their Plans.
The generic phase APIs accept arbitrary supported models; recipes provide
application-specific model/data setup around them.

For multi-worker runs, follow [distributed setup](api/distributed.md), including
communication and overlap calibration. Check `plan.simulation_only_reasons`
before execution: estimated workspace alone cannot establish a safe physical
layout. Worker startup also checks dependencies and available memory.

[Artifact reuse](advanced/artifact-store.md), [pool calibration](api/pool-memory.md),
[Plan reports](advanced/plan-report.md), [runtime diagnostics](advanced/step-diagnostics.md) and
[failure handling](advanced/failures.md) cover inspection and troubleshooting.
