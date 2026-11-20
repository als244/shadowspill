# Build, profile, plan and execute

For a supported model's complete training workflow, use the
[training client](client.md). This guide is
for callers who need separate phases or their own model/data integration.

ShadowSpill separates preparing a workload from running it. The phase functions
are available from `shadowspill`; `Executor` runs the selected Plan. The
[extension guide](advanced/writing-extensions.md) shows how to define
models, objectives and optimizers.

| Phase | Public API | Result | Needs GPU? |
| --- | --- | --- | --- |
| Build | `build_step` (or `build_forward` / `build_forward_backward`) | `Program`: captured calls and finalized task code | Yes |
| Profile tasks | `measure_program` → `apply_profiles` | `Profiles` → `ProfiledProgram`: measured tasks and memory | Measurement does; applying profiles is CPU-only |
| Measure communication | `measure_communication`, optionally `measure_overlap` | `CommunicationInfo` and overlap profiles | Yes, with the required participants |
| Plan | `plan_program` or `sweep_programs` | `Plan`: selected variants, schedule and memory layout | No |
| Execute | `Executor` | Results and optional checkpoints | Yes, in a fresh worker |

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

executor = ss.Executor(ss.Plan.load("training.ssp"))
executor.initialize(fill_tensor)
for batch in training_batches():
    result = executor.run(**batch).result()
    print([float(loss) for loss in result.losses])
    del result  # Release output views before the next invocation or shutdown.
executor.close()
```

`fill_tensor` fills bounded CPU views. Each item yielded by `training_batches()`
supplies one update's inputs, matching the Plan's fixed calls. The caller owns
its iteration count, schedule and data cursor. The [Executor reference](../reference/engine/execution.rst)
explains state loading, bounded checkpoints and sharing training/evaluation state.
Its default pinned-host pool uses the required spill peak plus 5%, capped by the
planned budget.

## Where options belong

| Change | Control | Work needed |
| --- | --- | --- |
| Model/objective/optimizer, precision or input shapes | [Build arguments](../reference/engine/build.rst) | Build and profile affected implementations, then plan |
| Microbatch count using already saved forms | `instantiate_examples` in [planning](../reference/engine/planning.rst) | Instantiate on CPU, reuse compatible profiles, then plan and admit |
| Checkpoint boundaries, split axes/chunk sizes or retained intermediates | `partition`, `chunking`, `graph_pair_policy` in [frontend policies](../reference/engine/frontend.rst) | Build/profile each candidate, then compare Plans |
| Timing windows, sample caps or external-memory headroom | [Profiling options](../reference/engine/profiling.rst) | Measure with the requested settings, apply profiles, then plan |
| Transfer backend, collective layout or communication costs | [Communication measurement](../reference/engine/profiling.rst) | Obtain compatible measurements, then plan; changed ownership can also require build |
| Device/spill budget, microbatch traversal, save-tail choices or search effort | [Planning options](../reference/engine/planning.rst) | Replan existing compatible profiles on CPU |
| Input values, captured learning-rate controls, logging or checkpoints | [Executor controls](../reference/engine/execution.rst) | Execute the chosen Plan |

Changing a retention policy changes **what each graph saves**. Planning separately
chooses **which graph runs** and **where saved values live**. Selecting a save
variant does not force GPU residency.

`sweep_programs` compares supplied ProfiledPrograms and budgets; it does not
create or profile candidates. Public-client [preparation settings](advanced/service-client.md)
expose microbatch, chunk and retention candidates. The generic phase APIs accept supported custom
models directly.

For multi-worker runs, declare [parallelism](../reference/engine/parallelism.rst)
at build, and measure communication and overlap on every worker. Check
`plan.simulation_only_reasons` before execution: estimated workspace alone
cannot establish a safe physical layout. Worker startup also checks
dependencies and available memory.

[Artifact reuse](advanced/artifact-store.md), [pool calibration](../reference/engine/profiling.rst),
[Plan reports](advanced/plan-report.md), [runtime diagnostics](advanced/step-diagnostics.md) and
[failure handling](advanced/failures.md) cover inspection and troubleshooting.
