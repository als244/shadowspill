# Reusable planning and budget sweeps

Preparing computation, choosing a plan and admitting a callable are separate
operations. Prepare once at the largest intended capacities; search many budget
pairs; admit the chosen plan inside a process with that budget's actual pools.

## Trainer artifacts across processes

The same model factory, objective and microbatch functions must be supplied to
both trainers. `make_model`, `objective`, `candidates` (named split functions) and `example` below are
application inputs; the [main examples](../../README.md#train-with-the-trainer-api)
provide real text workloads. Budget values passed to search are **bytes**.

```python
import torch
from shadowspill.training import PreparedTraining, Trainer, reset_parameters
from shadowspill.training.backends import ShadowSpill

budgets = [(8 << 30, 144 << 30), (16 << 30, 144 << 30), (28 << 30, 144 << 30)]
trainer = Trainer(
    make_model(), objective=objective, optimizer=torch.optim.AdamW,
    microbatch_candidates=candidates,
    backend=ShadowSpill(execution_gib=28, spill_gib=144),
)
prepared = trainer.prepare_artifacts(
    example, budgets=budgets, initialize=reset_parameters,
)
prepared.save("prepared/example")
trainer.close()

for budget, plan in prepared.search.winner_plans.items():
    print(budget, plan.summary.as_dict())
```

That process may exit. A separate case loads the artifact, initializes its own
weights and optimizer, and admits only its selected plan:

```python
prepared = PreparedTraining.load("prepared/example")
trainer = Trainer(
    make_model(), objective=objective, optimizer=torch.optim.AdamW,
    microbatch_candidates=candidates,
    backend=ShadowSpill(execution_gib=8, spill_gib=144),
)
trainer.prepare(example, initialize=reset_parameters, prepared=prepared)
trainer.fit(source, steps=5, run_dir="runs/budget-8")
trainer.close()
```

This retains independent initialization, timing and result ownership for each
case. It skips export and search during admission. Task binding, profile lookup,
selected-kernel loading and physical certification still happen locally. The
first executed backward creates each gradient; subsequent backward occurrences
accumulate it, regardless of which microbatch the selected ordering runs first.

## Lower-level prepare, search and admit

For direct runtime ownership, replace `plan_step` in the
[training loop](training-lifecycle.md) with:

```python
from shadowspill.pytorch import prepare_step, search_prepared_steps, admit_step

prepared = prepare_step(
    model, objective=objective, optimizer=torch.optim.AdamW, hyperparams=("lr",),
    example_inputs=[batch(4)], runtime=runtime, execution="execution", spill="spill",
)
prepared.save("prepared/step")
search = search_prepared_steps(
    {"batch4": prepared}, budgets=[(3 << 30, 2 << 30), (4 << 30, 2 << 30)],
)
selected = search.winner_plans[(4 << 30, 2 << 30)]
print(selected.summary.as_dict())
train_step = admit_step(
    model, prepared, selected, optimizer=torch.optim.AdamW, hyperparams=("lr",),
    example_inputs=[batch(4)], runtime=runtime, execution="execution", spill="spill",
)
```

Loading `PreparedStep.load(...)` and searching new budgets within its original
capacity bounds needs no GPU. Loading selects that stored objective deliberately;
its weights come from the current model. Use the same artifact store for
preparation and admission. The supported persistent path is currently single rank.

## Planner-only program artifacts

Use `build_step_programs()` when capture, graph-pair construction, compilation,
profiling, and canonical program lowering should occur once. It returns one
`StepProgram` per ordering asked for, the depth-first one alone by default; each
can be serialized and its
`ShadowSpillPlanningProblem` planned repeatedly without executing the model or
repeating compiler work. Capturing one needs the frontend; planning one
again does not, so the second half of this example imports no torch.

```python
from pathlib import Path

import torch

from shadowspill.planner import plan_program
from shadowspill.planner.program import TransferBandwidths
from shadowspill.pytorch import build_step_programs
from shadowspill.step import StepProgram


(step_program,) = build_step_programs(
    model,
    objective=objective,
    optimizer=torch.optim.AdamW,
    hyperparams=("lr",),
    example_inputs=example_inputs,
    runtime=runtime,
    execution="execution",
    spill="spill",
    artifact_store=artifact_store,
    profiling_metadata=profiling_metadata,
)

program_path = Path("step-program.json")
program_path.write_text(step_program.to_json(), encoding="utf-8")
loaded = StepProgram.from_json(program_path.read_text(encoding="utf-8"))

points = [
    (3 << 30, 2 << 30, 24_000_000_000),
    (4 << 30, 2 << 30, 24_000_000_000),
    (4 << 30, 2 << 30, 36_000_000_000),
]

for execution_budget, spill_budget, bandwidth in points:
    annotated = plan_program(
        loaded.problem,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
        transfer_bandwidths=TransferBandwidths(
            fetch_solo_bytes_per_second=bandwidth,
            fetch_concurrent_bytes_per_second=bandwidth,
            evict_solo_bytes_per_second=bandwidth,
            evict_concurrent_bytes_per_second=bandwidth,
            provenance="explicit sweep",
        ),
        artifact_store=artifact_store,
    )
    output = Path(f"annotated-{annotated.digest}.json")
    output.write_text(annotated.to_json(), encoding="utf-8")
    print(execution_budget, annotated.simulation.makespan_ns, output)
```

`ShadowSpillPlanningProblem.machine_inputs()` and `plan_program()` reject a
budget larger than the runtime capacities the source artifact was compiled and
profiled under. Lower budgets and alternate transfer bandwidths do not change
the logical program. To hand a search a different algorithm or different
generic options, pass `search_options=SearchOptions(...)`; omitting it uses the
search that ships.

The split runs through the store arguments too. `build_step_programs()` takes
`artifact_store`, `build_store`, and `build_store_mode`, and no plan-store
arguments at all, because it writes no plans; with an `export_bypass_key` it
reads each ordering's program back from the build store's step archive and
captures only for the orderings not found there. `plan_program()` takes
`artifact_store`, `plan_store`, and `plan_store_mode`, and no build
arguments, because it builds nothing. Rooting the two trees separately is
what lets one build store serve many sweeps that each keep their own plans.

`AnnotatedProgramPlan` is planning evidence, not a standalone executable
callable. `plan_step()` publishes the callable and can reuse the same stores.
The [JSON guide](../python/planning-json.md) documents which content
participates in each digest and what is revalidated during loading.

For reproducible sweeps, store together:

- the `StepProgram` JSON;
- every `AnnotatedProgramPlan` JSON;
- the sweep configuration and source revision;
- the artifact store on a fast local filesystem;
- any explicit transfer-bandwidth provenance.
