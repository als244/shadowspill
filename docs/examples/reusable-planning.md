# Reusable planning and budget sweeps

Use `build_step_program()` when capture, graph-pair construction, compilation,
profiling, and canonical program lowering should occur once. The resulting
`StepProgram` can be serialized and its recurrent or initial
`ShadowSpillPlanningProblem` planned repeatedly without executing the model or
repeating compiler work. Capturing one needs the frontend; planning one
again does not, so the second half of this example imports no torch.

```python
from pathlib import Path

import torch

from shadowspill.planner import plan_program
from shadowspill.planner.program import TransferBandwidths
from shadowspill.pytorch import build_step_program
from shadowspill.step import StepProgram


def zero_state(
    name: str, tensor: torch.Tensor, parameter: torch.nn.Parameter
) -> None:
    """Moment-based optimizers start at zero; ShadowSpill never assumes it."""

    with torch.no_grad():
        tensor.zero_()


step_program = build_step_program(
    model,
    objective=objective,
    optimizer=torch.optim.AdamW,
    hyperparams=("lr",),
    optimizer_state_init=zero_state,
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
        loaded.recurrent,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
        transfer_bandwidths=TransferBandwidths(
            fetch_bytes_per_second=bandwidth,
            evict_bytes_per_second=bandwidth,
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

The split runs through the store arguments too. `build_step_program()` takes
`artifact_store`, `build_store`, and `build_store_mode`, and no plan-store
arguments at all, because it writes no plans. `plan_program()` takes
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
