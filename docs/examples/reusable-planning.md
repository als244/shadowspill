# Reusable planning and budget sweeps

Use `make_step_program()` when capture, graph-pair construction, compilation,
profiling, and canonical Program lowering should occur once. The resulting
`StepProgram` can be serialized and its recurrent or initial
`PressureFitProgram` evaluated repeatedly without executing the model or
repeating compiler work.

```python
from pathlib import Path

from shadowspill.pytorch import (
    StepProgram,
    TransferBandwidths,
    make_step_program,
    pressurefit_program,
)

step_program = make_step_program(
    model,
    objective=objective,
    opt=optimizer_factory,
    example_inputs=example_inputs,
    runtime=runtime,
    execution="execution",
    spill="spill",
    artifact_store_dir=artifact_store,
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
    annotated = pressurefit_program(
        loaded.recurrent,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
        transfer_bandwidths=TransferBandwidths(
            fetch_bytes_per_second=bandwidth,
            evict_bytes_per_second=bandwidth,
            provenance="explicit sweep",
        ),
        artifact_store_dir=artifact_store,
    )
    output = Path(f"annotated-{annotated.digest}.json")
    output.write_text(annotated.to_json(), encoding="utf-8")
    print(execution_budget, annotated.simulation.makespan_ns, output)
```

`PressureFitProgram.pressurefit_inputs()` and `pressurefit_program()` reject a
budget larger than the runtime capacities used to compile/profile the source
artifact. Lower budgets and alternate transfer bandwidths do not change the
logical Program.

`AnnotatedProgramPlan` is planning evidence, not a standalone executable
callable. `plan_step()` publishes the callable and can reuse the same planning
cache. The [JSON guide](../python/planning-json.md) documents which content
participates in each digest and what is revalidated during loading.

For reproducible sweeps, store together:

- the `StepProgram` JSON;
- every `AnnotatedProgramPlan` JSON;
- the sweep configuration and source revision;
- the planning cache on a fast local filesystem;
- any explicit transfer-bandwidth provenance.
