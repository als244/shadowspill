# Reusable planning artifacts

Planning artifacts are immutable, content-addressed, JSON-serializable values.
Their stable digests exclude cache paths and measured orchestration wall time.
See [Program and annotated-plan JSON](../planning-json.md) for the complete
schema hierarchy and key-by-key interpretation.

## `StepProgram`

`StepProgram` is the result of `make_step_program()`. It contains:

- the recurrent `PressureFitProgram`;
- an optional initialization `PressureFitProgram`;
- optimizer ordering and input-signature digests;
- profiling metadata and structural-profile counts;
- transfer-capability evidence;
- phase timings and cache lineage.

Use `StepProgram.to_json()` and `StepProgram.from_json()` for a portable corpus.
`StepProgram.digest` identifies planning content rather than how long or where
it was produced.

```python
from pathlib import Path

from shadowspill.pytorch import StepProgram, make_step_program

step_program = make_step_program(
    model,
    objective=objective,
    opt=optimizer_factory,
    example_inputs=example_inputs,
    runtime=runtime,
    execution="device",
    spill="spill",
    planning_cachedir=planning_cache,
)
Path("program.json").write_text(step_program.to_json())

loaded = StepProgram.from_json(Path("program.json").read_text())
```

## `PressureFitProgram`

`PressureFitProgram` is a self-contained input to `pressurefit_program()`. It
contains the canonical `Program`, initial and final residency, simulation
configuration, `AdmissionFacts`, capacity contract, dynamic scratch
reserve, and `PressureFitOptions`.

`PressureFitProgram.pressurefit_inputs()` rebases execution budget, spill
budget, or `TransferBandwidths` without changing the Program. A requested
budget cannot exceed the runtime capacities used to compile and profile it.
Use `to_json()`, `from_json()`, or `from_value()` for serialization.

```python
from shadowspill.pytorch import TransferBandwidths, pressurefit_program

annotated = pressurefit_program(
    loaded.recurrent,
    execution_budget=16 << 30,
    spill_budget=96 << 30,
    transfer_bandwidths=TransferBandwidths(
        fetch_bytes_per_second=28_000_000_000,
        evict_bytes_per_second=28_000_000_000,
    ),
    planning_cachedir=planning_cache,
)
```

## `AnnotatedProgramPlan`

`AnnotatedProgramPlan` is a selected and physically admitted plan. It contains:

- source `PressureFitProgram`;
- `MemoryBudgets` and `TransferBandwidths`;
- full `PressureFitResult` and diagnostics;
- effective admission topology;
- fixed physical layout and layout digest;
- simulator admission and `SimulationResult`;
- every capacity-refinement attempt;
- separate PressureFit, physical-admission, orchestration, and total wall time.

`AnnotatedProgramPlan.to_json()` preserves the complete selected schedule and
diagnostic evidence. `from_json()` revalidates digests, residency, simulation,
layout identity, and timing reconciliation.

See [Physical admission and offset
handling](../../architecture/physical-admission.md) for the fixed layout,
dynamic reserves, causal reuse edges, and capacity-refinement contract.

## Small value objects

`MemoryBudgets` records physical execution and spill byte budgets.
`TransferBandwidths` records fetch and evict bytes per second, an optional
rational scaling factor, calibration digest, and provenance. Both participate
in annotated-plan identity.
