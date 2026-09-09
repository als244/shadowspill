# Reusable planning artifacts

The values that cross the boundary between building a program, planning it,
and running it. They live in `shadowspill.planner.program`, are immutable,
content-addressed and JSON-serializable, and their stable digests exclude
store paths and measured orchestration wall time, so where and how long
something took never changes what it is.

Three of them form one chain:

```text
build_step_program()  ->  StepProgram  (.recurrent is a ShadowSpillPlanningProblem)
plan_program()        ->  AnnotatedProgramPlan
```

See [program and annotated-plan JSON](../planning-json.md) for the complete
schema hierarchy and key-by-key interpretation, and [the artifact
store](../artifact-store.md#identity) for what each digest holds.

## `StepProgram`

The result of [`build_step_program()`](frontend.md#build_step_program):
everything capture, profiling and canonical lowering produced for one step,
with nothing planned yet. It records the `StepDataOrdering` the step was
lowered with, since a different walk is a different program. It contains:

- `recurrent`, the `ShadowSpillPlanningProblem` for the repeated step;
- `initial`, an optional `ShadowSpillPlanningProblem` for a first step that has lazy
  optimizer state to create, or `None`;
- `optimizer_ordering` and `data_ordering`;
- `signature_digests`, one per microbatch input signature;
- `profiling_metadata`, `unique_profile_count` and `captured_stage_count`;
- `transfer_capabilities_json`, the calibration the program was measured under;
- `phase_timings_ns`, `store_directories` and `cache_artifacts`.

`StepProgram.digest` identifies planning content rather than how long or where
it was produced. `to_json()` and `from_json()` carry it as a portable corpus.

```python
from pathlib import Path

from shadowspill.planner.program import StepProgram
from shadowspill.pytorch import build_step_program

step_program = build_step_program(
    model,
    objective=objective,
    optimizer=build_optimizer,
    example_inputs=example_inputs,
    runtime=runtime,
    execution="device",
    spill="spill",
    artifact_store=artifact_store,
)
Path("program.json").write_text(step_program.to_json())

loaded = StepProgram.from_json(Path("program.json").read_text())
```

## `ShadowSpillPlanningProblem`

A self-contained problem: the only input `plan_program()` needs. It holds its
`role` (`"recurrent"`, `"initial"` or `"forward"`), the canonical `ShadowSpillProgram`,
initial and final residency, the `SimulationConfig` describing the machine, the
`AdmissionFacts`, the budgets it was profiled under and the maxima it may be
replanned within, and the fixed, object-reserve and dynamic-scratch byte
deductions that reconcile a budget with a pool capacity.

It carries no `SearchOptions`. A program states what problem it is, and how
to search that problem belongs to whoever plans it, so options are passed to
`plan_program()` and a saved program answers any of them. This is also what
keeps a saved program readable when the planner gains an option: nothing about
the measured problem changed, so its identity does not move.

```text
ShadowSpillPlanningProblem.pressurefit_inputs(
    *,
    execution_budget_bytes=None,
    spill_budget_bytes=None,
    transfer_bandwidths=None,
) -> tuple[SimulationConfig, AdmissionFacts]
```

Rebases the budget-dependent inputs without changing the program: each argument
defaults to what the program itself records, and the pair that comes back is
what a search at those budgets and rates plans against. A requested budget
cannot exceed the runtime capacity the program was compiled and profiled under,
and one that leaves no positive pool or object capacity is refused. Use
`to_json()`, `from_json()`, or `from_value()` for serialization.

```python
from shadowspill.planner import plan_program
from shadowspill.planner.program import TransferBandwidths

annotated = plan_program(
    loaded.recurrent,
    execution_budget=16 << 30,
    spill_budget=96 << 30,
    transfer_bandwidths=TransferBandwidths(
        fetch_bytes_per_second=28_000_000_000,
        evict_bytes_per_second=28_000_000_000,
    ),
    artifact_store=artifact_store,
)
```

## `AnnotatedProgramPlan`

A selected and physically admitted plan: the answer `plan_program()` gives, and
what a callable runs. It contains:

- `program`, the source `ShadowSpillPlanningProblem`;
- `memory_budgets` and `transfer_bandwidths`, the inputs it was planned for;
- `result`, the full `ProgramPlanResult` with its diagnostics;
- `effective_facts`, the admission topology in force;
- `fixed_layout`, the fixed physical layout, and its digest;
- `simulation_admission` and `simulation`, the simulator's admission and
  `SimulationResult`;
- `attempts`, every capacity-refinement attempt;
- `plan_from_store`, whether it was read back rather than planned here;
- `wall_time_ns`, with `search_wall_time_ns`,
  `physical_admission_wall_time_ns` and `orchestration_wall_time_ns` splitting
  it.

`to_json()` preserves the complete selected schedule and diagnostic evidence.
`from_json()` revalidates digests, residency, simulation, layout identity, and
timing reconciliation. `digest` excludes store and wall-time evidence, and is
computed once and kept, because serializing a whole plan is expensive and
several callers want the same answer.

See [Physical admission and offset
handling](../../architecture/physical-admission.md) for the fixed layout,
dynamic reserves, causal reuse edges, and capacity-refinement contract.

## Small value objects

`MemoryBudgets` records the physical `execution_bytes` and `spill_bytes` one
plan was made for. `TransferBandwidths` records `fetch_bytes_per_second` and
`evict_bytes_per_second`, an optional rational scaling factor
(`scale_numerator`, `scale_denominator`), optional `fetch_latency_ns` and
`evict_latency_ns`, and a `calibration_digest` and `provenance` naming where
the measurement came from. The latencies are optional so a record written
without them still reads, and so an override that names only bandwidths leaves
a program's own latencies in place. Both participate in annotated-plan
identity.
