# Framework-neutral Python API

These modules expose the IR, planner, simulator, and physical-admission values
used by the PyTorch frontend and by standalone tooling.

## `shadowspill.ir`

Logical program values:

- `Program`, `TaskSpec`, `TaskProfile`
- `ObjectSpec`, `ObjectRole`, `Persistence`, `AliasGroupSpec`, `MutationSpec`
- `SharedResidencyPolicy`
- `SharedResidencyFootprint`, `shared_residency_footprint()`
- `ResourceSpec`, `ResourceKind`, `DeviceSpec`
- `RecomputationGroup`, `RecomputationOption`, `RecomputationSelection`

Scheduling and execution values:

- `MemorySchedule`, `MemoryAction`, `MemoryActionKind`, `MemoryLocation`
- `ResidencySpec`
- `ExecutionPlan`, `EntrypointSpec`, `PhysicalAdmission`, `PlanPrediction`

Indexed compiled projections:

- `IndexedProgram`, `IndexedMemorySchedule`, `IndexedExecutionPlan`
- `index_program()`, `index_memory_schedule()`,
  `index_execution_plan()`

Invalid construction or cross-reference raises `ValidationError`.

## `shadowspill.planner`

Call `plan_program()` to plan a Program: it expands the Program into its
resolved programs — one concrete task set per way of fixing the save/recompute
alternatives — and plans each. `pressurefit()` wraps it and is what most
callers want. Both accept `placement` facts beside `admission`: `placement`
lets each candidate measure whether its plan has a layout that fits the pool,
while `admission` switches on the dynamic-pool replay, which rejects schedules
that certified fixed placement accepts. PressureFit takes *one or more*
resolved programs in a single call and answers each of them, knowing only
tasks, runtimes, object accesses, budgets and bandwidths. Deciding which
resolved programs exist, and the order they are searched in, belongs to
`plan_program()` above it — the order it passes them in is the order they are
searched, and passing them together is what lets a plan placed under one bound
the search under the rest.

Call `validate_schedule_feasibility()` to check whether at least one legal
Program selection satisfies the required task-by-task residency floor. Use
`simulate()` to validate an explicit schedule. See the [PressureFit
formulation and algorithm](../../architecture/pressurefit.md) and the separate
[recomputation selector](../../architecture/recomputation-selection.md). The
task-allocation topology and exact range certificate are described in
[physical admission](../../architecture/physical-admission.md), and how a
schedule becomes leases in [from a resolved program to
leases](../../architecture/admission-leases.md).

Every step of physical admission runs in the library:
`pressurefit()` selects candidates there, the operations a schedule implies
are derived there, and each lease is placed at a fixed offset there. Missing
or ABI-incompatible libraries fail immediately rather than falling
back, and the readable Python equivalents live outside the package in
`reference/python/`, where production never imports them.

`PressureFitOptions.workers` sizes the threads the library runs the search on.
Python owns no threads: one call gets its own workers, so two callers planning
at the same time do not contend. The unit of work is one (resolved program,
candidate) pair, which is why worker count and resolved-program count are
independent — eight workers means eight threads whether the call was given one
resolved program or five. Zero takes one worker per logical CPU; one evaluates
every pair on the calling thread. It is a scheduling choice, not a search
input: it does not change which plans are legal or how they simulate, though
it does change how many candidates get skipped against the shared placement
record, so per-candidate counters move with it.

`PressureFitOptions.capacity_refinement_bytes` sets how much capacity a plan
gives back at a time when its layout does not fit the pool. It defaults to
256 MiB: the extent does not shrink byte for byte with the capacity, so
handing back the whole shortfall overshoots the capacity that would have fit.
Setting it to zero does hand back the whole shortfall, which converges in the
fewest rounds and is the setting to reach for when planning time matters more
than the last percent of makespan.

`PressureFitOptions.record_reduction_steps` asks each candidate to record its
reduction trajectory — one `ReductionStep` per plan it held. Off by default:
it costs an allocation per candidate that grows with the search, which is
worth paying to attribute planner time or explain a plan, and not worth
paying in a sweep.

Configuration and results:

- `PressureFitOptions`, `PressureFitResult`, `PressureFitDiagnostics`
- `InitialPlacement`
- `AdmissionFacts`, `StorageHandoff`, `TaskAdmissionSpec`
- `TaskAllocationStep`, `TaskAllocationStepKind`

Search diagnostics:

- `RecomputationChoiceDiagnostic`, `RecomputationProblemDiagnostics`
- `CandidateDiagnostic`
- `PressureFitRepairDiagnostics`, `PressureFitWorkDiagnostics`
- `PressureFitSectionTiming`, `ReductionStep`

`PressureFitWorkDiagnostics` counts what the search did and carries a
`PressureFitSectionTiming` saying where the time went. Sections are disjoint
spans named by the function that opened them, so `total_ns` equals `named_ns`
plus `residual_ns` at every level of the hierarchy — candidate, resolved
program, and whole call. `admit_ns` is the one exception: admission runs as
part of simulating, so it is nested inside `simulate_ns` rather than beside
it. Summing two of these adds every section, which is how the aggregate is
built.

Sections measure work rather than elapsed time, so with several workers a
resolved program's total exceeds the time the call took. `started_ns` and
`finished_ns`, on both `CandidateDiagnostic` and
`RecomputationProblemDiagnostics`, are the elapsed-time counterpart:
nanoseconds from the start of the call, shared by every span in it, so two
candidates ran at the same time exactly when their spans overlap.

`ReductionStep` is one plan a candidate held: its makespan, the bytes its
layout needed, the capacity it was built against, the objects the reducer
cut to reach it, and what became of it — simulated, measured, placed,
refined, best so far, or the answer. `CandidateDiagnostic.steps` is the whole
trajectory in order, and is empty unless the caller asked for it.

Failures are `PressureFitInfeasibleError` or
`PressureFitSearchExhaustedError`.

## `shadowspill.simulator`

`simulate()` replays one explicit schedule through the required compiled
simulator. Missing or ABI-incompatible libraries fail immediately.

Configuration and results:

- `SimulationConfig`, `DeviceSimulationConfig`, `SimulationAdmission`
- `SimulationResult`, `SimulationInfeasibleError`
- `TaskInterval`, `TransferInterval`, `TransferDirection`
- `MemorySnapshot`, `DeviceMemoryPeak`, `CapacityViolation`
- `ActionPhysicalDelta`, `TaskPhysicalDelta`, `MemoryReuseDependency`

A prefetch or task launch with nowhere to go waits for room rather than
failing, so a plan that comes up short is slower rather than rejected. Each
shortfall is reported as a `CapacityViolation` alongside the `device-capacity`
stall that records the wait: the stall says when and for how long, the
violation says by how much.

## `shadowspill.runtime`

The standalone physical-admission helpers are:

- `AdmissionPolicy`, `AdmissionError`
- `AllocationEvent`, `AllocationOperation`
- `SlabPlacement`, `SlabLayout`, `SlabReplay`
- `workspace_reserve_bytes()`
- `plan_slab_layout()`, `replay_slab_timeline()`,
  `admit_physical_budget()`

The production-memory-pool replay interface is:

- `AdmissionReplayOperation`, `AdmissionReplayOperationKind`
- `AdmissionReplayLeaseState`, `AdmissionReplayDecision`
- `AdmissionReuseDependency`, `AdmissionReplayResult`
- `run_admission_replay()`

`ObjectRef` is the framework-neutral retained handle for one runtime-global
logical object. Object identity is independent of its current pool lease or
residency generation. Framework integrations layer their own view metadata on
this handle and call `ObjectRef.close()` to release public ownership.
`ObjectConsistency` selects causal generation/readiness ordering or an
explicitly unordered cross-plan view for a plan binding.
