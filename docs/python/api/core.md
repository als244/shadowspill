# Framework-neutral Python API

These modules expose the IR, planner, simulator, and physical-admission values
used by the PyTorch frontend and by standalone tooling.

## `shadowspill.ir`

Logical program values:

- `Program`, `TaskSpec`, `TaskProfile`
- `ObjectSpec`, `ObjectRole`, `Persistence`, `AliasGroupSpec`, `MutationSpec`
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

Call `pressurefit()` to select recomputation and a memory schedule. Call
`validate_schedule_feasibility()` to check whether at least one legal Program
selection satisfies the required task-by-task residency floor. Use
`simulate()` to validate an explicit schedule. See the [PressureFit
formulation and algorithm](../../architecture/pressurefit.md) and the separate
[recomputation selector](../../architecture/recomputation-selection.md). The
task-allocation topology and exact range certificate are described in
[physical admission](../../architecture/physical-admission.md).

Configuration and results:

- `PressureFitOptions`, `PressureFitResult`, `PressureFitDiagnostics`
- `InitialPlacement`, `AdmissionRefinement`
- `AdmissionTopology`, `StorageHandoff`, `TaskAdmissionSpec`
- `TaskAllocationStep`, `TaskAllocationStepKind`

Search diagnostics:

- `RecomputationChoiceDiagnostic`, `RecomputationContextDiagnostics`
- `CandidateDiagnostic`
- `PressureFitRepairDiagnostics`, `PressureFitWorkDiagnostics`

Failures are `PressureFitInfeasibleError` or
`PressureFitSearchExhaustedError`.

## `shadowspill.simulator`

`simulate()` replays one explicit schedule through the required compiled
simulator. Missing or ABI-incompatible compiled libraries fail immediately.

Configuration and results:

- `SimulationConfig`, `DeviceSimulationConfig`, `SimulationAdmission`
- `SimulationResult`, `SimulationInfeasibleError`
- `TaskInterval`, `TransferInterval`, `TransferDirection`
- `MemorySnapshot`, `DeviceMemoryPeak`
- `ActionPhysicalDelta`, `TaskPhysicalDelta`, `MemoryReuseDependency`

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
