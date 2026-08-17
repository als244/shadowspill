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

Dense compiled projections:

- `DenseProgram`, `DenseMemorySchedule`, `DenseExecutionPlan`
- `project_dense()`, `project_dense_schedule()`,
  `project_dense_execution_plan()`

Invalid construction or cross-reference raises `ValidationError`.

## `shadowspill.planner`

Call `pressurefit()` to select recomputation and a memory schedule. Call
`validate_schedule_feasibility()` to check a supplied schedule against the
same logical constraints.

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

`simulate()` replays one explicit schedule. It uses the compiled simulator
unless `record_timeline=True` requests the Python diagnostic timeline.

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
