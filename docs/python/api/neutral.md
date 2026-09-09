# Framework-neutral Python API

These modules expose the IR, planner, simulator, and physical-admission values
used by the PyTorch frontend and by standalone tooling. Nothing here imports a
framework, so a saved program can be planned, simulated, and admitted with no
model and no device present.

## `shadowspill.errors`

What planning raises when it cannot answer. `PlanningError` is the base, and
callers catch it to mean "planning did not produce a plan" without caring which
phase gave up.

| Exception | Base | Raised when |
|---|---|---|
| `PlanningError` | `RuntimeError` | Base for every planning failure. |
| `CaptureError` | `PlanningError` | The frontend cannot represent the requested graph. |
| `TaskPhaseError` | `PlanningError` | A phase failed on one task. Carries `structural_contract`, `task_kind`, and `operators`. |
| `CompilationError` | `TaskPhaseError` | A captured structural task cannot be compiled. |
| `ProfilingError` | `TaskPhaseError` | An isolated task cannot be measured or audited. |
| `AdmissionError` | `PlanningError` | Requested memory cannot be physically admitted. |
| `PlanInfeasibleError` | `AdmissionError` | No schedule satisfies the declared constraints. Carries `kind`, `device_id`, `boundary_task_id`, `required_bytes`, `capacity_bytes`, and the rejecting `diagnostics`. |
| `PlanSearchExhaustedError` | `PlanningError` | A bounded search stopped without proving feasibility or infeasibility -- a different claim from `PlanInfeasibleError`. Carries `diagnostics`. |
| `ObjectiveError` | `PlanningError` | An objective does not satisfy the training contract. |
| `InputGuardError` | `ValueError` | Runtime inputs differ from the fixed template. Raised at invocation, before any mutation, which is why it is a `ValueError` rather than a planning failure. |

Compiling and profiling both fail per task and both want to say which one, so
they share `TaskPhaseError` rather than each inventing the same three fields.
These errors carry no framework types, and they live here rather than beside
any one phase because the planner has to raise and catch them without
importing a framework.

## `shadowspill.ir`

Logical program values:

- `ShadowSpillProgram`, `TaskSpec`, `TaskProfile`
- `ObjectSpec`, `ObjectRole`, `Persistence`, `AliasGroupSpec`, `MutationSpec`
- `SharedResidencyPolicy`
- `SharedResidencyFootprint`, `shared_residency_footprint()`
- `ResourceSpec`, `ResourceKind`, `DeviceSpec`
- `TaskAlternativeGroup`, `TaskAlternativeOption`, `TaskAlternativeChoice`

Scheduling and execution values:

- `MemorySchedule`, `MemoryAction`, `MemoryActionKind`, `MemoryLocation`
- `ResidencySpec`
- `ExecutionPlan`, `EntrypointSpec`, `PhysicalAdmission`, `PlanPrediction`

Indexed compiled projections:

- `IndexedProgram`, `IndexedMemorySchedule`, `IndexedExecutionPlan`
- `index_program()`, `index_memory_schedule()`, `index_execution_plan()`

`shared_residency_footprint(program)` returns the `SharedResidencyFootprint`
of objects a program keeps resident on every device throughout, which is what
the capacity a plan is searched against has already been reduced by. The three
`index_*()` functions project a value into its compiled counterpart once, so a
caller that simulates or plans the same program repeatedly pays the projection
once. Invalid construction or cross-reference raises `ValidationError`.

## `shadowspill.planner`

### Two layers

Planning is two layers, and which one a caller wants depends on whether they
want the answer written down.

| Call | Takes | Returns | Store |
|---|---|---|---|
| `plan_program()` | a `ShadowSpillPlanningProblem` -- self-contained, with its budgets and bandwidths | `AnnotatedProgramPlan` | the planning tree |
| a `SearchAlgorithm` | a `ShadowSpillProgram`, its residency, and the machine to plan it against | `ProgramPlanResult` | none |

`plan_program()` is the entry point, and the only one that touches disk. It
fixes the machine from a budget, keys the answer, runs a search, holds that
search to any plan it was handed, and physically admits the winner.

The search that ships is `PressureFit`. It receives a program with its
alternatives still open, expands it into resolved programs -- one concrete
task set per way of fixing the alternatives -- plans each, and answers with
the best. It knows only tasks, runtimes, object accesses, budgets and
bandwidths.

Building a program is the frontend's job:
[`build_step_program()`](frontend.md#build_step_program) captures, compiles,
profiles and lowers one, and takes only build-store arguments. `plan_program()`
plans one and takes only plan-store arguments. Nothing on this page needs a
device or a model.

### `SearchAlgorithm`

The base class a search is. Subclass it, give it a `name`, implement
`__call__`, and pass an instance -- there is no registry and no name to
reserve, so a search written outside ShadowSpill is a first-class one.

| member | kind | meaning |
|---|---|---|
| `name` | `ClassVar[str]` | The stable string the plan key records. Not the class's name, so renaming or moving the class leaves a stored corpus reachable. |
| `options` | `OptionRecord` | This search's own options, carried into the plan key whole and read by nothing outside the search. |
| `preflight(...)` | method | Refuse a machine no schedule can fit, before a search is paid for. Defaults to saying nothing, which is always correct. |
| `__call__(...)` | abstract | Answer with a `ProgramPlanResult`. The only method a subclass must write. |

[Writing a search algorithm](../../architecture/search-algorithm.md) is the
reference: every argument with its type and meaning, the defaults, and a
worked example. [Plan search](../../architecture/search.md) states what the
planner promises in return.

### `SearchOptions`

The whole of what one planning call is told about searching, in one
argument so neither half can be set without the other being visible.

| field | type | default | meaning |
|---|---|---|---|
| `generic` | `GenericPlanningOptions` | all defaults | What any search understands |
| `algorithm` | `SearchAlgorithm` \| `None` | `None` | The search itself, holding its own options. `None` runs the one that ships. |
| `workers` | `int` | `0` | Threads the search may use; `0` is every logical CPU, `1` forces serial |

`generic` and `algorithm` are both part of the plan key. **`workers` is
not** -- it says how much machine to spend, not what to decide, so two runs
at different worker counts ask the same question and read back the same
answer. The plan report records what was used.

`GenericPlanningOptions` holds `deterministic` (default `False`) and
`minimum_object_bytes_evict_eligible` (default `1 << 20`).

```python
plan = plan_program(
    problem,
    search_options=SearchOptions(
        generic=GenericPlanningOptions(deterministic=True),
        algorithm=PressureFit(PressureFitOptions(max_repair_attempts=256)),
        workers=8,
    ),
)
```

### `OptionRecord`

The base every option record derives from. `to_dict()` and `from_dict()`
come from the dataclass's own fields, so an option added later is keyed,
archived and replayed without a second edit, and a stored record missing an
option is refused rather than read as though the absent option held today's
default.

Each subclass names a `KIND` and registers itself under it, so a record
nested inside another says on the wire what it is and reads back as the type
it was written as. That is what lets a search store its own options without
the planner knowing the type.

### `toolkit`

Re-exported as `shadowspill.planner.toolkit`, and its names are also
available flat from `shadowspill.planner`.

What a search may call into, and need not write itself. Everything here is
search-agnostic: it takes a program, a machine, or a plan, and says
something true about it whichever search asked. A search uses what it wants
and ignores the rest.

| name | signature | what it does |
|---|---|---|
| `validate_search_inputs` | `(program, initial_residency, final_residency, config, admission) -> None` | Refuses malformed inputs, and a capacity that does not reconcile with the pool's declared object capacity. Raises `TypeError` or `ValueError`. |
| `resolutions` | `(program, resolution_options) -> tuple[Resolution, ...]` | Expands a program into resolved programs, one per share of the flexible groups recomputing. `resolution_options` is required. |
| `validate_resolution_options` | `(values) -> tuple[Fraction, ...]` | Exact fractions in `[0, 1]`, sorted and deduplicated. Floats are refused: `0.1` is not one tenth. |
| `CostedAlternatives` | `.from_program(program)` | What each alternative group offers and what each option costs, in bytes retained and runtime. Says whether a group is a real decision at all. |
| `Resolution` | `tuple[TaskAlternativeChoice, ...]` | One option chosen per group -- what makes a program concrete. |
| `ShareValue` | `Fraction \| int \| str` | How a caller may spell one share: `Fraction(3, 8)`, `0`, `1`, or `"3/8"`. |
| `DEFAULT_RESOLUTION_OPTIONS` | `tuple[Fraction, ...]` | Every quarter from none recomputing to all. |

Two more toolkits sit outside this package because they are phases rather
than helpers: `shadowspill.simulator` prices a schedule, and
`shadowspill.planner.admission` proves one fits real memory. A search calls
those the same way.

`answer_no_worse_than(result, *, incumbent, config, placement)` is the
planner's own, not a search's: it replays an incumbent on this machine after
a search returns and answers with it when it is strictly faster and still
places. It is exported because it states a guarantee worth reading.

### Resolution options

Which resolved programs exist is the caller's to say, through
`PressureFitOptions.resolution_options`: the shares of flexible alternative
groups to recompute, as exact fractions -- `Fraction` values, integers or
strings such as `"3/8"` -- sorted and deduplicated on the way in, one
resolved program per share. The default is every quarter, from none
recomputing to all. It lives on PressureFit's options rather than the
planner's because expanding a program into resolved programs is a search's
own work. The planning store keys every plan by the search options it was
searched over, so a plan found under one set is never read back for another.
Inventories small enough to enumerate are planned exhaustively whatever
options are named.

### `plan_program()`

Selects and physically admits a saved `ShadowSpillPlanningProblem` under requested
budgets and `TransferBandwidths`, without capture, compilation or profiling.
It is model- and runtime-independent, so it may be repeated across a
budget/bandwidth frontier from one built program.

<!-- source-signature: src/shadowspill/planner/plan.py:plan_program -->
```text
plan_program(
    problem,
    *,
    execution_budget=None,
    spill_budget=None,
    transfer_bandwidths=None,
    search_options=None,
    incumbent=None,
    artifact_store=None,
    plan_store=None,
    verbose=True,
    plan_store_mode='contribute',
    implementation_revision=None,
) -> AnnotatedProgramPlan
```

| argument | type | default | meaning |
|---|---|---|---|
| `problem` | `ShadowSpillPlanningProblem` | required | The question to answer, normally `build_step_program(...).recurrent` or a value read back with `ShadowSpillPlanningProblem.from_value()`. |
| `execution_budget` | `int` \| `None` | `None` | Device bytes to plan for; the problem's own budget when `None`, and never more than the capacity it was compiled and profiled under. |
| `spill_budget` | `int` \| `None` | `None` | Spill bytes to plan for, with the same bound. |
| `transfer_bandwidths` | `TransferBandwidths` \| `None` | `None` | Fetch and evict rates to price copies at; the problem's embedded calibration when `None`. |
| `search_options` | `SearchOptions` \| `None` | `None` | How to answer the question: `generic` for what any search understands, `algorithm` for the search itself carrying its own options, and `workers` for how much of the machine it may use (zero for every logical CPU, one to force serial evaluation). `None` runs the search that ships with its defaults. A problem carries none of this, because a problem is a question and how to answer it belongs to whoever plans it. `generic` and `algorithm` are part of the plan key; `workers` is not, since it changes how long an answer takes rather than which answer is right, and is recorded on the report instead. |
| `incumbent` | `AnnotatedProgramPlan` \| `None` | `None` | The plan to beat, for the same program, found under another budget. |
| `artifact_store` | path \| `None` | `None` | Roots the store. |
| `plan_store` | path \| `None` | `None` | Roots the planning tree of its own, so one store serves many runs that each own their plans. |
| `plan_store_mode` | `"contribute"` \| `"reuse"` \| `"require"` \| `"refresh"` | `"contribute"` | What this call does with the planning tree. |
| `verbose` | `bool` | `True` | Reports search progress as it runs. |
| `implementation_revision` | `str` \| `None` | `None` | Marks the operation implementations the plan was measured against. |

The store arguments mean exactly what [the frontend
page](frontend.md#store-arguments) says; there are no build-store arguments
here, because planning writes nothing under `build/`.

`incumbent` is the plan to beat. It reaches the search as a bound, and the
answer is held to it here: the incumbent's schedule is replayed on the
requested machine after the search returns, and answered with when it is
strictly faster and its layout still fits. A plan that fits in less memory
fits in more, which is what lets a budget sweep hand each budget the best
plan found below it and never plan worse with more memory;
`plan_step_search()` does exactly that. The guarantee is the planner's, so it
holds whichever search runs. The plan report's diagnostics say what became of
it under each resolved program, and a plan that won is reported as candidate
`incumbent`. The planning store treats it as provenance rather than identity:
a request reads back the plan its search chose, whatever that search was
handed, so a run that replans the budget it is about to execute gets the
sweep's answer.

### `pressurefit`

The search that ships, answering one `ShadowSpillProgram` directly. Capacity
is settled inside it: a candidate measures its own plan against the pool
`placement` describes and gives capacity back until the plan fits, so there
is nothing to retry at this level. It takes no store: nothing it does is
written down.

<!-- source-signature: src/shadowspill/planner/search/algorithms/pressurefit/__init__.py:PressureFit.__call__ -->
```text
pressurefit(
    program,
    *,
    initial_residency,
    final_residency=(),
    config,
    generic,
    workers=0,
    admission=None,
    placement=None,
    progress=None,
    incumbent=None,
    best=None,
) -> ProgramPlanResult
```

| argument | type | default | meaning |
|---|---|---|---|
| `program` | `ShadowSpillProgram` | required | The canonical program, alternatives still open. |
| `initial_residency` | `tuple[ResidencySpec, ...]` | required | Where each object lives before the first task. |
| `final_residency` | `tuple[ResidencySpec, ...]` | `()` | Where each object must live after the last task. |
| `config` | `SimulationConfig` | required | The machine: devices, capacities, bandwidths and latencies every candidate is simulated against. |
| `generic` | `GenericPlanningOptions` | required | What any search understands. This is the generic half of a `SearchOptions`; the algorithm's own options are held by the instance being called, not passed here. |
| `workers` | `int` | `0` | How much of the machine to search on: zero for every logical CPU, one to evaluate on the calling thread. |
| `admission` | `AdmissionFacts` \| `None` | `None` | Switches on the dynamic-pool replay, which rejects schedules that certified fixed placement accepts. |
| `placement` | `AdmissionFacts` \| `None` | `None` | Lets each candidate measure whether its plan has a layout that fits the pool. |
| `progress` | `(str) -> None` \| `None` | `None` | Receives one line per search milestone. |
| `incumbent` | `ProgramPlanResult` \| `None` | `None` | The plan to beat, as a result for this same program. It reaches the resolved program it was found for; a search over resolution options that exclude that one carries none. |
| `best` | `BestPlaced` \| `None` | `None` | A placement record shared with a wider search, so this call is bounded by what that search already placed. `None` means this call starts from nothing and keeps its own. |

The result records the effective object capacity and the `AdmissionFacts` it
was planned under. Raises `PlanInfeasibleError` when every resolution is
analytically infeasible at this capacity, and `ValueError` when the incumbent
is a plan for a different program.

### `validate_schedule_feasibility()`

Checks whether at least one legal selection of the program, over the same
resolved programs PressureFit would plan, satisfies the required task-by-task
residency floor. It returns `None` and raises `PlanInfeasibleError` when
nothing does, so it rejects an irreducible capacity failure before a search is
paid for. What passes here is what the search can reach.

```text
validate_schedule_feasibility(
    program,
    *,
    initial_residency,
    final_residency=(),
    config,
    admission=None,
    resolution_options=None,
) -> None
```

Use `simulate()` instead to validate an explicit schedule that already exists.

### Search policy

`SearchOptions` is the search policy. Every field is part of a planned
program's identity, worker count included.

| field | type | default | meaning |
|---|---|---|---|
| `initial_placement` | `InitialPlacement` | `GREEDY` | How host-origin objects may be placed before the first task: `REQUIRED` places only what a task demands, `GREEDY` places what fits. |
| `residency_strategies` | `tuple[str, ...]` | `("headroom-stall", "tight-stall")` | Which residency policies the candidate set is built from. |
| `fetch_rules` | `tuple[str, ...]` | `("packed-fifo", "packed-fit", "latest-safe", "demand")` | Which fetch orderings the candidate set is built from. |
| `evaluate_coalesced` | `bool` | `True` | Also evaluate the coalesced form of each candidate. |
| `max_repair_attempts` | `int` | `256` | How many monotonic repairs one candidate may make before answering with the best plan it reached. |
| `capacity_refinement_bytes` | `int` | `256 MiB` | How much capacity a plan gives back at a time when its layout does not fit. |
| `record_reduction_steps` | `bool` | `False` | Record each candidate's reduction trajectory, one `ReductionStep` per plan it held. |
| `workers` | `int` | `0` | Library threads the search runs on. Zero takes one per logical CPU; one evaluates every pair on the calling thread. |
| `deterministic` | `bool` | `False` | Make every candidate's outcome a pure function of its inputs, so any worker count answers the same. |
| `minimum_object_bytes_evict_eligible` | `int` | `1 << 20` | Objects smaller than this stay resident from first to last access. |
| `split_write_backs` | `bool` | `False` | Let a plan split an eviction whose copy fits in idle evict-lane time, keeping the split only if the replan is faster. |

Four of these repay a longer word.

**`workers`.** Python owns no threads: one call gets its own workers, so two
callers planning at the same time do not contend. The unit of work is one
(resolved program, candidate) pair, which is why worker count and
resolved-program count are independent -- eight workers means eight threads
whether the call was given one resolved program or five. It is a scheduling
choice, not a search input: it does not change which plans are legal or how
they simulate, though it does change how many candidates get skipped against
the shared placement record, so per-candidate counters move with it.

**`deterministic`.** Candidates measure a layout only when the shared
best-placed record says it could win, so which worker places first decides
which candidates are ever measured, and two searches over one problem at
different worker counts can answer with different plans. Setting this makes the
placement gate consult only the candidate's own placed plans instead of the
shared record. It costs wall time, because that shared bound is what lets a
candidate skip measuring a plan which cannot win.

**`capacity_refinement_bytes`.** The extent does not shrink byte for byte with
the capacity, so handing back the whole shortfall overshoots the capacity that
would have fit. Zero does hand back the whole shortfall, which converges in the
fewest rounds and is the setting to reach for when planning time matters more
than the last percent of makespan.

**`minimum_object_bytes_evict_eligible`.** The reducer never cuts an object
under this size, so it is never evicted and fetched mid-step, while its
boundary contract -- an opening fetch, a release after the last access, a
terminal writeback when modified -- is emitted as for any object. The default
is the size below which a copy is latency-bound and its bytes hardly relieve a
boundary, while every such object is still a cut candidate, a dispatch, and an
event. Zero makes every object eligible, which is what a caller planning
byte-sized objects wants. Every lease of an object it holds gets a static home
in the `ResidentSlice` the result carries -- their sum, one home per lease,
sized at problem preparation -- and the capacity the reducer plans against is
reduced by that slice, so they are never charged again. A slice a budget cannot
hold is reported as infeasible rather than quietly relaxed, and each graph-pair
problem reports how many objects it held, their bytes, and their peak resident
bytes.

### Step data ordering

`StepDataOrdering` records how a training step walks its microbatches: `depth`
passes of `breadth` microbatches, each pass stage-major forward and stage-major
back, with `pair_loss` running each microbatch's last stage forward and
backward together and `reverse_breadth` walking a pass's microbatches in
reverse during backward. `StepDataOrdering.resolve()` fills in whichever count a
caller left out and refuses a product that is not the microbatch count;
`creates()` says which microbatch's backward creates a stage's gradient (the
first the walk reaches) and which add into it. The record travels with a
`StepProgram` and a plan report, and its `label` (`2x4rp`) names the ordering in
figures.

### Results and diagnostics

Configuration and results:

- `SearchOptions`, `InitialPlacement`
- `ProgramPlanResult`, `PlanningDiagnostics`, `ResidentSlice`
- `AdmissionFacts`, `StorageHandoff`, `TaskAdmissionSpec`
- `TaskAllocationStep`, `TaskAllocationStepKind`

Search diagnostics:

- `TaskAlternativeChoiceDiagnostic`, `ResolvedProgramDiagnostics`
- `CandidateDiagnostic`
- `PlanningRepairDiagnostics`, `PlanningWorkDiagnostics`
- `PlanningSectionTiming`, `ReductionStep`

`PlanningWorkDiagnostics` counts what the search did and carries a
`PlanningSectionTiming` saying where the time went. Sections are disjoint spans
named by the function that opened them, so `total_ns` equals `named_ns` plus
`residual_ns` at every level of the hierarchy -- candidate, resolved program,
and whole call. `admit_ns` is the one exception: admission runs as part of
simulating, so it is nested inside `simulate_ns` rather than beside it. Summing
two of these adds every section, which is how the aggregate is built.

Sections measure work rather than elapsed time, so with several workers a
resolved program's total exceeds the time the call took. `started_ns` and
`finished_ns`, on both `CandidateDiagnostic` and `ResolvedProgramDiagnostics`,
are the elapsed-time counterpart: nanoseconds from the start of the call,
shared by every span in it, so two candidates ran at the same time exactly when
their spans overlap.

`ReductionStep` is one plan a candidate held: its makespan, the bytes its
layout needed, the capacity it was built against, the objects the reducer cut
to reach it, and what became of it -- simulated, measured, placed, refined,
best so far, or the answer. `CandidateDiagnostic.steps` is the whole trajectory
in order, and is empty unless `record_reduction_steps` asked for it.

`PlanningRepairDiagnostics` records what one candidate's monotonic repairs did
and why they stopped.

### Where the work happens

Every step of physical admission runs in the library: `pressurefit()` selects
candidates there, the operations a schedule implies are derived there, and each
lease is placed at a fixed offset there. Missing or ABI-incompatible libraries
fail immediately rather than falling back, and the readable Python equivalents
live outside the package in `reference/python/`, where production never imports
them.

See the [PressureFit formulation and
algorithm](../../architecture/pressurefit.md) and the separate [graph-pair
selector](../../architecture/graph-pair-selection.md). The task-allocation
topology and exact range certificate are described in [physical
admission](../../architecture/physical-admission.md), and how a schedule
becomes leases in [from a resolved program to
leases](../../architecture/admission-leases.md).

## `shadowspill.simulator`

`simulate()` replays one explicit schedule through the required compiled
simulator and returns what it cost. Missing or ABI-incompatible libraries fail
immediately.

```text
simulate(
    program,
    schedule,
    *,
    selections=(),
    config,
    admission=None,
) -> SimulationResult
```

| argument | type | default | meaning |
|---|---|---|---|
| `program` | `ShadowSpillProgram` | required | The program the schedule belongs to. |
| `schedule` | `MemorySchedule` | required | The explicit actions to replay. |
| `selections` | `tuple[TaskAlternativeChoice, ...]` | `()` | Which option each task-alternative group takes. |
| `config` | `SimulationConfig` | required | Devices, capacities, bandwidths and latencies to price against. |
| `admission` | `SimulationAdmission` \| `None` | `None` | Physical admission to replay alongside, when the caller has one. |

Configuration and results:

- `SimulationConfig`, `DeviceSimulationConfig`, `SimulationAdmission`
- `SimulationResult`, `SimulationInfeasibleError`
- `TaskInterval`, `TransferInterval`, `TransferDirection`
- `MemorySnapshot`, `DeviceMemoryPeak`, `CapacityViolation`
- `ActionPhysicalDelta`, `TaskPhysicalDelta`, `MemoryReuseDependency`

`TransferInterval.kind` names the `MemoryActionKind` behind a copy beside its
`direction`: a write-back shares the evict lane with evictions and is told
apart by it.

A fetch or task launch with nowhere to go waits for room rather than failing,
as [simulation](../../architecture/simulation.md#trigger-time-capacity)
specifies, so a plan that comes up short is slower rather than rejected. Each
shortfall is reported as a `CapacityViolation` alongside the `device-capacity`
stall that records the wait: the stall says when and for how long, the
violation says by how much.

## `shadowspill.runtime`

Standalone physical-admission helpers, for tooling that wants to reproduce or
inspect what admission does without a runtime:

| Function | Returns | Purpose |
|---|---|---|
| `workspace_reserve_bytes(maximum_task_workspace_bytes, *, policy=None)` | `int` | The conservative contiguous-workspace allowance a budget must hold back. |
| `plan_slab_layout(slab_bytes, events, *, dynamic_allocation_ids=frozenset())` | `SlabLayout` | One deterministic address per complete allocation lifetime, which online best-fit cannot always find. |
| `replay_slab_timeline(slab_bytes, events)` | `SlabReplay` | Replays the production two-ended policy over an allocation timeline and rejects spatial infeasibility. |
| `admit_physical_budget(*, device_budget_bytes, spill_budget_bytes, baseline_bytes, observed_external_bytes, maximum_task_workspace_bytes, predicted_spill_peak_bytes, allocation_timeline=(), policy=None)` | `(PhysicalAdmission, SlabReplay)` | Computes the explicit reserves and spatially validates the slab, raising `AdmissionError` when the headroom leaves none. |
| `run_admission_replay(capacity_bytes, operations, *, lease_count, dependency_count, minimum_alignment=256, large_request_threshold_bytes=0)` | `AdmissionReplayResult` | Replays an ordered script through the exact production memory-pool policy. |

`AdmissionPolicy` is the tunable margin policy those take; `AllocationEvent`
and `AllocationOperation` are the timeline they read; `SlabPlacement`,
`SlabLayout` and `SlabReplay` are what the layout helpers answer with.

The production-memory-pool replay interface is `AdmissionReplayOperation` and
`AdmissionReplayOperationKind` (the script), `AdmissionReplayLeaseState`,
`AdmissionReplayDecision` and `AdmissionReuseDependency` (what each step did),
and `AdmissionReplayResult` (the whole replay).

`ObjectRef` is the framework-neutral retained handle for one runtime-global
logical object. Object identity is independent of its current pool lease or
residency generation. Framework integrations layer their own view metadata on
this handle and call `ObjectRef.close()` to release public ownership.
`ObjectConsistency` selects causal generation/readiness ordering or an
explicitly unordered cross-plan view for a plan binding.
