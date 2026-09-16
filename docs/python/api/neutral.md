# Framework-neutral Python API

These packages expose the errors, the IR, the step's shape, the artifact store,
the planner, the simulator, and physical admission. Nothing here imports a
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

## `shadowspill.step`

A training step's shape and provenance, with no framework present. Two values,
both readable and serializable with no PyTorch installed, which is what lets a
corpus collected on one machine be planned on another.

### `StepDataOrdering`

How a training step walks its microbatches.

| field | type | default | meaning |
|---|---|---|---|
| `depth` | `int` | required | Passes over the microbatches. |
| `breadth` | `int` | required | Microbatches per pass, each pass stage-major forward and stage-major back. |
| `reverse_breadth` | `bool` | `True` | Walk a pass's microbatches in reverse during backward. |
| `pair_loss` | `bool` | `True` | Run each microbatch's last stage forward and backward together. |

```text
StepDataOrdering.resolve(
    *,
    microbatches,
    depth=None,
    breadth=None,
    reverse_breadth=True,
    pair_loss=True,
) -> StepDataOrdering

StepDataOrdering.creates(position, stage_index, *, stage_count) -> bool
```

`resolve()` fills in whichever of `depth` and `breadth` a caller left out, and
refuses a product that is not `microbatches`. `creates()` answers whether the
microbatch at `position` is the one whose backward *creates* a stage's gradient
-- the first the walk reaches -- rather than adding into it. The `microbatches`
property is the product; `depth_first(microbatches, *, reverse_breadth=True,
pair_loss=True)` builds the one-at-a-time walk; `positions(pass_index)` and
`backward_positions(pass_index)` give one pass's forward and backward order.
`label` names the ordering in figures (`2x4rp`: the two counts, then `r` and `p`
for the flags that are set) and `from_label()` reads one back. `to_dict()` and
`from_dict(value, path)` carry it beside a `StepProgram` and a plan report; the
`path` names the position being read, so a malformed document is refused where
it went wrong.

### `StepProgram`

The recurrent and optional initial `ShadowSpillPlanningProblem` a captured step
lowered to, with the provenance that says what produced them: the ordering, the
measured profiles, and the digests that identify the content rather than the
run. `build_step_programs()` returns one per ordering; `to_json()` and `from_json()` round it
through a file, and `digest` identifies what it would plan as. Its fields are
documented in [reusable artifacts](artifacts.md#stepprogram).

## `shadowspill.task`

One task's shape and provenance, as a frontend captured, compiled and measured it.
`shadowspill.step` is a step's; this is a task's. Every record here is neutral, and
everything downstream -- planning, simulation, the runtime, the diagnostics --
reads them rather than anything a framework produced.

### `shadowspill.task.storage`

What a task promised about its storages. `TaskStorageContract` is the promise;
`StorageRoot` is one storage the task reads or writes, `OutputView` is a view on
one of its outputs, and `MutationBinding` is a mutation it declares. A contract
that does not match what the task did is a refusal, not a warning.

### `shadowspill.task.allocations`

What a task promised about its allocations, and what the profiled run observed.
`TaskAllocationContract` is the promise, `TaskAllocationEvent` and
`TaskAllocationOperation` are the timeline it implies, and
`TaskAllocationPathObservation` is what the allocator actually saw.

### `shadowspill.task.profiles`

One measurement: what the task cost, filed under the key that identifies the code
and the values it was measured with.

### `shadowspill.task.inputs`

`TaskInputRole` says what a task argument is -- an activation, a parameter, a
buffer, a constant, a piece of optimizer state. `RepresentativeInputSummary` is
the provenance of the value profiling materialized for one argument: its geometry,
never its contents, so a profile artifact says what was measured and carries no
data. `REPRESENTATIVE_VALUE_POLICY` is the policy it was generated under, and a
profile recorded under an older one is not comparable.

## `shadowspill.store`

Content-addressed storage for what each stage produced, and the modes that gate
it. A store is not part of planning: it holds captured graphs, compiled
profiles, programs and plans keyed by the digest of their inputs, so a later run
asking the same question reads the answer instead of recomputing it. The
frontend's build store, the planner's plan store, and the qualification
harnesses all sit on this one.

`ArtifactStore` is the store: two trees under one root.

```text
ArtifactStore.resolve(
    value,
    *,
    build_store=None,
    plan_store=None,
    build_store_mode='contribute',
    plan_store_mode='contribute',
    export_bypass_key=None,
) -> ArtifactStore
```

`value` is the root both trees live under, or `None` for the default cache;
`build_store` and `plan_store` root one tree elsewhere; the two modes say what
this run may do with each tree; `export_bypass_key` is the caller's name for
the code the artifacts were produced from, under which they are filed. Every entry point's store
arguments reach this one call, and [the frontend
page](frontend.md#store-arguments) defines them. `initialize()` creates the
tree, `artifacts()` returns the `PlanningArtifact` records this call touched,
and `record(...)` adds one.

`PlanningArtifact` is one such record: its `category` and `kind`, the `digest`
it is keyed by (SHA-256, or `None` for an unkeyed entry), the `path` it landed
at, how the run `access`ed it, its `schema`, and the `dependencies` digests it
was derived from.

`StoreMode` is the literal `"contribute" | "reuse" | "require" | "refresh"`, and
`STORE_MODES` is those four in order, so a CLI or a config validates against one
tuple. `StorePolicy` turns one mode into the four gates the code checks --
`read_enabled`, `write_enabled`, `overwrite` and `require_hit` -- which is what
stops a caller spelling out a combination that means nothing;
`StorePolicy.for_mode(mode)` builds one, `refuse_miss(what, key, request=None)`
raises the refusal that names the fix -- and, when the store passes `request`,
what the key was made of, so a miss can be read against the records the store
holds -- and `CONTRIBUTE` is the default policy. `digest_directory(root, digest)`
is where one digest's entry lives under a tree. `atomic_json(path, value)` and
`atomic_text(path, value)` are the writers every record goes through, written
beside their destination and renamed into place, so a reader never sees a
partial record. `ArtifactRecorder` is the callback a store takes to publish
each artifact it touched as a `PlanningArtifact`; the build store, the profile
and manifest stores, the plan store and the step archive all take this one
protocol.

[The artifact store](../artifact-store.md) has the layout and what each digest
covers.

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
search to any plan it was handed, and physically admits the winner. When the
planning store already holds the answer, the plan is read back with the
simulation and the fixed-layout certificate recorded beside it and nothing is
simulated or placed; when it holds a refusal, that refusal is raised again.
`summarize_plan()` asks the store the same question and reads only the summary
it keeps beside a certified plan, for a caller that compares many plans and
runs one.

Building a program is the frontend's job:
[`build_step_programs()`](frontend.md#build_step_programs) captures, compiles,
profiles and lowers them, and takes only build-store arguments. `plan_program()`
plans one and takes only plan-store arguments. Nothing on this page needs a
device or a model.

### `plan_program()`

Selects and physically admits a saved `ShadowSpillPlanningProblem` under
requested budgets and `TransferBandwidths`, without capture, compilation or
profiling. It is model- and runtime-independent, so it may be repeated across a
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
    export_bypass_key=None,
) -> AnnotatedProgramPlan
```

| argument | type | default | meaning |
|---|---|---|---|
| `problem` | `ShadowSpillPlanningProblem` | required | The question to answer, normally `build_step_programs(...)[0].recurrent` or a value read back with `ShadowSpillPlanningProblem.from_value()`. |
| `execution_budget` | `int` \| `None` | `None` | Device bytes to plan for; the problem's own budget when `None`, and never more than the capacity it was compiled and profiled under. |
| `spill_budget` | `int` \| `None` | `None` | Spill bytes to plan for, with the same bound. |
| `transfer_bandwidths` | `TransferBandwidths` \| `None` | `None` | Fetch and evict rates to price copies at; the problem's embedded calibration when `None`. |
| `search_options` | `SearchOptions` \| `None` | `None` | How to answer the question. `None` runs the search that ships with its defaults. A problem carries none of this, because a problem is a question and how to answer it belongs to whoever plans it. |
| `incumbent` | `AnnotatedProgramPlan` \| `None` | `None` | The plan to beat, for the same program, found under another budget. |
| `artifact_store` | path \| `None` | `None` | Roots the store. |
| `plan_store` | path \| `None` | `None` | Roots the planning tree of its own, so one store serves many runs that each own their plans. |
| `verbose` | `bool` | `True` | Reports search progress as it runs. |
| `plan_store_mode` | `"contribute"` \| `"reuse"` \| `"require"` \| `"refresh"` | `"contribute"` | What this call does with the planning tree. |
| `export_bypass_key` | `str` \| `None` | `None` | The caller's name for the code the plan was measured against; artifacts are filed under it. |

Raises `TypeError` when `search_options` is neither a `SearchOptions` nor
`None`. The store arguments mean exactly what [the frontend
page](frontend.md#store-arguments) says; there are no build-store arguments
here, because planning writes nothing under `build/`.

`incumbent` is the plan to beat. It reaches the search as a bound, and the
answer is held to it here: the incumbent's schedule is replayed on the
requested machine after the search returns, and answered with when it is
strictly faster and its layout still fits. A plan that fits in less memory fits
in more, which is what lets a budget sweep hand each budget the best plan found
below it and never plan worse with more memory; `plan_step_search()` does
exactly that, and the guarantee is the planner's, so it holds whichever search
runs. The plan report's diagnostics say what became of it under each resolved
program, and a plan that won is reported as candidate `incumbent`. The planning
store treats it as provenance rather than identity: a request reads back the
plan its search chose, whatever that search was handed, so a run that replans
the budget it is about to execute gets the sweep's answer.

### `summarize_plan()`

What the planning store already holds for one problem, without its plan. The
question is the one `plan_program()` would ask -- the same key -- and the
answer is the summary the store writes beside a plan when it certifies it.
Nothing but that summary is read, so a caller that compares many plans and
runs one reads kilobytes per question and fetches a whole plan, through
`plan_program()`, only for the one it will run; `plan_step_search()` is that
caller.

<!-- source-signature: src/shadowspill/planner/plan.py:summarize_plan -->
```text
summarize_plan(
    problem,
    *,
    execution_budget=None,
    spill_budget=None,
    transfer_bandwidths=None,
    search_options=None,
    artifact_store=None,
    plan_store=None,
    plan_store_mode='contribute',
) -> PlanSummaryLookup | None
```

The arguments mean what they mean for `plan_program()`. It returns a
`PlanSummaryLookup` (from `shadowspill.planner.plan_store`): the `key` the
plan is filed under, its `makespan_ns` as the certificate re-simulated it, its
`summary` as a `PlanSummary`, the `graph_pair_outcomes` of every graph-pair
selection the search evaluated, and `answered_with_incumbent`, whether the
search answered with the plan it was handed to beat. It returns `None` when the
store has no answer, or one nobody has certified yet; the caller then plans,
and `plan_program()` applies the store's mode to the miss. A refusal the store
recorded is raised as `plan_program()` would raise it. A store written before
summaries were kept answers from the plan once and keeps the summary it built,
when the mode allows writing.

A plan to beat is not taken here. A plan in hand that claims to be faster than
the stored answer is a question only a search settles, and `plan_program()`
is where it is asked.

### `validate_schedule_feasibility()`

Asks the search that will run whether any schedule could fit this machine,
before a search is paid for. It returns `None` and raises
`PlanInfeasibleError` when nothing can, so an irreducible capacity failure is
rejected early. What passes here is what that search can reach; it is not a
promise that a plan exists.

```text
validate_schedule_feasibility(
    program,
    *,
    initial_residency,
    final_residency=(),
    config,
    admission=None,
    search_options=None,
) -> None
```

| argument | type | default | meaning |
|---|---|---|---|
| `program` | `ShadowSpillProgram` | required | The canonical program, alternatives still open. |
| `initial_residency` | `tuple[ResidencySpec, ...]` | required | Where each object lives before the first task. |
| `final_residency` | `tuple[ResidencySpec, ...]` | `()` | Where each object must live after the last task. |
| `config` | `SimulationConfig` | required | The machine to check against. |
| `admission` | `AdmissionFacts` \| `None` | `None` | Pool topology to check alongside, when the caller has one. |
| `search_options` | `SearchOptions` \| `None` | `None` | Which search to ask, and what it is told. `None` asks the one that ships with its defaults. |

It is the `preflight()` of `search_options.resolved_algorithm`, nothing more.
Use `simulate()` instead to validate an explicit schedule that already
exists.

### Physical admission

`shadowspill.planner.admission.physical` decides what a plan needs physically,
before any runtime exists: the explicit reserves a budget must hold back, and
whether the slab can actually hold the plan's allocations at some address. The
runtime does not call it -- planning does, and then a runtime is opened against
the budget it approved. Standalone, so tooling can reproduce or inspect the
decision without a runtime:

| Function | Returns | Purpose |
|---|---|---|
| `workspace_reserve_bytes(maximum_task_workspace_bytes, *, policy=None)` | `int` | The conservative contiguous-workspace allowance a budget must hold back. |
| `plan_slab_layout(slab_bytes, events, *, dynamic_allocation_ids=frozenset())` | `SlabLayout` | One deterministic address per complete allocation lifetime, which online best-fit cannot always find. |
| `replay_slab_timeline(slab_bytes, events)` | `SlabReplay` | Replays the production two-ended policy over an allocation timeline and rejects spatial infeasibility. |
| `admit_physical_budget(*, device_budget_bytes, spill_budget_bytes, baseline_bytes, observed_external_bytes, maximum_task_workspace_bytes, predicted_spill_peak_bytes, allocation_timeline=(), policy=None)` | `(PhysicalAdmission, SlabReplay)` | Computes the explicit reserves and spatially validates the slab, raising `PhysicalAdmissionError`, a `ValueError` carrying `kind`, `required_bytes` and `capacity_bytes`, when the headroom leaves none. |
| `run_admission_replay(capacity_bytes, operations, *, lease_count, dependency_count, minimum_alignment=256, large_request_threshold_bytes=0)` | `AdmissionReplayResult` | Replays an ordered script through the exact production memory-pool policy. |

`AdmissionPolicy` is the tunable margin policy those take; `AllocationEvent`
and `AllocationOperation` are the timeline they read; `SlabPlacement`,
`SlabLayout` and `SlabReplay` are what the layout helpers answer with.

### Results and diagnostics

Configuration and results:

- `SearchOptions`, `GenericPlanningOptions`, `InitialPlacement`, `OptionRecord`
- `ProgramPlanResult`, `PlanningDiagnostics`, `ResidentSlice`
- `AdmissionFacts`, `StorageHandoff`, `TaskAdmissionSpec`
- `TaskAllocationStep`, `TaskAllocationStepKind`

Search diagnostics:

- `TaskAlternativeChoiceDiagnostic`, `ResolvedProgramDiagnostics`
- `CandidateDiagnostic`
- `PlanningRepairDiagnostics`, `PlanningWorkDiagnostics`
- `PlanningSectionTiming`, `ReductionStep`
- `GraphPairOutcome`

`ProgramPlanResult` is what a search answers with: the `program` it planned,
the `search_options` it was told, the `initial_residency` and `final_residency`
it honoured, the `simulation_config` it was priced against, the chosen
`schedule` and `selections`, the `simulation` that priced them, its
`diagnostics`, the `resident_slice` of objects held resident throughout, and
the `admission_facts` and `placement_facts` it was planned under.

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

`GraphPairOutcome` is what one graph-pair selection cost, read off a finished
result: its `selection_id`, how many groups it asked to recompute
(`recompute_groups` of `group_count`), the `makespan_seconds` of its best plan
or `None` when no candidate fit, the compute it asked for and the cheapest any
selection could (`selected_compute_seconds`, `unconstrained_seconds`, their
difference as `recomputation_overhead_seconds`, the rest of the makespan as
`waiting_seconds`), how many candidates were `valid` of those evaluated, and
the bytes its best plan fetched and evicted.
`shadowspill.planner.diagnostics.graph_pair_outcomes(result)` derives one per
resolved program of a `ProgramPlanResult`, ordered by how many groups
recompute, from the program's task profiles and the search's own record --
nothing is simulated. `AlternativeCosts`, beside it, is the cost table both it
and `PlanSummary` read: every option's compute by group, each group's
cheapest, and the compute no option can avoid.

### `shadowspill.planner.search`

The search is a replaceable part: the planner states the protocol, the shipped
search implements it, and a caller's own search is no less a first-class one.
`SearchAlgorithm`, `SearchOptions` and `answer_no_worse_than()` are also
available flat from `shadowspill.planner`.

#### `SearchAlgorithm`

The base class a search is. Subclass it, give it a `name`, implement
`__call__`, and pass an instance -- nothing is looked up by name on the calling
path, so there is no registry to reserve anything in.

| member | kind | meaning |
|---|---|---|
| `name` | `ClassVar[str]` | The stable string the plan key records. Not the class's name, so renaming or moving the class leaves a stored corpus reachable. Defining a concrete subclass without one is refused. |
| `options` | `OptionRecord` | This search's own options, carried into the plan key whole and read by nothing outside the search. |
| `__init__(options=None)` | method | Build the search with its own record; `None` means its defaults. Every search is constructed this way, which is what lets an archived plan be handed the search that made it. |
| `named(name)` | `staticmethod` | The subclass registered under `name`, which defining it registered. Raises `KeyError`, naming the searches it knows, when the defining module has not been imported. |
| `preflight(...)` | method | Refuse a machine no schedule can fit, before a search is paid for. Takes `program`, `initial_residency`, `final_residency=()`, `config`, `admission=None` and `generic`, and returns `None`. Defaults to saying nothing, which is always correct. |
| `__call__(...)` | abstract | Answer with a `ProgramPlanResult`. The only method a subclass must write. |

An instance holds no per-call state, so one serves every budget of a sweep and
every worker of a search. [Writing a search
algorithm](../../architecture/search-algorithm.md) is the reference: every
argument with its type and meaning, the defaults, and a worked example. [Plan
search](../../architecture/search.md) states what the planner promises in
return.

#### `SearchOptions`

The whole of what one planning call is told about searching, in one argument so
neither half can be set without the other being visible.

| field | type | default | meaning |
|---|---|---|---|
| `generic` | `GenericPlanningOptions` | all defaults | What any search understands. |
| `algorithm` | `SearchAlgorithm` \| `None` | `None` | The search itself, holding its own options. `None` runs the one that ships. |
| `workers` | `int` | `0` | Threads the search may use; `0` is every logical CPU, `1` forces serial. |

`generic` and `algorithm` are both part of the plan key. **`workers` is
not** -- it says how much machine to spend, not what to decide, so two runs
at different worker counts ask the same question and read back the same
answer. The plan report records what was used. `resolved_algorithm` is the
search that will actually run: `algorithm`, or the one that ships. `to_dict()`
writes the keyed half and `from_dict(value, path)` reads it back, rebuilding the
search from its name and handing it its own options; `workers` appears in
neither and comes back zero.

A worker count is one thread per `(resolved program, candidate)` pair, so it is
independent of how many resolved programs a call has. It does change how many
candidates get skipped against the shared placement record, so per-candidate
counters move with it. Python owns none of these threads: one call gets its own
workers, and two callers planning at once do not contend.

`GenericPlanningOptions` is what every search is told, whichever one runs.

| field | type | default | meaning |
|---|---|---|---|
| `deterministic` | `bool` | `False` | Make every candidate's outcome a pure function of its inputs, so any worker count answers the same. Candidates otherwise measure a layout only when the shared best-placed record says it could win, so which worker places first decides which candidates are ever measured; setting this has the placement gate consult only the candidate's own placed plans, and costs wall time. |
| `minimum_object_bytes_evict_eligible` | `int` | `1 << 20` | Objects smaller than this stay resident from first to last access instead of being evicted and fetched mid-step, while their boundary contract -- an opening fetch, a release after the last access, a terminal writeback when modified -- is emitted as for any object. The default is the size below which a copy is latency-bound. Zero makes every object eligible. |

Each lease the floor holds gets a static home in the `ResidentSlice` the result
carries, and the capacity the search plans against is reduced by that slice, so
those bytes are never charged again. A slice a budget cannot hold is reported as
infeasible rather than quietly relaxed.

```python
from shadowspill.planner import (
    GenericPlanningOptions,
    SearchOptions,
    plan_program,
)
from shadowspill.planner.search.algorithms.pressurefit import (
    PressureFit,
    PressureFitOptions,
)

plan = plan_program(
    problem,
    search_options=SearchOptions(
        generic=GenericPlanningOptions(deterministic=True),
        algorithm=PressureFit(PressureFitOptions(max_repair_attempts=256)),
        workers=8,
    ),
)
```

#### `OptionRecord`

The base every option record derives from. `to_dict()` and `from_dict()`
come from the dataclass's own fields, so an option added later is keyed,
archived and replayed without a second edit, and a stored record missing an
option is refused rather than read as though the absent option held today's
default. `record_from_value(value)` reads any record back as the type it was
written as.

Each subclass names a `KIND` and registers itself under it, so a record
nested inside another says on the wire what it is. That is what lets a search
store its own options without the planner knowing the type.

#### `toolkit`

What a search may call into, and need not write itself. Everything here is
search-agnostic: it takes a program, a machine, or a plan, and says something
true about it whichever search asked. A search uses what it wants and ignores
the rest. Reached as `shadowspill.planner.toolkit`, and every name below except
`ShareValue` is also available flat from `shadowspill.planner`.

| name | signature | what it does |
|---|---|---|
| `validate_search_inputs` | `(program, initial_residency, final_residency, config, admission) -> None` | Refuses malformed inputs, and a capacity that does not reconcile with the pool's declared object capacity. Raises `TypeError` or `ValueError`. |
| `resolutions` | `(program, resolution_options) -> tuple[Resolution, ...]` | Expands a program into resolved programs, one per share of the flexible groups recomputing. `resolution_options` is required. |
| `validate_resolution_options` | `(values) -> tuple[Fraction, ...]` | Exact fractions in `[0, 1]`, sorted and deduplicated. Floats are refused: `0.1` is not one tenth. |
| `CostedAlternatives` | `.from_program(program)` | What each alternative group offers and what each option costs, in bytes retained and runtime. `groups`, `forced`, `flexible_count`, `binary_endpoints` and `combination_count` say whether a group is a real decision at all. |
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

#### `pressurefit`

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

`PressureFitOptions` is this search's own record, and lives with it under
`shadowspill.planner.search.algorithms.pressurefit` rather than on
`shadowspill.planner`, because the planner never reads it.

| field | type | default | meaning |
|---|---|---|---|
| `initial_placement` | `InitialPlacement` | `REQUIRED` | How host-origin objects may be placed before the first task: `REQUIRED`, the default, places only what a task demands; `GREEDY` places what fits, into the opening restore the simulated makespan does not count. |
| `resolution_options` | `tuple[Fraction, ...]` | `DEFAULT_RESOLUTION_OPTIONS` | Which resolved programs exist: the shares of flexible alternative groups to recompute, as exact fractions, one resolved program per share. |
| `residency_strategies` | `tuple[str, ...]` | `("headroom-stall", "tight-stall")` | Which residency policies the candidate set is built from. |
| `fetch_rules` | `tuple[str, ...]` | `("packed-fifo", "packed-fit", "latest-safe", "demand")` | Which fetch orderings the candidate set is built from. |
| `evaluate_coalesced` | `bool` | `True` | Also evaluate the coalesced form of each candidate. |
| `max_repair_attempts` | `int` | `256` | How many monotonic repairs one candidate may make before answering with the best plan it reached. |
| `capacity_refinement_bytes` | `int` | `256 MiB` | How much capacity a plan gives back at a time when its layout does not fit. Zero hands back the whole shortfall, which converges in the fewest rounds; a smaller step overshoots less, because the layout's extent does not shrink byte for byte with the capacity. |
| `record_reduction_steps` | `bool` | `False` | Record each candidate's reduction trajectory, one `ReductionStep` per plan it held. |

`resolution_options` takes `Fraction` values, integers or strings such as
`"3/8"`, sorted and deduplicated on the way in. It lives here rather than on the
planner because expanding a program into resolved programs is a search's own
work. Every plan is keyed by the options it was searched under, so a plan found
under one set is never read back for another; inventories small enough to
enumerate are planned exhaustively whatever options are named.

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

## `shadowspill.search`

A step planned at every geometry, budget and walk, and the report that answers.
Framework-neutral: enumerating the splits of a sequence total, planning each
point, and reading the answers back need no framework. Building the programs a
point is planned from does, so `shadowspill.pytorch.plan_step_search()` drives
this package and is documented on [the frontend page](frontend.md#plan_step_search).

| Name | Is |
|---|---|
| `search_geometries(total_sequences_per_step, *, sequence_length, min_tokens_per_microbatch=None, max_tokens_per_microbatch=None)` | Every admitted `(sequences_per_microbatch, accumulation)` pair, largest microbatch first, and the pairs the token bounds skipped with the reason for each. |
| `default_orderings(accumulation)` | Every `depth x breadth` factor pair of the accumulation count, depth-first first. |
| `StepSearchReport` | What a search answered: its points, its builds, and the winner at each budget. `load()` reads one back. |
| `StepSearchPoint` | One geometry-ordering-budget point: what it planned, or the refusal it hit. |
| `StepSearchGeometryBuild` | What one geometry cost to build, once, before any point was planned. |

A refusal is recorded rather than raised, so a search answers with the points it
could plan and the reasons for the rest.

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

One runtime in this process, framework-neutral: it bootstraps over a frontend's
process allocator, registers the configured pools and routes, calibrates the
real directed transfers between their addresses, and is what every plan is
placed on. Importing this package does not import a framework.

### `Runtime`

`Runtime` is opened with a `shadowspill.frontend.RuntimeFrontend` -- the one
object through which it selects and synchronizes a device, installs the process
allocator, and finds or detaches the framework objects holding a lease:

<!-- source-signature: src/shadowspill/runtime/core.py:Runtime.__init__ -->
```text
Runtime(
    *,
    frontend: RuntimeFrontend,
    pools: Mapping[str, MemoryPoolConfig],
    routes: Mapping[str, TransferRoute],
    library_path: str | Path | None = None,
    calibrate: bool = True,
    worker_poll_nanoseconds: int = 1_000,
    background_transfer_window_bytes: int = DEFAULT_BACKGROUND_WINDOW_BYTES,
    backend: str | None = None,
)
```

PyTorch callers construct `shadowspill.pytorch.Runtime` instead, which supplies
that frontend and takes the same remaining arguments; [the frontend
page](frontend.md#runtime) documents them, the pools a runtime holds, and what a
closing plan leaves behind. Allocator selection is process-global and
irreversible, so exactly one runtime is opened per process, before any device
tensor exists.

`runtime.frontend` is the frontend it was opened with, and is the only route
from a runtime to a framework.

### What a runtime publishes

| Value | Is |
|---|---|
| `MemoryPool` | one registered pool: its name, kind, capacity, and device ordinal if it has one |
| `RuntimeRoute` | one directed pool pair bytes may move along |
| `TransferCapabilities` | the measured matrix over every registered route |
| `TransferProfile` | one route's measured bandwidth and latency |

Planning reads the published `transfer_capabilities` snapshot rather than
recalibrating.

### What a failed call leaves

| Value | Is |
|---|---|
| `RuntimeConfigurationError` | a runtime or plan asked for incompatible pool resources |
| `RuntimeExecutionError` | a call failed against a live runtime, carrying the latched diagnostics |
| `RuntimeFailureDiagnostics` | what the runtime latched about the first and most recent failure |
| `ExecutionTaskIdentity` | which task a latched failure belongs to |

[The frontend page](frontend.md#exceptions) shows what raises them.

The production-memory-pool replay interface is `AdmissionReplayOperation` and
`AdmissionReplayOperationKind` (the script), `AdmissionReplayLeaseState`,
`AdmissionReplayDecision` and `AdmissionReuseDependency` (what each step did),
and `AdmissionReplayResult` (the whole replay).

`ObjectRef` is the framework-neutral retained handle for one runtime-global
logical object. Object identity is independent of its current pool lease or
residency generation. Framework integrations layer their own view metadata on
this handle and call `ObjectRef.close()` to release public ownership.
`ObjectConsistency` selects causal generation/readiness ordering or an
explicitly unordered cross-plan view for a plan binding; [the frontend
page](frontend.md#plan_forward) shows it in use.
