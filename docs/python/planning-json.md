# Program and annotated-plan JSON

ShadowSpill exposes content-addressed JSON artifacts at distinct planning
boundaries. They are canonical, schema-tagged, validated on load, and designed
for corpus collection, budget sweeps, inspection, and reproducible planning.
Every schema below carries the one artifact version
(`shadowspill.schema.ARTIFACT_VERSION`, see `artifact-store.md`), so a change
to any stored structure moves them all together.

| Python value | Schema | Boundary |
|---|---|---|
| `ShadowSpillProgram` | `shadowspill.program/v1` | Framework-neutral logical tasks, objects, costs, sharing policies, and task alternatives. |
| `ShadowSpillPlanningProblem` | `shadowspill.plan_program/v1` | One program plus residency, machine inputs, and admission topology. It carries no search options: a program is a problem, and how to search it is the caller's. |
| `StepProgram` | `shadowspill.step_program/v1` | Complete PyTorch capture/profile result with recurrent and optional initial PressureFit Programs. |
| `AnnotatedProgramPlan` | `shadowspill.annotated_program_plan/v1` | The winning plan, physical admission, and simulator evidence for one budget/bandwidth point. |

The ordinary reusable workflow is:

```python
from pathlib import Path

from shadowspill.planner.program import AnnotatedProgramPlan, StepProgram

Path("step-program.json").write_text(step_program.to_json(), encoding="utf-8")
loaded_program = StepProgram.from_json(
    Path("step-program.json").read_text(encoding="utf-8")
)

Path("annotated-plan.json").write_text(annotated.to_json(), encoding="utf-8")
loaded_plan = AnnotatedProgramPlan.from_json(
    Path("annotated-plan.json").read_text(encoding="utf-8")
)
```

## Canonical encoding and identity

`to_json()` emits compact UTF-8 JSON with sorted object keys. Digests are
SHA-256 over canonical content. Array order remains semantically meaningful
for tasks, actions, allocation operations, and attempts.

Three identity rules matter:

1. A `ShadowSpillProgram.digest` covers the complete logical program.
2. A `StepProgram.digest` excludes phase time and cache paths, so the same
   planning content has one identity regardless of where or how long it took
   to construct.
3. An `AnnotatedProgramPlan.digest` covers what was planned, not how it was
   found: it excludes `selection.from_store`, the search diagnostics at both
   levels (`selection.diagnostics` and each attempt's
   `search_diagnostics`), and the whole `timing` block, while retaining
   the source program, budgets, transfer bandwidths, selected schedule and
   residency, the simulation, and the physical certificate.

Never edit a digest independently of its value. `from_json()` recomputes and
validates embedded identities and rejects inconsistent content.

## Program format

An abridged `ShadowSpillProgram` has this shape:

```json
{
  "schema": "shadowspill.program/v1",
  "devices": [
    {"device_id": "device_0", "process_id": "process_0", "kind": "accelerator", "index": 0}
  ],
  "alias_groups": [
    {"alias_group_id": "alias_0", "device_id": "device_0", "size_bytes": 4096,
     "initial_version": 0, "retain_spill_copy": false,
     "shared_residency": null}
  ],
  "objects": [
    {"object_id": "activation_0", "alias_group_id": "alias_0", "offset_bytes": 0,
     "size_bytes": 4096, "role": "activation", "persistence": "step"}
  ],
  "profiles": [
    {"profile_id": "profile_0", "runtime_ns": 120000, "workspace_bytes": 2097152,
     "compatibility_digest": "..."}
  ],
  "tasks": [
    {"task_id": "task_0", "resource": {"device_id": "device_0", "kind": "compute", "lane": 0},
     "profile_id": "profile_0", "dependencies": [], "inputs": [],
     "outputs": ["activation_0"], "mutations": [], "phase": "forward",
     "requires_entrypoint": true}
  ],
  "task_alternative_groups": []
}
```

### Top-level program keys

| Key | Value |
|---|---|
| `schema` | Exact schema label used for versioned parsing. |
| `devices` | Logical execution devices and their process-local indices. |
| `alias_groups` | Physical logical storage roots; capacity is charged at this level. |
| `objects` | Named tensor views into alias groups. |
| `profiles` | Deduplicated task runtime/workspace measurements. |
| `tasks` | Topologically ordered executable/control tasks. |
| `task_alternative_groups` | Mutually exclusive task/retention alternatives. The Python attribute is `ShadowSpillProgram.task_alternative_groups`; the serialized key keeps its original spelling so an existing corpus stays readable. |

### Device, alias, and object records

| Record | Keys | Meaning |
|---|---|---|
| Device | `device_id`, `process_id`, `kind`, `index` | Logical resource identity. |
| Alias group | `alias_group_id`, `device_id`, `size_bytes`, `initial_version`, `retain_spill_copy`, `shared_residency` | One storage root and its version, spill-retention, and optional runtime-global sharing policy. |
| Object | `object_id`, `alias_group_id`, `offset_bytes`, `size_bytes`, `role`, `persistence` | One view into a root. |

`ObjectSpec.size_bytes` is the logical view span. `AliasGroupSpec.size_bytes` is
the full root extent used for residency and transfer accounting. Several
objects may name the same alias group with different offsets.

Object roles are `input`, `parameter`, `buffer`, `activation`, `gradient`,
`optimizer_state`, `output`, `control`, or `other`. Persistence is `step`,
`run`, or `checkpoint`.

### Profile and task records

| Record | Key | Meaning |
|---|---|---|
| Profile | `profile_id` | Stable profile identity referenced by tasks. |
| Profile | `runtime_ns` | Isolated measured task duration used by simulation. |
| Profile | `workspace_bytes` | Peak anonymous task workspace used by logical simulation. |
| Profile | `compatibility_digest` | Structural/physical contract compatibility identity. |
| Task | `task_id` | Stable canonical IR identity. |
| Task | `resource` | Device, resource kind, and lane. |
| Task | `profile_id` | Cost profile reference. |
| Task | `dependencies` | Earlier tasks that causally produce or order this task. |
| Task | `inputs`, `outputs` | Object IDs read and produced. |
| Task | `mutations` | `{object_id, version_delta}` records. |
| Task | `phase` | Semantic phase such as forward, backward, or optimizer. |
| Task | `requires_entrypoint` | Whether runtime materialization must bind a compiled callable. |

Tasks appear in topological order. Dependencies and all object/profile
references are validated during construction and loading.

### Graph-pair groups

Each record has a `group_id` and an `options` array. Each option contains:

| Key | Meaning |
|---|---|
| `option_id` | Stable alternative identity. |
| `active_task_ids` | Tasks included when this option is selected. |
| `retained_alias_group_ids` | Alias groups retained across the alternative's boundary. |

Exactly one option must be selected from every group before schedule
validation. Graph-pair construction may emit save/recompute alternatives, but
the IR permits more than two options.

## ShadowSpillPlanningProblem format

`ShadowSpillPlanningProblem` packages a `ShadowSpillProgram` for independent calls to
`plan_program()`:

```text
shadowspill.plan_program/v1
├── role
├── program
│   ├── digest
│   └── value                 complete shadowspill.program/v1
├── residency
│   ├── initial
│   └── final
├── capacity_contract
├── simulation_config
└── admission_facts
```

| Key | Meaning |
|---|---|
| `role` | `initial`, `recurrent`, or `forward`. |
| `program.digest` | Integrity identity for `program.value`. |
| `residency.initial`, `residency.final` | Required alias-group location/version at the phase boundaries. |
| `capacity_contract` | Source/max execution and spill budgets plus fixed, object, and dynamic-scratch deductions. |
| `simulation_config` | Logical device object capacity, spill capacity, directional bandwidth, and latency. |
| `admission_facts` | Current-schema (`shadowspill.admission_facts/v1`) per-task allocation traces, derived anonymous peaks, ownership transitions, handoffs, and physical capacity. |

Residency entries identify `alias_group_id` and `location` (`device` or `host`,
the schema-v1 spelling of the spill pool
in the neutral IR). These serialized IR labels should not be confused with
user-chosen runtime pool names such as `execution` and `spill`.

Each admission task contains `allocation_steps`, `workspace_extents`, fresh
and replacement alias lists, and storage handoffs. Allocation steps record
allocate/free order, charged bytes, stable task-local ordinals, optional
persistent alias ownership, and same-task reuse. They never contain pointers
or slab offsets. `workspace_extents` is the anonymous peak reconstructed from
those steps, not a second source of physical geometry. Missing allocation
evidence is a planning error; loaders do not synthesize it or accept an older
schema.

The capacity contract keys are:

| Key | Meaning |
|---|---|
| `source_execution_budget_bytes` | Budget used when constructing this artifact. |
| `maximum_execution_budget_bytes` | Largest execution budget allowed without recompilation/reprofiling. |
| `maximum_spill_budget_bytes` | Largest spill budget allowed by the source runtime. |
| `fixed_execution_bytes` | Problem/provider/fixed-service bytes outside the callable pool. |
| `object_reserve_bytes` | Capacity leeway: pool bytes withheld from PressureFit's object capacity so a fixed layout whose extent exceeds the planner's instantaneous bound can still be admitted without capacity refinement. Not a workspace partition — task workspace is charged per boundary and placed inside the fixed slice. |
| `dynamic_scratch_reserve_bytes` | Measured or user-raised optional dynamic scratch requirement. |

## StepProgram format

A training `StepProgram` retains both recurrent and optional initialization
roles:

```text
shadowspill.step_program/v1
├── identity
│   ├── signature_digests
│   ├── recurrent_program_digest
│   └── initial_program_digest
├── programs
│   ├── recurrent             ShadowSpillPlanningProblem
│   └── initial               ShadowSpillPlanningProblem or null
├── profiling
│   ├── metadata
│   ├── unique_profile_count
│   └── captured_stage_count
├── planning
│   ├── optimizer_ordering
│   ├── data_ordering         depth, breadth, reverse_breadth, pair_loss
│   └── phase_timings_ns
├── transfer_capabilities
└── cache_lineage
    ├── directories
    └── artifacts
```

`profiling.metadata` is planning identity for data-dependent measurement
effects; it is not a runtime model input. `transfer_capabilities` is the
runtime calibration matrix captured during program construction.
`cache_lineage` explains where artifacts came from but does not participate in
`StepProgram.digest`.

## AnnotatedProgramPlan format

An annotated plan is one admitted planning point:

```text
shadowspill.annotated_program_plan/v1
├── source_program
├── memory_budgets
├── transfer_bandwidths
├── selection
├── simulation
├── physical_admission
└── timing
```

### Top-level annotated-plan keys

| Key | Meaning |
|---|---|
| `source_program` | Complete `ShadowSpillPlanningProblem` from which the point was selected. |
| `memory_budgets` | Requested physical execution and spill capacities. |
| `transfer_bandwidths` | Exact fetch/evict rates and calibration identity used for the point. |
| `selection` | The winning resolved program and candidate, schedule, residency, and search diagnostics. |
| `simulation` | Final admitted simulation result and physical deltas/dependencies consumed by it. |
| `physical_admission` | Effective topology, fixed layout, layout digest, and all refinement attempts. |
| `timing` | Separate search, admission, orchestration, and total planning wall time. |

`memory_budgets` contains `execution_bytes` and `spill_bytes`.
`transfer_bandwidths` contains:

- `fetch_bytes_per_second` and `evict_bytes_per_second`;
- `fetch_latency_ns` and `evict_latency_ns`, the per-transfer latencies the
  same calibration measured, or `null` in a record written before they were
  carried and in an override that names only bandwidths, where the program's
  own latency applies;
- `scale_numerator` and `scale_denominator` for an exact rational benchmark
  scaling factor;
- optional `calibration_digest` and `provenance`.

### Selection

| Key | Meaning |
|---|---|
| `from_store` | Whether the selected plan was read back from the plan store rather than planned during this call. |
| `diagnostics` | Full resolved-program and candidate-policy search evidence. |
| `initial_residency`, `final_residency` | Selected boundary state. |
| `search_options` | Everything the search was told: `generic`, and the `algorithm` that ran with its own `options`. `workers` is absent, since it changes how long an answer takes rather than which answer is right. |
| `resident_slice` | The slice reserved for the objects `minimum_object_bytes_evict_eligible` kept resident: its `bytes`, the sum of the static homes their leases take, and the `aliases` it holds; empty when it kept none. |
| `schedule` | `shadowspill.memory_schedule/v1` with ordered actions. |
| `selections` | One chosen option per task-alternative group. |

The schedule contains `initial_residency`, ordered `actions`, and
`final_residency`. An action records its kind (`release`, `evict`,
`fetch` or `write_back`), trigger task, and alias group. Array order is the directive order
at equal or increasing trigger boundaries; the alias group identifies its
device through the program.

The diagnostics hierarchy is:

```text
selected resolved program
└── selected candidate policy

all resolved programs
├── incumbent — the plan to beat, when one was handed in
└── candidate-policy evaluations
    ├── outcome
    ├── repair counts
    ├── placement counts
    ├── work counters
    ├── work.sections — where the time went
    ├── span — when it ran
    └── steps — the reduction trajectory, when recorded
```

Each evaluation's outcome carries what the candidate's own capacity search
did, alongside its makespan:

| Field | Meaning |
|---|---|
| `capacity_violation_count` | Times the plan waited for memory it did not have. A plan that stalls is valid but unfinished. |
| `placements_attempted` | Layouts measured. Only a plan that could still win is measured, so this counts plans that were worth the cost. |
| `placements_admitted` | Measured layouts that fit the pool. The candidate answers with the best of these. |
| `capacity_refinements` | Times the candidate gave capacity back because its layout did not fit, and planned again. |
| `repairs_at_best` | Repairs spent when the plan the candidate answers with was placed; `null` when it placed none. |
| `pressure_escalations` | Pressure repairs that asked for more than the shortfall because the same failure had repeated at the same task and moment. |
| `escalations_taken_back` | Escalated asks no cut could meet, taken back for a plain ask. |

A candidate whose status is `infeasible` with failure kind `unplaceable`
reached no plan that fit, so it has no answer regardless of what it
simulated.

A resolved program that was handed the plan to beat — a plan for it already in
hand, found at a smaller budget, say — carries an `incumbent` block saying
what became of it at this capacity. The search measures it before any
candidate runs and answers with it unless a candidate does strictly better;
when it answers with it, the resolved program's
`selected_candidate_policy.candidate_id` is `incumbent`. `null` on a resolved
program handed none, and absent from records written before there was one.

| Field | Meaning |
|---|---|
| `status` | `valid` when it simulated and its layout fit the pool; `unplaceable` when the layout did not; `infeasible` when it did not simulate or admit here; `error` when the library could not measure it. |
| `makespan_ns` | What it simulated to at this capacity. |
| `required_bytes` | Pool bytes its layout needed, when it was measured against a pool. |
| `selected` | Whether it is the resolved program's answer. |
| `schedule_digest`, `found_by`, `found_at_capacity_bytes` | Which plan it is, the candidate policy that first found it, and the object capacity it was first found at, both read through any chain of hand-offs. |

`work` counts what the search did — residency and schedule cache hits,
simulation calls, admission calls — and `work.sections` says where its time
went. The sections are disjoint spans named for the stage that produced them:

| Key | Span |
|---|---|
| `prepare_ns` | Deriving the residency problem. Problem level only. |
| `setup_ns` | Schedule facts and the candidate workspace. |
| `reduce_ns` | Choosing what stays resident, before any candidate repairs it. |
| `emit_ns` | Turning residency gaps into an ordered schedule. |
| `simulate_ns` | Replaying the schedule for a makespan. |
| `repair_ns` | Moving a transfer or making room for one, including the reduction that takes. |
| `digest_ns` | Naming the schedule. |
| `place_ns` | Measuring whether the layout fits. |
| `select_ns` | Deciding what to answer with, and materialising it. |
| `teardown_ns` | Releasing what the evaluation held. |
| `residual_ns` | The part of `total_ns` no named section claimed. |
| `admit_ns` | Admitting the schedule. **Nested inside `simulate_ns`.** |

`total_ns` equals the sum of every key above except `admit_ns`, at each level
of the hierarchy, so a breakdown always accounts for the whole span rather
than most of it.

Sections measure work, not elapsed time. A problem's sections are the sum of
its candidates', so with several workers the total exceeds the time the call
took — that gap is the point of the workers. `span` is the other measurement,
wall clock rather than work:

| Key | Meaning |
|---|---|
| `started_ns` | When this candidate or problem started, from the beginning of the call that evaluated it. |
| `finished_ns` | When it finished, on the same clock. |

Because every span in one call shares an origin, two candidates ran at the
same time exactly when their spans overlap, and a problem spans its
candidates. Both are zero for a candidate no worker reached. Spans are
measurements of a run rather than properties of a plan, so anything compared
or digested across runs leaves them out, as it does `sections`.

`steps` is present only when planning was asked to record trajectories. Each
entry is one plan the candidate held, in order:

| Key | Meaning |
|---|---|
| `makespan_ns` | What that plan simulated to. |
| `required_bytes` | Bytes its layout spanned. Zero unless it was measured. |
| `capacity_bytes` | The object capacity it was built against. |
| `cut_aliases` | Objects the reducer cut to reach it. |
| `repairs` | Repairs spent by the time it was reached. |
| `capacity_violations` | Places it came up short and waited. |
| `outcome` | `simulated`, `measured`, `placed`, `refined`, `best`, `answer`. |

See [Interpreting a PlanReport](plan-report.md#pressurefit-diagnostics) for the
meaning of a problem versus a policy.

### Simulation

`simulation.result` records:

- `makespan_ns`;
- task intervals with ready/start/end, resource, workspace, and stall reasons;
- transfer intervals with trigger, direction, sequence, bytes, ready/start/end,
  and stall reasons;
- per-device object/workspace/total peaks;
- spill peak;
- optional memory timeline;
- `capacity_violations`, each an instant where the plan wanted more than its
  budget allowed, with the reason, location, capacities and excess, and
  `capacity_violation_count`. A violation is stall the plan paid for, not a
  rejection; a count larger than the recorded list means the list was
  truncated.

`simulation.admission` records timing-independent physical facts: initial
physical bytes, device capacities, task start/completion deltas, action
trigger/completion deltas, and cross-lane memory-reuse dependencies.

### Physical admission

| Key | Meaning |
|---|---|
| `effective_facts` | Capacity-adjusted topology used by the accepted attempt. |
| `fixed_layout` | Complete `shadowspill.fixed_physical_layout/v1` certificate. |
| `fixed_layout_digest` | Integrity identity of that certificate. |
| `attempts` | Ordered capacity-refinement trials and optional PressureFit diagnostics. |

The fixed-layout certificate binds program, schedule, and topology digests. It
records pool/fixed/dynamic/scratch/required bytes, every placement, causal
reuse dependencies, dynamic lifetimes, initial-object leases, task-allocation
leases, and transfer-action destination leases. Offsets are relative to the
callable fixed slice, not raw process pointers.

Each attempt records requested/effective object capacity, required bytes,
pool capacity, accepted status, and the PressureFit evidence for that trial.

### Timing

| Key | Meaning |
|---|---|
| `total_wall_time_ns` | Complete `plan_program()` wall time. |
| `search_wall_time_ns` | Sum of PressureFit/cache-resolution intervals across attempts. |
| `physical_admission_wall_time_ns` | Sum of physical-layout construction intervals. |
| `orchestration_wall_time_ns` | Remaining validated orchestration time. |
| `refinement_attempts` | Per-attempt PressureFit and physical-admission timing. |

The three component totals reconcile with total wall time. Search work time
inside PressureFit diagnostics is normally larger than wall time, because the
search evaluates many candidates at once and `sections` counts work rather
than elapsed time; `span` is the wall-clock counterpart.

## Loading and validation

The `from_json()` constructors do more than parse syntax. Depending on the
artifact they validate:

- schema labels and field types;
- Program cross-references and topological order;
- embedded program, schedule, topology, and layout digests;
- one legal task-alternative option per group;
- residency and memory-action legality;
- physical-layout identity and capacity;
- simulation makespan against the selected result;
- timing reconciliation.

Treat a load failure as an invalid, stale, or incompatible artifact. Do not
strip validation evidence to make an artifact load.

## Choosing the right artifact

| Goal | Use |
|---|---|
| Inspect or hand-author a framework-neutral workload | `ShadowSpillProgram` |
| Sweep budgets or bandwidths for one recurrent/forward role | `ShadowSpillPlanningProblem` |
| Preserve all capture/profile work for a PyTorch training step | `StepProgram` |
| Preserve one selected, simulated, physically admitted point | `AnnotatedProgramPlan` |
| Preserve one planning call's explanatory tree | `PlanReport.diagnostics.as_dict()` |
| Preserve one real execution observation | `StepDiagnostics.as_dict()` |

The final two are diagnostic dictionaries, not reloadable planning artifacts.
The [reusable artifact API](api/artifacts.md) shows the public constructors and
planning calls.
