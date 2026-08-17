# Program and annotated-plan JSON

ShadowSpill exposes content-addressed JSON artifacts at distinct planning
boundaries. They are canonical, schema-tagged, validated on load, and designed
for corpus collection, budget sweeps, inspection, and reproducible planning.

| Python value | Schema | Boundary |
|---|---|---|
| `Program` | `shadowspill.program/v1` | Framework-neutral logical tasks, objects, costs, and recomputation choices. |
| `PressureFitProgram` | `shadowspill.pressurefit_program/v1` | One Program plus residency, machine inputs, admission topology, and search options. |
| `StepProgram` | `shadowspill.step_program/v1` | Complete PyTorch capture/profile result with recurrent and optional initial PressureFit Programs. |
| `AnnotatedProgramPlan` | `shadowspill.annotated_program_plan/v2` | PressureFit winner, physical admission, and simulator evidence for one budget/bandwidth point. |

The ordinary reusable workflow is:

```python
from pathlib import Path

from shadowspill.pytorch import AnnotatedProgramPlan, StepProgram

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

1. A `Program.digest` covers the complete logical Program.
2. A `StepProgram.digest` excludes phase time and cache paths, so the same
   planning content has one identity regardless of where or how long it took
   to construct.
3. An `AnnotatedProgramPlan.digest` excludes cache-hit status, diagnostic work
   time, and orchestration wall time, while retaining selected schedule,
   budgets, transfer bandwidths, and the physical certificate.

Never edit a digest independently of its value. `from_json()` recomputes and
validates embedded identities and rejects inconsistent content.

## Program format

An abridged `Program` has this shape:

```json
{
  "schema": "shadowspill.program/v1",
  "devices": [
    {"device_id": "cuda_0", "process_id": "process_0", "kind": "cuda", "index": 0}
  ],
  "alias_groups": [
    {"alias_group_id": "alias_0", "device_id": "cuda_0", "size_bytes": 4096,
     "initial_version": 0, "retain_spill_copy": false}
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
    {"task_id": "task_0", "resource": {"device_id": "cuda_0", "kind": "compute", "lane": 0},
     "profile_id": "profile_0", "dependencies": [], "inputs": [],
     "outputs": ["activation_0"], "mutations": [], "phase": "forward",
     "requires_entrypoint": true}
  ],
  "recomputation_groups": []
}
```

### Top-level Program keys

| Key | Value |
|---|---|
| `schema` | Exact schema label used for versioned parsing. |
| `devices` | Logical execution devices and their process-local indices. |
| `alias_groups` | Physical logical storage roots; capacity is charged at this level. |
| `objects` | Named tensor views into alias groups. |
| `profiles` | Deduplicated task runtime/workspace measurements. |
| `tasks` | Topologically ordered executable/control tasks. |
| `recomputation_groups` | Mutually exclusive task/retention alternatives. |

### Device, alias, and object records

| Record | Keys | Meaning |
|---|---|---|
| Device | `device_id`, `process_id`, `kind`, `index` | Logical resource identity. |
| Alias group | `alias_group_id`, `device_id`, `size_bytes`, `initial_version`, `retain_spill_copy` | One storage root and its version/spill-retention policy. |
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
| Profile | `compatibility_digest` | Structural/physical ABI compatibility identity. |
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

### Recomputation groups

Each record has a `group_id` and an `options` array. Each option contains:

| Key | Meaning |
|---|---|
| `option_id` | Stable alternative identity. |
| `active_task_ids` | Tasks included when this option is selected. |
| `retained_alias_group_ids` | Alias groups retained across the alternative's boundary. |

Exactly one option must be selected from every group before schedule
validation. Graph-pair construction may emit save/recompute alternatives, but
the IR permits more than two options.

## PressureFitProgram format

`PressureFitProgram` packages a `Program` for independent calls to
`pressurefit_program()`:

```text
shadowspill.pressurefit_program/v1
├── role
├── program
│   ├── digest
│   └── value                 complete shadowspill.program/v1
├── residency
│   ├── initial
│   └── final
├── capacity_contract
├── simulation_config
├── admission_topology
└── pressurefit_options
```

| Key | Meaning |
|---|---|
| `role` | `initial`, `recurrent`, or `forward`. |
| `program.digest` | Integrity identity for `program.value`. |
| `residency.initial`, `residency.final` | Required alias-group location/version at the phase boundaries. |
| `capacity_contract` | Source/max execution and spill budgets plus fixed, object, and dynamic-scratch deductions. |
| `simulation_config` | Logical device object capacity, spill capacity, directional bandwidth, and latency. |
| `admission_topology` | Per-task allocation geometry, ownership transitions, handoffs, and physical capacity. |
| `pressurefit_options` | Bounded search controls and repair limits. |

Residency entries identify `alias_group_id` and `location` (`device` or `host`
in the neutral IR). These serialized IR labels should not be confused with
user-chosen runtime pool names such as `execution` and `spill`.

The capacity contract keys are:

| Key | Meaning |
|---|---|
| `source_execution_budget_bytes` | Budget used when constructing this artifact. |
| `maximum_execution_budget_bytes` | Largest execution budget allowed without recompilation/reprofiling. |
| `maximum_spill_budget_bytes` | Largest spill budget allowed by the source runtime. |
| `fixed_execution_bytes` | Context/provider/fixed-service bytes outside the callable pool. |
| `object_reserve_bytes` | Pool bytes withheld from PressureFit object capacity for physical allocation needs. |
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
│   ├── recurrent             PressureFitProgram
│   └── initial               PressureFitProgram or null
├── profiling
│   ├── metadata
│   ├── unique_profile_count
│   └── captured_stage_count
├── planning
│   ├── optimizer_ordering
│   └── phase_timings_ns
├── transfer_capabilities
└── cache_lineage
    ├── directories
    └── artifacts
```

`profiling.metadata` is planning identity for data-dependent measurement
effects; it is not a runtime model input. `transfer_capabilities` is the
runtime calibration matrix captured during Program construction.
`cache_lineage` explains where artifacts came from but does not participate in
`StepProgram.digest`.

## AnnotatedProgramPlan format

An annotated plan is one admitted planning point:

```text
shadowspill.annotated_program_plan/v2
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
| `source_program` | Complete `PressureFitProgram` from which the point was selected. |
| `memory_budgets` | Requested physical execution and spill capacities. |
| `transfer_bandwidths` | Exact fetch/evict rates and calibration identity used for the point. |
| `selection` | PressureFit context/policy winner, schedule, residency, and search diagnostics. |
| `simulation` | Final admitted simulation result and physical deltas/dependencies consumed by it. |
| `physical_admission` | Effective topology, fixed layout, layout digest, and all refinement attempts. |
| `timing` | Separate PressureFit, admission, orchestration, and total planning wall time. |

`memory_budgets` contains `execution_bytes` and `spill_bytes`.
`transfer_bandwidths` contains:

- `fetch_bytes_per_second` and `evict_bytes_per_second`;
- `scale_numerator` and `scale_denominator` for an exact rational benchmark
  scaling factor;
- optional `calibration_digest` and `provenance`.

### Selection

| Key | Meaning |
|---|---|
| `cache_hit` | Whether the selected PressureFit result came from the artifact cache. |
| `diagnostics` | Full recomputation-context and candidate-policy search evidence. |
| `initial_residency`, `final_residency` | Selected boundary state. |
| `options` | Effective `PressureFitOptions`. |
| `schedule` | `shadowspill.memory_schedule/v1` with ordered actions. |
| `selections` | One chosen option per recomputation group. |

The schedule contains `initial_residency`, ordered `actions`, and
`final_residency`. An action records its kind (`release`, `offload`, or
`prefetch`), trigger task, and alias group. Array order is the directive order
at equal or increasing trigger boundaries; the alias group identifies its
device through the Program.

The diagnostics hierarchy is:

```text
selected recomputation context
└── selected candidate policy

all recomputation contexts
└── candidate-policy evaluations
    ├── outcome
    ├── repair counts
    └── simulation/admission/work counters and time
```

See [Interpreting a PlanReport](plan-report.md#pressurefit-diagnostics) for the
meaning of a context versus a policy.

### Simulation

`simulation.result` records:

- `makespan_ns`;
- task intervals with ready/start/end, resource, workspace, and stall reasons;
- transfer intervals with trigger, direction, sequence, bytes, ready/start/end,
  and stall reasons;
- per-device object/workspace/total peaks;
- spill peak;
- optional memory timeline.

`simulation.admission` records timing-independent physical facts: initial
physical bytes, device capacities, task start/completion deltas, action
trigger/completion deltas, and cross-lane memory-reuse dependencies.

### Physical admission

| Key | Meaning |
|---|---|
| `effective_topology` | Capacity-adjusted topology used by the accepted attempt. |
| `fixed_layout` | Complete `shadowspill.fixed_physical_layout/v3` certificate. |
| `fixed_layout_digest` | Integrity identity of that certificate. |
| `attempts` | Ordered capacity-refinement trials and optional PressureFit diagnostics. |

The fixed-layout certificate binds Program, schedule, and topology digests. It
records pool/fixed/dynamic/scratch/required bytes, every placement, causal
reuse dependencies, dynamic lifetimes, initial-object leases, task-allocation
leases, and transfer-action destination leases. Offsets are relative to the
callable fixed slice, not raw process pointers.

Each attempt records requested/effective object capacity, required bytes,
pool capacity, accepted status, and the PressureFit evidence for that trial.

### Timing

| Key | Meaning |
|---|---|
| `total_wall_time_ns` | Complete `pressurefit_program()` wall time. |
| `pressurefit_wall_time_ns` | Sum of PressureFit/cache-resolution intervals across attempts. |
| `physical_admission_wall_time_ns` | Sum of physical-layout construction intervals. |
| `orchestration_wall_time_ns` | Remaining validated orchestration time. |
| `refinement_attempts` | Per-attempt PressureFit and physical-admission timing. |

The three component totals reconcile with total wall time. Search work time
inside PressureFit diagnostics may be larger than wall time when independent
contexts are evaluated concurrently.

## Loading and validation

The `from_json()` constructors do more than parse syntax. Depending on the
artifact they validate:

- schema labels and field types;
- Program cross-references and topological order;
- embedded Program, schedule, topology, and layout digests;
- one legal recomputation option per group;
- residency and memory-action legality;
- physical-layout identity and capacity;
- simulation makespan against the selected result;
- timing reconciliation.

Treat a load failure as an invalid, stale, or incompatible artifact. Do not
strip validation evidence to make an artifact load.

## Choosing the right artifact

| Goal | Use |
|---|---|
| Inspect or hand-author a framework-neutral workload | `Program` |
| Sweep budgets or bandwidths for one recurrent/forward role | `PressureFitProgram` |
| Preserve all capture/profile work for a PyTorch training step | `StepProgram` |
| Preserve one selected, simulated, physically admitted point | `AnnotatedProgramPlan` |
| Preserve one planning call's explanatory tree | `PlanReport.diagnostics.as_dict()` |
| Preserve one real execution observation | `StepDiagnostics.as_dict()` |

The final two are diagnostic dictionaries, not reloadable planning artifacts.
The [reusable artifact API](api/artifacts.md) shows the public constructors and
planning calls.
