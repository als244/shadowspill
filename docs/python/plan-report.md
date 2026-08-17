# Interpreting a PlanReport

Every successful `plan_step()` or `plan_forward()` returns a callable whose
`plan_report` explains what was captured, measured, selected, admitted, and
published. Planning diagnostics are always collected; `verbose=False` only
suppresses console progress.

```python
report = train_step.plan_report

print(report.mode)
print(report.program.digest)
print(report.predicted_makespan_ns / 1e9)
print(report.predicted_device_peak_bytes)
print(report.predicted_host_peak_bytes)
```

`PlanReport` is the in-memory report attached to a callable. It is distinct
from the portable `StepProgram`, `PressureFitProgram`, and
`AnnotatedProgramPlan` JSON artifacts described in [Program and annotated-plan
JSON](planning-json.md).

## Read the report from the outside in

The most useful inspection order is:

1. Confirm pools, budgets, device, and calibrated transfer rates.
2. Check the predicted makespan and physical peaks.
3. Inspect planning phase time and cache behavior.
4. Inspect the selected task sequence by chronological execution ID.
5. Compare each selected task with its unique stage and graph-pair options.
6. Inspect PressureFit search evidence and physical-layout admission.
7. Drill into a structural graph profile only when timing or byte accounting
   needs explanation.

The top-level fields are grouped below.

| Area | Important fields | Meaning |
|---|---|---|
| Identity | `mode`, `capture_identity`, `program.digest` | Planning mode and content identities. |
| Selected plan | `execution_plan`, `initial_execution_plan` | Recurrent/forward plan and optional first-step plan for lazy state. |
| Prediction | `predicted_makespan_ns`, `predicted_device_peak_bytes`, `predicted_host_peak_bytes` | Simulator result after physical admission. |
| Capacity | `execution_pool`, `spill_pool`, `execution_budget_bytes`, `spill_budget_bytes`, `fixed_slab_bytes`, `requested_dynamic_scratch_reserve_bytes` | Pool selection, public budgets, process-persistent deductions, and requested scratch floor. |
| Transfers | `fetch_profile`, `evict_profile`, `transfer_actions`, `transfer_bytes_fetched`, `transfer_bytes_evicted` | Calibration consumed by planning and selected traffic. |
| Profiling | `task_profiles`, profile hit/miss counts, allocation-probe counts | Deduplicated structural measurements and their provenance. |
| Selection | `pressurefit_result`, `initial_pressurefit_result` | PressureFit winner, schedule, selections, and search evidence. |
| Detailed evidence | `diagnostics` | Phase, cache, stage, graph-pair, profile, search, and layout records. |

For training, `report.program` and `report.pressurefit_result` refer to the
recurrent plan. `initial_program` and `initial_pressurefit_result` refer to the
optional first invocation. Forward planning has one plan.

## Transfer assumptions and budgets

Always verify these before interpreting a predicted makespan:

```python
print(report.execution_pool, report.execution_budget_bytes)
print(report.spill_pool, report.spill_budget_bytes)

print(report.fetch_profile)
print(report.evict_profile)
print(report.transfer_capabilities)
```

`fetch_profile` is the measured spill-to-execution direction;
`evict_profile` is execution-to-spill. Runtime initialization measures each
direction alone and during simultaneous bidirectional traffic. Planning uses
the conservative concurrent-direction rates and route latency retained in
`transfer_capabilities`.

The predicted device peak is the simulator's admitted physical peak, not just
the sum of logical objects. `fixed_slab_bytes` is process-persistent
fixed/provider memory removed before callable admission; it is not the
callable's `fixed_slice_bytes`. The physical-layout diagnostics explain how
the remaining pool is divided among logical object capacity, the fixed
callable slice, terminal dynamic outputs, and bounded dynamic scratch.

## Planning wall time and cache use

`report.diagnostics.phases` contains mutually exclusive frontend planning
intervals. Their sum is `measured_wall_time_ns`; adding
`unattributed_overhead_ns` equals `total_wall_time_ns`.

```python
diagnostics = report.diagnostics

for phase in diagnostics.phases:
    print(phase.name, phase.duration_seconds)

print(diagnostics.measured_wall_time_ns)
print(diagnostics.unattributed_overhead_ns)
print(diagnostics.total_wall_time_ns)
```

The other planning-cost views are:

| Field | Interpretation |
|---|---|
| `compiler_phase_timings_ns` | Aggregate compiler phases across structural ABIs. |
| `compiler_profiles` | Per-structural-ABI compiler phase breakdown. |
| `profile_unique_keys` | Number of structural profiles needed by the call. |
| `profile_cache_hits`, `profile_cache_misses` | Measurement reuse versus fresh profiling. |
| `aot_graph_pair_cache_hits`, `aot_graph_pair_cache_misses` | Reuse versus construction of differentiated graph pairs. |
| `recomputation_cache_hits`, `recomputation_cache_misses` | Reuse of recomputation-selection artifacts. |
| `cache_directories` | Cache roots used for this call. |
| `cache_artifacts` | Every managed, matched, read, or written artifact and its dependencies. |

An artifact with `access="matched"` agreed with a freshly produced in-memory
value but was not read as planning authority. `managed` identifies a directory
owned by another component, such as the compiler cache. The [planning cache
guide](planning-cache.md) defines the directory and identity contract.

## Tasks are keyed by execution ID

`execution_XXXXXX` is the primary runtime and diagnostics identity. It is a
dense chronological ordinal after recomputation selection. The canonical IR
task ID remains available as `task_id` for stable Program lookup.

```python
tasks = report.diagnostics.tasks

task = tasks["execution_000017"]
print(task.execution_ordinal)
print(task.semantic_name)
print(task.phase, task.microbatch)
print(task.task_id)
print(task.unique_stage_id)
print(task.structural_abi_key)
print(task.chosen_graph_pair_variant)
```

`report.diagnostics.task(execution_task_id)` performs the same primary lookup.
Use `task_by_ir_id(task_id)` only when starting from a canonical Program task.

`task_stage_map` contains both selected and unselected task variants. Its
`selected` flag and optional `execution_task_id` distinguish the chosen
projection. This is useful when answering why one save/recompute alternative
was absent from execution.

## Unique stages and graph pairs

`unique_stages` deduplicates repeated model positions with the same structural
ABI. Each `PlanUniqueStage` records module targets, occurrence count, and all
legal `PlanGraphPair` alternatives.

```python
stage = next(
    item
    for item in report.diagnostics.unique_stages
    if item.unique_stage_id == task.unique_stage_id
)

for pair in stage.graph_pairs:
    print(
        pair.variant,
        pair.recomputation,
        pair.saved_value_count,
        pair.forward.runtime_ns,
        None if pair.backward is None else pair.backward.runtime_ns,
    )
```

A graph pair records the selected memory-budget alternative, recomputation
flag, saved-value categories and bytes, and forward/backward physical profiles.
The task record's `chosen_graph_pair_variant` is the direct bridge from an
execution ID to the chosen alternative.

## Interpreting a graph profile

`PlanGraphProfile` separates semantic identity from compiled physical
behavior.

| Group | Fields | Question answered |
|---|---|---|
| Semantic contract | `semantic_contract_digest`, `semantic_roots`, `semantic_output_views`, `semantic_mutations` | What aliases, views, and mutations does the graph mean? |
| Executable contract | `executable_contract_digest`, `executable_roots`, `executable_output_views`, `executable_mutations` | What storage relationships must the callable expose? |
| Compiled layout | `compiled_layout_digest`, `compiled_roots`, `compiled_output_views` | Which compiled allocations back returned roots and views? |
| Values | `representative_inputs`, `provenance` | Which initialized, caller-supplied, producer-derived, or deterministic synthetic values were profiled? |
| Timing | `runtime_ns`, `samples_ns`, `timing_relative_mad`, `timing_half_drift`, `timing_unstable` | How stable is the task-duration estimate? |
| Object bytes | `inputs`, `mutations`, `outputs` and logical/allocation totals | Which named values create pressure? |
| Temporary bytes | workspace fields and extent lists | How much anonymous task memory was live? |
| Allocation behavior | `allocation_contract_digest`, `allocation_contract`, `allocation_timeline` | Which strict core operations and observed lifetime events were admitted? |

Logical bytes describe tensor views; allocation bytes describe the containing
storage extents. Do not add input, mutation, and output totals blindly: a view
may share an alias group, and a mutation may replace an existing generation.
Use the alias-group/object model in the canonical Program when computing a
physical inventory.

`timing_unstable=True` means the profiling sampler did not converge within its
configured limit. A material unstable task should be investigated before its
simulator prediction is used as a performance authority.

## PressureFit diagnostics

`report.diagnostics.pressurefit_runs` contains one entry for each selected
planning role/refinement run. Each run has this hierarchy:

```text
PressureFit invocation
└── recomputation context (one complete graph-pair selection)
    └── candidate-policy evaluation
        ├── residency strategy
        ├── fetch-trigger rule
        ├── coalescing mode
        ├── outcome
        ├── repair counts
        └── work counts and component time
```

A candidate policy is the combination of residency strategy, fetch rule, and
coalescing mode. A recomputation context is a complete selection of one option
from every recomputation group. One policy can therefore be evaluated in many
contexts.

Start with the invocation's selected selection/candidate IDs and makespan,
then inspect:

- summary counts for contexts, policies, evaluations, valid evaluations, and
  status categories;
- `repairs` to see whether admission or simulation repeatedly moved fetches or
  pressure boundaries;
- `work.simulation` and `work.admission` to count compiled calls and cache
  reuse;
- each failed candidate's `failure_kind` and `failure_detail`.

Invocation-level work includes shared setup and result materialization. It
need not equal the sum of candidate work. Component work time is summed work,
not necessarily elapsed wall time when independent contexts run concurrently.

## Physical-layout diagnostics

Each `PlanPhysicalLayout` describes one admitted role.

| Field | Meaning |
|---|---|
| `strategy` and `layout_digest` | Placement strategy and certificate identity. |
| `pool_capacity_bytes` | Callable-attributable physical execution-pool capacity. |
| `original_object_capacity_bytes` | Initial logical object capacity sent to PressureFit. |
| `effective_object_capacity_bytes` | Object capacity of the accepted refinement. |
| `object_capacity_reduction_bytes` | Capacity ceded to make physical placement feasible. |
| `fixed_slice_bytes` | Reusable fixed range required by admitted lifetimes. |
| `dynamic_reserve_bytes` | Terminal outputs that may outlive the reusable slice. |
| `scratch_reserve_bytes` | Bounded optional dynamic task allocations. |
| `required_bytes`, `slack_bytes` | Total admitted bytes and remaining pool capacity. |
| `reuse_dependency_count` | Cross-lane causal edges required for safe range reuse. |
| `attempts` | PressureFit/admission refinement history. |
| `task_memory_envelopes` | Per-task strict-core and dynamic-scratch limits. |

For every accepted layout, `required_bytes <= pool_capacity_bytes`. Repeated
failed attempts with a large `object_capacity_reduction_bytes` indicate a
physical-placement constraint, not automatically a logical PressureFit bug.
Use [Physical admission and offset handling](../architecture/physical-admission.md)
to interpret the certificate.

## Exporting diagnostic evidence

`PlanDiagnostics.as_dict()` returns a JSON-friendly value with schema
`shadowspill.plan_diagnostics/v1`:

```python
import json

payload = report.diagnostics.as_dict()
with open("plan-diagnostics.json", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
```

This dictionary is diagnostic evidence, not the portable planning input or
selected-plan artifact. Use `StepProgram.to_json()` or
`AnnotatedProgramPlan.to_json()` when the value must be loaded back into a
planning workflow.

## Common investigations

| Symptom | Look here first |
|---|---|
| Planning is slow | `diagnostics.phases`, compiler profiles, cache hits/misses, then PressureFit work counts. |
| Predicted step is slow | Selected recomputation context, candidate policy, transfer bytes, task profiles, and simulator makespan. |
| One task is unexpectedly large | Execution task → unique stage → chosen graph pair → forward/backward graph profile byte fields. |
| Save and recompute look identical | Graph-pair saved-value counts/bytes, active tasks, and semantic root/output contracts. |
| Plan repeatedly refines capacity | Physical-layout attempts, required/slack bytes, dynamic/scratch reserves, and PressureFit repairs. |
| Cache reuse is surprising | `cache_artifacts`, dependency digests, profiling metadata, implementation revision, and allocation-probe policy. |
| Real execution disagrees with the plan | Resolve a traced step and use the [Step diagnostics guide](step-diagnostics.md). |
