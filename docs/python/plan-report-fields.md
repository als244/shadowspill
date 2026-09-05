# PlanReport field reference

Every field of every record a `PlanReport` carries, with what it holds. The
narrative guide is [Interpreting a PlanReport](plan-report.md); this page is
the lookup table behind it, and it is checked against the source so that a
field added without an entry here fails the documentation tests. The
execution-side equivalent is [step diagnostics](step-diagnostics.md), which
is a separate document because it describes a different object measured on
different clocks.

## Field names

| Suffix | Kind | Example |
|---|---|---|
| `_at_seconds`, `_timestamp_nanoseconds` | An instant | `calibrated_timestamp_nanoseconds` |
| `_seconds`, `_ns`, `_nanoseconds` | A duration | `duration_ns` |
| `_bytes` | A size, or an offset where the name says so | `required_bytes` |
| `_count`, plural nouns | A count | `task_alternative_group_count` |
| `_digest`, `_id`, `_key` | An identity | `schedule_digest` |

No record stores a value that is the difference of two of its own fields.
Where such a number is useful it is a subtraction at the point of use, so
there is one place to read each measurement. The three that used to be stored
are named where they were removed.

Two names carry a trap worth stating. `semantic_contract_capture_ns` and
`executable_contract_capture_ns` are durations, not instants: they are how
long capturing the contract took. `started_ns` and `finished_ns` on the
PressureFit records are the opposite, instants on the clock of the call that
evaluated them.

## What hangs off what

```text
PlanReport
├── execution_plan / initial_execution_plan   ExecutionPlan (IR)
├── task_profiles[]                           TaskProfile (IR)
├── transfer_actions[]                        MemoryAction (IR)
├── transfer_capabilities                     TransferCapabilities → TransferProfile[]
├── pressurefit_results[]                     PressureFitResult → ResidentSlice
├── summary            (property)             PlanSummary
└── diagnostics                               PlanDiagnostics
    ├── phases[]                              PlanPhaseTiming
    ├── compiler_profiles[]                   PlanCompilerProfile → PlanPhaseTiming[]
    ├── cache_artifacts[]                     PlanCacheArtifact
    ├── profiling_metadata[]                  PlanProfilingMetadata
    ├── task_stage_map[] / tasks[]            PlanTaskStage
    ├── unique_stages[]                       PlanUniqueStage → PlanGraphPair → PlanGraphProfile
    ├── physical_layouts[]                    PlanPhysicalLayout
    │   ├── attempts[]                        PlanFixedLayoutAttempt → PressureFitDiagnostics
    │   └── task_memory_envelopes[]           PlanTaskMemoryEnvelope
    └── pressurefit_runs[]                    PressureFitDiagnostics
        └── resolved_programs[]          ResolvedProgramDiagnostics
            ├── choices[]                     TaskAlternativeChoiceDiagnostic
            ├── work                          PressureFitWorkDiagnostics → PressureFitSectionTiming
            └── candidate_evaluations[]       CandidateDiagnostic
                ├── repairs                   PressureFitRepairDiagnostics
                └── steps[]                   ReductionStep
```

## PlanReport

What one planning call produced. `mode` is `forward` or `training`.

| Field | Meaning |
|---|---|
| `mode` | Which planning entry point produced this report. |
| `capture_identity` | Digest over mode, signature, artifacts, and profiling metadata: the identity of what was captured. |
| `execution_plan` | The recurrent step's plan. `program` reads through it. |
| `initial_execution_plan` | The first step's plan when it differs, as it does for a lazily initialized optimizer; `None` otherwise. |
| `task_profiles` | The isolated profile behind every task the plan prices. |
| `transfer_actions` | The schedule's memory actions. Named for the transfers but carries releases too, which move no bytes. |
| `transfer_bytes_evicted`, `transfer_bytes_fetched` | Bytes the schedule's evictions and fetches move. Distinct from the identically named fields on `summary`, which count what the simulation ran rather than what the schedule asked for. |
| `profile_unique_keys`, `profile_cache_hits`, `profile_cache_misses` | Distinct task profiles needed, and how many came from the store. |
| `allocation_probe_seeds`, `allocation_probe_repetitions` | How the allocation probe was run when profiling. |
| `profiling_provenance` | Deduplicated labels saying where each profile came from. |
| `phase_timings_ns` | Name and elapsed time for each planning phase, ending with `total` for the whole call. |
| `diagnostics` | Everything below, under `PlanDiagnostics`. |
| `execution_pool`, `spill_pool` | The pools this plan was built for. |
| `execution_budget_bytes`, `spill_budget_bytes` | The budgets it was given. |
| `requested_dynamic_scratch_reserve_bytes` | Scratch the caller asked to keep outside the fixed layout. |
| `execution_device` | Device ordinal of the execution pool. |
| `transfer_capabilities` | The measured transfer matrix the simulator planned against. |
| `optimizer_ordering` | How optimizer work was ordered, or `None` for a forward plan. |
| `planned_program_cache_hits`, `planned_program_cache_misses` | Whether the selected plan was read back from the store. Exactly one is 1. |
| `fixed_slab_bytes` | The slab the fixed layout occupies. |
| `captured_stage_count` | Stages the capture produced. |
| `aot_unique_stage_contracts` | Distinct structural contracts among them. |
| `aot_graph_pair_cache_hits`, `aot_graph_pair_cache_misses` | Graph pairs served from the store rather than compiled. |
| `pressurefit_results` | The selected plans, first-step first and recurrent last. |

Derived on access, not stored: `program`, `initial_program`,
`pressurefit_result`, `initial_pressurefit_result`,
`predicted_device_peak_bytes`, `predicted_spill_peak_bytes`,
`predicted_makespan_ns`, `summary`, `shared_aliases`,
`shared_execution_bytes`, `shared_spill_bytes`,
`callable_execution_budget_bytes`, `callable_spill_budget_bytes`,
`fetch_profile`, `evict_profile`.

## PlanSummary

`report.summary`. What the selected plan promises, in one place. The four
parts add up: `simulated_step_seconds` equals `unconstrained_step_seconds`
plus `recomputation_overhead_seconds` plus `idle_seconds` plus
`terminal_writeback_seconds`.

| Field | Meaning |
|---|---|
| `simulated_step_seconds` | The simulated makespan of one step. |
| `unconstrained_step_seconds` | The compute floor: every task-alternative group at its cheapest option, and no waiting. |
| `recomputation_overhead_seconds` | What the chosen options cost above that floor. |
| `idle_seconds` | Time the step spends waiting rather than computing. |
| `terminal_writeback_seconds` | The tail after the last task, writing back what the step produced. |
| `recomputing_group_count` | Groups whose chosen option costs strictly more compute than that group's cheapest. |
| `task_alternative_group_count` | Task-alternative groups in the program. |
| `transfer_bytes_fetched`, `transfer_bytes_evicted` | Traffic the simulation ran, summed from its transfer intervals. |
| `fetch_bandwidth_bytes_per_second`, `evict_bandwidth_bytes_per_second` | The per-direction bandwidth the simulator planned against. Solo calibration lives on the report's transfer profiles. |
| `planning_phase_seconds` | Each planning phase's wall time in phase order, ending with `total`. A view over the report's `phase_timings_ns`, which stays the stored record. |
| `selected_candidate` | The candidate whose plan was selected: residency strategy, fetch rule, coalescing, and repairs at best. |

`recomputing_group_fraction` is derived from the two counts.

## PlanDiagnostics

`report.diagnostics`. Phase intervals are mutually exclusive, and
`measured_wall_time_ns` adds them up. The difference from
`total_wall_time_ns` is the remainder spent between measured intervals and
constructing the report; it is a subtraction rather than a field, so the two
can never disagree.

| Field | Meaning |
|---|---|
| `phases` | The mutually exclusive planning intervals. |
| `total_wall_time_ns` | The whole planning call. |
| `profile_unique_keys`, `profile_cache_hits`, `profile_cache_misses` | Task profiles needed and served from the store. |
| `allocation_probe_seeds`, `allocation_probe_repetitions` | How the allocation probe was run. |
| `captured_stage_count` | Stages the capture produced. |
| `aot_unique_stage_contracts` | Distinct structural contracts among them. |
| `aot_graph_pair_cache_hits`, `aot_graph_pair_cache_misses` | Graph pairs served from the store. |
| `planned_program_cache_hits`, `planned_program_cache_misses` | Whether the plan was read back rather than searched. |
| `task_stage_map` | Every task variant considered, selected or not. |
| `unique_stages` | The deduplicated stages and their legal choices. |
| `compiler_phase_timings_ns` | Compiler phases across the whole call, as name and elapsed pairs. |
| `compiler_profiles` | The same, per structural contract. |
| `store_directories` | Which artifact store directory each category used. |
| `cache_artifacts` | Every persistent artifact this call touched. |
| `profiling_metadata` | Canonical planning-only workload metadata per input position. |
| `pressurefit_runs` | One `PressureFitDiagnostics` per search this call ran. |
| `physical_layouts` | One admission summary per execution phase. |

## PlanPhaseTiming

| Field | Meaning |
|---|---|
| `name` | The phase label. |
| `duration_ns` | How long it took. `duration_seconds` is the same number in seconds. |

## PlanCompilerProfile

| Field | Meaning |
|---|---|
| `structural_contract_key` | The contract these phases compiled. |
| `phases` | Its non-overlapping compiler phases. `total_wall_time_ns` sums them. |

## PlanCacheArtifact

One persistent artifact this call touched. `access` separates bytes actually
read or written from an artifact that merely matched a freshly produced
in-memory result; Inductor's private directory is reported as `managed`.

| Field | Meaning |
|---|---|
| `category` | Which store directory it lives in. |
| `kind` | What kind of document it is. |
| `digest` | Its content digest, or `None` for a directory. |
| `path` | Where it is on disk. |
| `access` | Read, written, matched, or managed. |
| `schema` | The document's own schema, when it declares one. |
| `dependencies` | Digests this artifact was derived from. |

## PlanProfilingMetadata

| Field | Meaning |
|---|---|
| `position` | The input position it describes. |
| `digest` | Digest of the canonical form. |
| `canonical_json` | The canonical, content-free description of that input. |

## PlanTaskStage

One task's identity across every naming scheme, and which variant the search
chose. Reachable as `diagnostics.tasks[execution_task_id]` for the selected
tasks and `task_stage_map` for all of them.

| Field | Meaning |
|---|---|
| `task_id` | The Program's own task identity. |
| `execution_ordinal` | Chronological position in the step, or `None` when not selected. |
| `execution_task_id` | The chronological identity, `execution_NNNNNN`, shared with step diagnostics. |
| `semantic_name` | The module path the stage came from. |
| `phase` | Forward, backward, or optimizer. |
| `microbatch` | Which accumulation round, when the task belongs to one. |
| `stage_occurrence_id` | This occurrence of the stage in the capture. |
| `unique_stage_id` | The deduplicated stage it is an occurrence of. |
| `structural_contract_key` | The structural contract it compiles under. |
| `semantic_contract_digest`, `executable_contract_digest`, `compiled_layout_digest` | The three contract layers, from what the model means to what the compiler laid out. |
| `graph_pair_variant` | This record's variant. |
| `chosen_graph_pair_variant` | The variant the search chose for the stage. |
| `selected` | Whether this task is in the plan. Not the same statement as the two variants matching. |
| `profile_compatibility_digest` | Which profile may be reused for it. |
| `profiling_metadata_digest` | The workload metadata it was profiled under. |

## PlanUniqueStage

| Field | Meaning |
|---|---|
| `unique_stage_id` | The stage's identity. |
| `structural_key` | Its structural contract. |
| `module_targets` | The module paths that map to it. |
| `occurrence_count` | How many tasks are occurrences of it. |
| `graph_pairs` | Every legal choice for it. |

## PlanGraphPair

One legal stage choice. Forward-only choices omit `backward`.

| Field | Meaning |
|---|---|
| `variant` | The choice's name. |
| `memory_budget` | The budget the partitioner was given for this variant. |
| `recomputation` | Whether it recomputes rather than saving. |
| `saved_value_count` | Values it saves across the boundary. |
| `specialized_unit_tangent_count` | Saved values specialized to a unit tangent. |
| `saved_input_root_count`, `saved_boundary_root_count`, `saved_internal_root_count` | Saved roots by where they come from. |
| `saved_input_minimum_bytes`, `saved_boundary_minimum_bytes`, `saved_internal_minimum_bytes` | The bytes those roots need at minimum. |
| `forward`, `backward` | The measured profile of each half. |

## PlanGraphProfile

Measured cost and memory geometry for one executable graph contract.

| Field | Meaning |
|---|---|
| `direction` | Forward or backward. |
| `structural_contract_key` | The contract this profile is for. |
| `semantic_contract_digest` | Digest of what the model means. |
| `semantic_contract_capture_ns` | How long capturing it took. |
| `semantic_roots`, `semantic_output_views`, `semantic_mutations` | Its storage roots, returned views, and in-place mutations. |
| `executable_contract_digest` | Digest of the traced, executable form. |
| `executable_contract_capture_ns` | How long capturing that took. |
| `executable_roots`, `executable_output_views`, `executable_mutations` | The same three, after tracing. |
| `compiled_layout_digest` | Digest of what the compiler actually laid out. |
| `compiled_roots`, `compiled_output_views` | The observed physical allocation and binding for each. |
| `physical_profile_wall_time_ns` | How long the profiling harness spent measuring. |
| `representative_task_id` | The task whose shapes were profiled. |
| `runtime_ns` | The measured task runtime the plan prices. |
| `samples_ns` | The individual timing samples behind it. |
| `provenance` | Where the measurement came from. |
| `representative_inputs` | Content-free provenance for each profiled input. |
| `profile_phase_timings_ns` | Where the profiling time went, as name and elapsed pairs. |
| `timing_relative_mad` | Relative median absolute deviation across samples. |
| `timing_half_drift` | Drift between the first and second halves of the samples. |
| `timing_unstable` | Whether those two put the measurement outside tolerance. |
| `inputs`, `mutations`, `outputs` | One footprint per object in each role. |
| `input_logical_bytes`, `mutation_logical_bytes`, `output_logical_bytes` | The views' own bytes in each role. |
| `input_allocation_bytes`, `mutation_allocation_bytes`, `output_allocation_bytes` | The allocator extents behind them, counted once per alias group. |
| `workspace_requested_bytes`, `workspace_charged_bytes` | Workspace the graph asked for, and what the pool charged. |
| `replacement_transition_bytes` | Bytes live across an in-place replacement. |
| `task_workspace_bytes` | The workspace the task profile records. |
| `workspace_extent_bytes`, `persistent_extent_bytes` | The extents behind workspace and persistent allocations. |
| `allocation_contract_digest` | Identity of the pointer-free allocation contract. |
| `allocation_contract` | Its steps, which the runtime replays. |
| `allocation_timeline` | Every allocation and free in the graph's local order. |

## PlanStorageRoot

| Field | Meaning |
|---|---|
| `root_id` | The root's identity within the graph. |
| `kind` | What kind of root it is. |
| `source_input` | The input position it came from, when it is an input. |
| `producer_node`, `producer_target`, `producer_result` | The node that produced it, its operator, and which result. |
| `minimum_span_bytes` | The smallest allocation that can hold it. |

## PlanOutputView

| Field | Meaning |
|---|---|
| `leaf_index` | Which returned tensor leaf this is. |
| `root_id` | The root it views. |
| `offset_bytes` | Where the view starts inside that root. |
| `span_bytes` | How far it reaches. |
| `shape`, `stride` | Its geometry, strides in elements. |
| `dtype`, `layout` | Its element type and tensor layout. |

## PlanMutationBinding

| Field | Meaning |
|---|---|
| `input_position` | The input the graph mutates. |
| `replacement_output_leaf` | The output leaf that replaces it, when there is one. |
| `producer_node`, `producer_target` | The node performing the mutation and its operator. |
| `argument_name` | The argument it binds to. |

## PlanCompiledRoot

| Field | Meaning |
|---|---|
| `root_id` | The semantic root this allocation serves. |
| `allocation_ordinal` | Which allocation in the contract it is. |
| `requested_bytes` | What the compiler asked for. |
| `charged_bytes` | The aligned range the pool charged. |

## PlanCompiledOutputView

| Field | Meaning |
|---|---|
| `leaf_index` | Which returned tensor leaf this is. |
| `root_id` | The root it views. |
| `allocation_ordinal` | The allocation behind that root. |
| `offset_bytes` | Where the leaf starts inside it. |

## PlanRepresentativeInput

Content-free value provenance for one independently profiled input.

| Field | Meaning |
|---|---|
| `position` | The input position. |
| `role` | What the input is to the graph. |
| `source` | Where its value came from, when it has a source. |
| `value_policy` | How a stand-in value was produced for profiling. |
| `dtype`, `shape`, `stride` | Its element type and geometry, strides in elements. |
| `storage_offset` | Its offset into its storage, in elements rather than bytes. |
| `alias_group` | The alias group it belongs to. |
| `consumer_targets` | The operators that consume it. |

## PlanObjectFootprint

| Field | Meaning |
|---|---|
| `object_id` | The object this footprint is for. |
| `alias_group_id` | The alias group it shares an allocation with. |
| `role` | Input, mutation, or output. |
| `logical_size_bytes` | The view's own bytes. |
| `allocation_size_bytes` | The allocator extent containing it. |
| `offset_bytes` | Where the view sits inside that extent. |

## PlanAllocationABIStep

One pointer-free allocator operation a compiled task requires.

| Field | Meaning |
|---|---|
| `operation_index` | Position in the contract. |
| `allocation_ordinal` | Which allocation the operation concerns. |
| `operation` | Allocate or free. |
| `requested_bytes` | What the graph asks for. |
| `charged_bytes` | The aligned range the pool charges. |
| `alignment_bytes` | The alignment that produced it. |
| `output_leaf_indices` | Returned leaves this allocation backs. |
| `mutation_input_positions` | Mutated inputs it backs. |
| `persistent_after_task` | Whether it outlives the task. |

## PlanAllocationEvent

One allocation or free in the graph's local order. The order is
`allocation_ordinal`; nothing here is a time.

| Field | Meaning |
|---|---|
| `allocation_ordinal` | The allocation this event concerns. |
| `operation` | Allocate or free. |
| `requested_bytes`, `charged_bytes` | Asked and charged. |
| `output_leaf_indices` | Returned leaves it backs. |
| `output_view_offsets` | Each leaf's byte offset into it, in the same order. |
| `reuses_ordinal` | The earlier allocation whose range this one takes. |

## PlanPhysicalLayout

Complete fixed-layout admission summary for one execution phase. The slack is
`pool_capacity_bytes` minus `required_bytes`, and the capacity the refinement
gave back is `original_object_capacity_bytes` minus
`effective_object_capacity_bytes`; both are subtractions rather than fields.

| Field | Meaning |
|---|---|
| `plan_role` | Which plan this layout admitted, first-step or recurrent. |
| `strategy` | How the layout was built. |
| `layout_digest`, `program_digest`, `schedule_digest`, `facts_digest` | The identities this admission is a function of. |
| `pool_capacity_bytes` | The pool it had to fit inside. |
| `original_object_capacity_bytes` | The object capacity the search started against. |
| `effective_object_capacity_bytes` | What it planned against after refinement. |
| `fixed_slice_bytes` | The slice the fixed placements occupy. |
| `resident_slice_bytes` | The part of it reserved for objects kept resident. |
| `dynamic_reserve_bytes` | Reserved for allocations the layout does not fix. |
| `scratch_reserve_bytes` | Reserved for caller scratch. |
| `required_bytes` | What the layout spans in total. |
| `placement_count` | Fixed placements it holds. |
| `dynamic_lifetime_count` | Lifetimes left dynamic. |
| `reuse_dependency_count` | Range-reuse edges between them. |
| `placements_by_purpose` | Those placements grouped by what they are for. |
| `attempts` | Every capacity trial made on the way here. |
| `task_memory_envelopes` | The admitted allocator limits, per selected task. |

## PlanFixedLayoutAttempt

| Field | Meaning |
|---|---|
| `requested_object_capacity_bytes` | The capacity this trial asked the search for. |
| `effective_object_capacity_bytes` | What it planned against. |
| `required_bytes` | What the resulting layout spans. |
| `pool_capacity_bytes` | The pool it had to fit inside. |
| `accepted` | Whether this trial's layout is the one admitted. |
| `pressurefit_wall_time_ns` | Search and cache-resolution time, cumulative across refinements rather than this trial alone. |
| `physical_admission_wall_time_ns` | Time spent admitting the layout. |
| `pressurefit_diagnostics` | The search evidence behind it, when it was recorded. |

## PlanTaskMemoryEnvelope

Fail-closed allocator limits admitted for one selected task. The `maximum_`
fields are the largest single allocation observed; the `_limit_` fields are
the ceilings the runtime enforces on what is live at once.

| Field | Meaning |
|---|---|
| `task_id` | The task these limits bind. |
| `maximum_requested_allocation_bytes`, `maximum_charged_allocation_bytes` | Largest single allocation, as asked and as charged. |
| `live_requested_allocation_limit_bytes`, `live_charged_allocation_limit_bytes` | Ceilings on what the task may hold live. |
| `dynamic_scratch_maximum_allocation_bytes`, `dynamic_scratch_live_limit_bytes` | The same two for scratch outside the fixed layout. |
| `allocation_contract_digest` | The contract the task replays. |
| `allocation_contract_operation_count` | How many operations it has. |
| `allocation_path_digests` | The allocation paths admitted for it. |

## PressureFitDiagnostics

One PressureFit search: its problems, the policies it evaluated, and the
totals. Reachable as `diagnostics.pressurefit_runs[...]`, on an admission
attempt, and on a `PressureFitResult`.

| Field | Meaning |
|---|---|
| `selected_candidate_id` | The candidate policy the search answered with. |
| `selected_selection_id` | The resolved program that answer came from. |
| `selected_makespan_ns` | The admission-aware makespan of that answer. |
| `resolved_programs` | Every resolved program this search evaluated. |
| `work` | Exact operation counts and where the time went. |
| `effective_object_capacity_bytes` | The capacity it finally planned against. |

## ResolvedProgramDiagnostics

| Field | Meaning |
|---|---|
| `selection_id` | The problem's identity. |
| `choices` | The task-alternative choice it fixes for each group. |
| `selected_candidate_id` | The policy chosen for it. |
| `selected_makespan_ns` | That policy's makespan. |
| `candidate_evaluations` | Every policy evaluated for this problem. |
| `work` | Its operation counts and section times. |
| `started_ns`, `finished_ns` | This problem's span, on the same clock its candidates use. Problems evaluated in one call overlap, because workers take whatever task is next. |
| `evict_ineligible_aliases`, `evict_ineligible_bytes` | How many objects the evict-eligibility threshold kept resident, and their bytes. |
| `fetched_bytes`, `evicted_bytes` | What this problem's own best plan moves, summed over the FETCH and EVICT actions of its selected schedule. The winner's traffic is also on `PlanSummary`; these are the alternatives', which is what says whether a problem that asks for less compute pays for it on the lanes instead. Zero when it placed nothing, and on a plan read back from a store written before these were recorded. |

## TaskAlternativeChoiceDiagnostic

| Field | Meaning |
|---|---|
| `group_id` | The task-alternative group. |
| `option_id` | The option chosen for it. |

## CandidateDiagnostic

One candidate policy evaluated in one problem. `candidate_id` identifies only
the reusable policy: residency strategy, fetch rule, and coalescing mode.

| Field | Meaning |
|---|---|
| `candidate_id` | The policy's identity. |
| `selection_id` | The problem it was evaluated for. |
| `status` | Valid, or how it failed. |
| `makespan_ns` | The makespan it reached, when it reached one. |
| `capacity_violation_count` | Places the accepted plan came up short of capacity and waited for room. Zero means it never waited for memory. |
| `placements_attempted`, `placements_admitted` | Layouts measured for it, and how many fitted. |
| `capacity_refinements` | How many times a plan gave back what it overran and was rebuilt. |
| `repairs_at_best` | Repairs spent when the plan it answers with was placed; `None` when it placed none. |
| `started_ns`, `finished_ns` | When this candidate ran, from the start of the call. Two candidates ran at the same time exactly when their spans overlap. |
| `schedule_digest` | Identity of the schedule it produced. |
| `failure_kind`, `failure_detail` | Why it failed, when it did. |
| `repairs` | Its repairs, by category. |
| `work` | Its operation counts and section times. |
| `steps` | Every plan it held, in order. Empty unless a trajectory was asked for. |
| `residency_strategy`, `fetch_rule`, `coalesced` | The policy, parsed back out of `candidate_id`. |

## PressureFitWorkDiagnostics

Exact search operations, and the sections the time went to. Invocation-level
values include work done before or across candidates, so they need not equal
the sum of the candidates'.

| Field | Meaning |
|---|---|
| `schedule_emissions` | Schedules emitted. |
| `schedule_cache_hits` | Emissions served from the cache instead. |
| `simulation_calls` | Simulations run. |
| `simulation_cache_hits` | Simulations served from the cache. |
| `admission_calls` | Admission checks run. |
| `sections` | Where the time went. |

## PressureFitSectionTiming

Disjoint spans of one planning step, as its orchestrator measured them.
Exactly one section is open at a time. `admit_ns` is nested inside
`simulate_ns` and stands outside the sum, so adding every field double-counts
admission.

| Field | Meaning |
|---|---|
| `total_ns` | The whole span the orchestrator covered. |
| `prepare_ns` | Deriving the residency problem from the program. Problem level only. |
| `setup_ns` | Schedule facts and the candidate workspace. |
| `reduce_ns` | Choosing what stays resident, before any candidate repairs it. |
| `emit_ns` | Turning residency gaps into an ordered schedule. |
| `simulate_ns` | Replaying the schedule for a makespan. |
| `repair_ns` | Moving a transfer or making room for one, and reducing again when that is what it took. |
| `digest_ns` | Naming the schedule. |
| `place_ns` | Measuring whether the plan has a layout that fits. |
| `select_ns` | Deciding what to answer with, and materializing it. |
| `teardown_ns` | Releasing everything the evaluation held. |
| `admit_ns` | Admitting the schedule into the pool, inside `simulate_ns`. |
| `residual_ns` | `total_ns` less every named section above. |

## PressureFitRepairDiagnostics

Categorized monotonic changes made while repairing one search path. Each
category names what refused the plan and what the repair did about it.

| Field | Meaning |
|---|---|
| `unclassified_attempts` | Repairs the search could not attribute to a category. |
| `admission_fetch_advance_attempts` | Admission refused the plan; a fetch was moved earlier. |
| `admission_fetch_delay_attempts` | Admission refused it; a fetch was moved later. |
| `admission_pressure_boundary_attempts` | Admission refused it; room was made at a pressure boundary. |
| `simulation_fetch_delay_attempts` | Simulation refused it; a fetch was moved later. |
| `simulation_pressure_boundary_attempts` | Simulation refused it; room was made at a pressure boundary. |

`total_attempts` and `pressure_boundary_attempts` are sums over these.

## ReductionStep

One plan a candidate held, and what became of it. A candidate reaches its
answer by holding a succession of plans, and the steps in order are the
search itself. Recorded only when the caller asks for a trajectory.

| Field | Meaning |
|---|---|
| `makespan_ns` | The makespan this plan simulated to. |
| `required_bytes` | Bytes its layout spans. Zero unless it was measured. |
| `capacity_bytes` | The object capacity it was built against, which falls as the candidate hands capacity back. |
| `cut_aliases` | Objects the reducer cut to reach it, by alias index. |
| `repairs` | Repairs the candidate had made when it reached this plan. |
| `simulation_status` | What the simulator returned for it. |
| `capacity_violations` | Places it came up short of capacity and waited. |
| `simulated`, `measured`, `placed`, `refined`, `best`, `answer` | What became of it, in flag order: simulated at all, measured for a layout, placed, refined, best so far, and the one the candidate answered with. |

## PressureFitResult

`report.pressurefit_results[...]`. The selected logical schedule and the
simulator evidence behind it.

| Field | Meaning |
|---|---|
| `program` | The program the schedule is for. |
| `options` | The search options it was produced under. |
| `initial_residency`, `final_residency` | Where each object sits when the step begins and ends. |
| `simulation_config` | The device and capacity model the simulation ran against. |
| `schedule` | The memory actions themselves. |
| `selections` | The task-alternative choice per group. |
| `simulation` | The simulator's evidence. Not persisted: a stored plan replays it and cross-checks the makespan, so a stale plan is caught rather than trusted. |
| `diagnostics` | The search evidence, as `PressureFitDiagnostics`. |
| `resident_slice` | The slice reserved for objects kept resident. |
| `admission_facts` | The capacity facts the plan was admitted against. |
| `placement_facts` | The placement topology it was measured under. |

## ResidentSlice

Objects under `minimum_object_bytes_evict_eligible` are never cut, so every
lease of theirs gets a static home in a slice at the end of the fixed layout.

| Field | Meaning |
|---|---|
| `bytes` | What the planner reserved for it: the sum of those homes, sized before the search and taken out of the capacity the search plans against. |
| `aliases` | The alias groups whose leases it holds, sorted and distinct. |

An empty slice has no bytes and no aliases.

## TransferCapabilities

The measured transfer matrix one runtime generation published. `profiles` is
an n by n matrix flattened in `pool_names` order.

| Field | Meaning |
|---|---|
| `generation` | Which publication this is. |
| `pool_names` | The pools, in the order that indexes the matrix. |
| `profiles` | One profile per directed pool pair. |
| `digest` | Identity of the whole matrix. |

## TransferProfile

Measured performance for one directed pool-pair route.

| Field | Meaning |
|---|---|
| `source`, `destination` | The pools the route runs between. |
| `source_pool_id`, `destination_pool_id` | Their indices into `pool_names`. |
| `generation` | The publication this profile belongs to. |
| `latency_nanoseconds` | Fixed cost of a copy on this route. |
| `bandwidth_bytes_per_second` | The rate the simulator plans against. |
| `solo_bandwidth_bytes_per_second` | The rate measured with no other route running. |
| `concurrent_bandwidth_bytes_per_second` | The rate measured with the opposite route running. |
| `solo_measurement_nanoseconds`, `concurrent_measurement_nanoseconds` | How long each of those measurements took. |
| `calibrated_timestamp_nanoseconds` | When the route was calibrated. |
| `small_copy_bytes`, `large_copy_bytes` | The two copy sizes the calibration used. |
| `measured_copies` | How many copies it timed. |
| `available` | Whether the route exists at all. |
| `calibrated` | Whether these numbers are measured rather than assumed. |
| `provenance` | Where they came from. |
| `calibration_mode` | How the calibration was run. |
| `concurrent_route_count` | How many routes ran during the concurrent measurement. |
