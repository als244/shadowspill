# Planning diagnostic fields

`Plan.diagnostics` contains a `PlanningDiagnostics` tree describing the search.
The [Plan guide](plan-report.md) covers the artifact, its summary and physical
layout; [step diagnostics](step-diagnostics.md) covers measured execution.

```text
PlanningDiagnostics
├── work                                      PlanningWorkDiagnostics
└── resolved_programs[]                       ResolvedProgramDiagnostics
    ├── choices[]                             TaskAlternativeChoiceDiagnostic
    ├── incumbent                             IncumbentDiagnostic
    ├── work                                  PlanningWorkDiagnostics
    └── candidate_evaluations[]                CandidateDiagnostic
        ├── repairs                           PlanningRepairDiagnostics
        └── steps[]                           ReductionStep
```

Durations end in `_ns` or `_seconds`; `started_ns` and `finished_ns` are
wall-clock instants. Byte counts, stable IDs and digests retain explicit suffixes.
The tables below describe stored fields and useful derived properties.

## PlanningDiagnostics

One search: its problems, the policies it evaluated, and the totals. Reachable
as `diagnostics.search_runs[...]`, on an admission attempt, and on a
`ProgramPlanResult`. The record has two halves. Which plan won, at what cost,
and which search found it is what any search must say; the rest --
`resolved_programs`, `work`, and the capacity it settled on -- is that search's
own report, written under the name in `search`, so a reader that does not
recognise the name reads the generic half and stops.

| Field | Meaning |
|---|---|
| `selected_candidate_id` | The candidate policy the search answered with. |
| `selected_selection_id` | The resolved program that answer came from. |
| `selected_makespan_ns` | The admission-aware makespan of that answer. |
| `resolved_programs` | Every resolved program this search evaluated. |
| `work` | Exact operation counts and where the time went. |
| `effective_object_capacity_bytes` | The capacity it finally planned against. |
| `search` | Which search produced this, as that search names itself. The key its own half is written under. |
| `workers` | How many threads it was given. Recorded for visibility and excluded from anything compared across runs, because it changes how long an answer took and not which answer was right. |

Derived on access, not stored: `resolved_program_count`,
`valid_resolved_program_count`, `candidate_policy_count`,
`candidate_evaluation_count`, `valid_candidate_evaluation_count`,
`candidate_status_counts`, and `repairs`, the run's repairs summed over every
candidate.

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
| `fetched_bytes`, `evicted_bytes` | What this problem's own best plan moves, summed over the FETCH and EVICT actions of its selected schedule. The winner's traffic is also on `PlanSummary`; these are the alternatives', which is what says whether a problem that asks for less compute pays for it on the lanes instead. Zero when it placed nothing. |
| `incumbent` | What became of the plan to beat this problem was handed, or `None` when it was handed none. When that plan won, `selected_candidate_id` is `incumbent`. |

Derived on access, not stored: `candidate_policy_count`,
`valid_candidate_policy_count`, `candidate_status_counts`, and `repairs`, this
problem's repairs summed over its candidates.

## IncumbentDiagnostic

What became of the plan to beat, measured at the problem's capacity before any
candidate ran.

| Field | Meaning |
|---|---|
| `status` | `valid` when it simulated and its layout fit the pool; `unplaceable` when the layout did not; `infeasible` when it did not simulate or admit here; `error` when the library could not measure it. |
| `makespan_ns` | What it simulated to at this capacity. |
| `required_bytes` | Pool bytes its layout needed, when it was measured against a pool. |
| `selected` | Whether it is the problem's answer. |
| `schedule_digest` | Which plan it is. |
| `found_by` | The candidate policy that first found it, read through any chain of hand-offs: a plan handed on from a search that answered with its own plan in hand names the candidate that found that one. |
| `found_at_capacity_bytes` | The object capacity it was first found at. |

## TaskAlternativeChoiceDiagnostic

| Field | Meaning |
|---|---|
| `group_id` | The task-alternative group. |
| `option_id` | The option chosen for it. |

## CandidateDiagnostic

One candidate policy evaluated in one problem. `candidate_id` identifies only
the reusable policy: residency strategy and fetch rule. Scoring and ranking
are fixed by the algorithm: `E, writeback, -slack, -size`, followed by stable
identity tie breaks. Transfer strategies omit `E`.

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
| `pressure_escalations`, `escalations_taken_back` | Pressure repairs that asked for more than the shortfall because the same failure had repeated at the same task and moment, and how many of those asks no cut could meet and were taken back. |
| `best_unplaced_makespan_ns`, `unplaced_plans` | The fastest plan it simulated whose layout did not fit the pool, and how many such plans it measured; `None` and 0 when none. Beside `makespan_ns` they say what placing cost. |
| `started_ns`, `finished_ns` | When this candidate ran, from the start of the call. Two candidates ran at the same time exactly when their spans overlap. |
| `schedule_digest` | Identity of the schedule it produced. |
| `failure_kind`, `failure_detail` | Why it failed, when it did. |
| `repairs` | Its repairs, by category. |
| `work` | Its operation counts and section times. |
| `steps` | Every plan it held, in order. Empty unless a trajectory was asked for. |
| `residency_strategy`, `fetch_rule` | The policy, parsed back out of `candidate_id`. |

## PlanningWorkDiagnostics

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

`simulation_requests` is derived: calls plus cache hits.

## PlanningSectionTiming

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

`named_ns` is derived: what every named section claimed, nested ones excluded.
`total_ns` equals it plus `residual_ns` at every level.

## PlanningRepairDiagnostics

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
| `layout_fetch_delay_attempts` | Its layout overran the pool; a fetch was moved later so it could fit. |

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
| `simulated`, `measured`, `placed`, `refined`, `best`, `answer`, `moved` | What became of it, in flag order: simulated at all, measured for a layout, placed, refined, best so far, the one the candidate answered with, and a fetch moved later after it so its layout could fit. |

## ProgramPlanResult

The internal planner's selected logical schedule and simulator evidence,
available as `AnnotatedProgramPlan.result`.

| Field | Meaning |
|---|---|
| `program` | The program the schedule is for. |
| `search_options` | Everything the search was told: the generic options, the algorithm that ran, and that algorithm's own options. `workers` is absent from the archived form, since two runs at different worker counts ask the same question. |
| `initial_residency`, `final_residency` | Where each object sits when the step begins and ends. |
| `simulation_config` | The device and capacity model the simulation ran against. |
| `schedule` | The memory actions themselves. |
| `selections` | The task-alternative choice per group. |
| `simulation` | The simulator's evidence. Not persisted: a stored plan replays it and cross-checks the makespan, so a stale plan is caught rather than trusted. |
| `diagnostics` | The search evidence, as `PlanningDiagnostics`. |
| `resident_slice` | The slice reserved for objects kept resident. |
| `admission_facts` | The capacity facts the plan was admitted against. |
| `placement_facts` | The placement topology it was measured under. |

## PlanSummary

The internal `AnnotatedProgramPlan.summary` record. The public phase
`Plan.summary` is a separate mapping described in the [Plan guide](plan-report.md).
The five time components add up: `simulated_step_seconds` equals `unconstrained_step_seconds`
plus `recomputation_overhead_seconds` plus `task_contention_seconds` plus `idle_seconds` plus
`terminal_writeback_seconds`.

| Field | Meaning |
|---|---|
| `simulated_step_seconds` | The simulated makespan of one step. |
| `unconstrained_step_seconds` | The compute floor: every task-alternative group at its cheapest option, and no waiting. |
| `recomputation_overhead_seconds` | What the chosen options cost above that floor. |
| `task_contention_seconds` | Time added inside task intervals above isolated profiles, from compute/transfer interference or waiting for shared resources. |
| `idle_seconds` | Time outside task intervals before the last task finishes, including initial and inter-task readiness waits. |
| `terminal_writeback_seconds` | The tail after the last task, writing back what the step produced. |
| `recomputing_group_count` | Groups whose chosen option costs strictly more compute than that group's cheapest. |
| `task_alternative_group_count` | Task-alternative groups in the program, forced ones included. |
| `flexible_group_count` | Groups that are a real decision, which is what a resolution share is taken of and so the denominator `recomputing_group_fraction` uses. A group is forced instead when structure leaves one option -- a sink of the forward phase has to keep its value -- or when its options retain the same bytes, which is one plan spelled twice. |
| `spill_peak_bytes` | The most the spill pool ever holds while the step runs, as this plan's simulation reached it. Beside the spill budget it says whether the budget shaped the plan or merely bounded it. |
| `transfer_bytes_fetched`, `transfer_bytes_evicted` | Traffic the simulation ran, summed from its transfer intervals. |
| `fetch_solo_bandwidth_bytes_per_second`, `fetch_concurrent_bandwidth_bytes_per_second`, `evict_solo_bandwidth_bytes_per_second`, `evict_concurrent_bandwidth_bytes_per_second` | Coarsened per-direction rates used by the simulator. Payload uses the solo rate while the opposite lane is idle, and the concurrent rate while both move bytes. Raw measurements remain on the transfer profiles. |
| `fetch_busy_ns`, `evict_busy_ns` | Sum of simulated transfer durations on each lane, excluding per-transfer latency. Utilization uses these intervals, not bytes divided by a fixed bandwidth. |
| `fetch_latency_ns`, `evict_latency_ns` | Coarsened calibration latency retained with the plan for diagnostics and provenance. The simulator does not use these values for transfer timing. |
| `planning_phase_seconds` | Wall time of each recorded planning phase, in seconds; empty when no phase timings were supplied. |
| `selected_candidate` | The candidate whose plan was selected: residency strategy, fetch rule, repairs at best, the fastest plan it could not place (`best_unplaced_makespan_ns`, `unplaced_plans`) and `placement_gap`, the answer over that plan (`None` when every measured plan fit). When the search answered with the plan it was handed, `incumbent` with that plan's outcome instead. |

`device_peak_bytes` is the simulated peak of logical device allocations,
including task workspace. `physical_execution_peak_bytes` adds the fixed runtime
costs to the certified slab requirement on an `AnnotatedProgramPlan`; it is
`None` where no physical certificate is available. This is the required execution
footprint, not the whole pool reserved by the process. `spill_peak_bytes` is the
simulated live spill footprint, distinct from the host pool held at runtime.

`recomputing_group_fraction` is derived from the two counts; the JSON field
`recomputing_group_percent` multiplies it by 100. Its denominator is flexible
graph-pair groups, not FLOPs or time. `idle_percent`, `fetch_utilization_percent`
and `evict_utilization_percent` divide their respective simulated idle/busy times
by the estimated step time. Transfer lanes can overlap compute and one another,
so these percentages do not sum to 100. Blended transfer throughput divides bytes
by lane busy time and is `None` when the lane is unused. All these quantities are
estimates from profiles and simulation; serialized summaries mark `estimate=true`.
Each returned `AnnotatedProgramPlan.summary` and its JSON `summary` include them.
