# PressureFit C API

Include `<shadowspill/pressurefit/pressurefit.h>`. PressureFit is the search
that ships with ShadowSpill: given a `ShadowSpillIndexedProblem` it reduces
residency, emits the memory actions that follow from what it cut, times the
result on the simulator, and measures it against exact schedule admission and
fixed-offset placement — over as many resolved programs as the caller hands it
at once.

Everything on this page is PressureFit's own, which is why it all carries
`PressureFit` in its C names. The problem it is given, the schedule it returns
and the certification that schedule passes are generic and documented in the
[planner C API](planner.md). The residency problem its reducer works on and the
resolved problem one candidate is placed against are derived inside the search
and do not cross the library boundary, so a caller never builds either.

The framework-neutral problem formulation and complete algorithm are in the
[PressureFit architecture page](../architecture/pressurefit.md).

## Data model

`ShadowSpillPressureFitOptions` selects residency strategies, fetch rules,
coalescing modes, repair limit, initial placement, how much capacity a plan
gives back at a time when its layout does not fit
(`capacity_refinement_bytes`, zero for the whole shortfall), whether each
candidate records its reduction trajectory (`record_reduction_steps`), how many
threads the call searches on (`workers`, zero for one per logical CPU and one
for the calling thread), whether the placement gate consults the shared record
or only the candidate's own placed plans (`deterministic`), whether a plan that
has simulated may split an eviction into a write-back where the evict lane is
idle and a release where the eviction was (`split_write_backs`, kept only if
the split simulated faster), the shared best-placed record to measure against,
and which objects are too small to be worth cutting
(`minimum_object_bytes_evict_eligible`, 1 MiB by default through the Python
request, zero for none): those stay resident from first to last access, take
static homes in a resident slice whose size is reserved at preparation out of
the capacity handed to the reducer, and are fetched at a trigger chosen once.

The three policy axes are supplied as explicit `uint8_t` arrays with counts, so
a caller evaluates exactly the combinations it asks for; their product is the
candidate count per problem. Their values are
`ShadowSpillPressureFitResidencyStrategy`
(`SHADOWSPILL_PRESSUREFIT_RESIDENCY_` then `HEADROOM_STALL`,
`HEADROOM_TRANSFER`, `TIGHT_STALL`, `TIGHT_TRANSFER` or `RELAXED_STALL`),
`ShadowSpillPressureFitFetchRule` (`SHADOWSPILL_PRESSUREFIT_FETCH_` then
`PACKED_FIFO`, `PACKED_FIT`, `INTERVAL_ENTRY`, `LATEST_SAFE` or `DEMAND`), and,
for coalescing, 0 plain and 1 coalesced; `initial_placement` takes a
`ShadowSpillPressureFitInitialPlacement`, required or greedy. What each one does
to a plan is tabulated in [PressureFit](../architecture/pressurefit.md).

A plan that comes up short of capacity is never finished, whatever its
makespan: the waiting is time it pays, and the shortfall behind the waiting is
what reduction relieves. A candidate reaching such a plan keeps going and
answers with the best plan it found, not the first that ran.

`ShadowSpillPressureFitResult` holds the status, the selected
`ShadowSpillIndexedSchedule` and its makespan, one
`ShadowSpillPressureFitCandidateDiagnostic` per candidate, the repair and work
counters summed over them, when the problem ran (`started_ns`/`finished_ns`),
and the objects kept resident (`evict_ineligible_aliases`, their bytes, the
resident slice reserved for them as `resident_slice_bytes` per device, and
which they are as `alias_evict_eligible` per alias, both owned by the result).

When an incumbent was given, the result says what became of it
(`incumbent_given`, `incumbent_status` in the candidate-status vocabulary,
`incumbent_makespan_ns`, `incumbent_required_bytes`, `incumbent_selected`);
when it is the answer, `selected_candidate_index` is
`SHADOWSPILL_PLANNER_NO_INDEX` and the selected schedule and makespan are its
own. An incumbent that places is offered to the shared record like any other
placed plan, so in the default mode it bounds every candidate from the start;
under `deterministic` the record is not consulted and the incumbent bounds
nothing, though it still wins unless a candidate beats it.

The candidate diagnostic is where the detail is: a
`ShadowSpillPressureFitCandidateStatus`, the three policy codes that produced
it, its own repair, work and section counters, the placement counters, the
schedule digest, its own span, and the boundary a failure stopped at — task,
alias, device, location, boundary, time, and the capacity, used, requested and
required bytes around it. A candidate reports
`SHADOWSPILL_PRESSUREFIT_CANDIDATE_UNPLACEABLE` when every plan it reached
needed more contiguous pool than the pool has.

## Functions

| Call | Arguments | Returns |
|---|---|---|
| `shadowspill_pressurefit_search` | `const ShadowSpillIndexedProblem *problems`, `uint32_t problem_count`, `const ShadowSpillPressureFitOptions *options`, `ShadowSpillPressureFitResult *results` (`problem_count` entries) | `ShadowSpillStatus` |
| `shadowspill_pressurefit_preflight` | `const ShadowSpillIndexedProblem *problem`, `ShadowSpillPressureFitPreflightResult *result` | `ShadowSpillStatus` |
| `shadowspill_pressurefit_result_destroy` | `ShadowSpillPressureFitResult *result` | `void` |
| `shadowspill_pressurefit_best_placed_create` | none | `ShadowSpillPressureFitBestPlaced *` |
| `shadowspill_pressurefit_best_placed_destroy` | `ShadowSpillPressureFitBestPlaced *best` | `void` |
| `shadowspill_pressurefit_best_placed_read` | `const ShadowSpillPressureFitBestPlaced *best`, `ShadowSpillPressureFitBestPlacedRecord *record` | `void` |

- `shadowspill_pressurefit_search()` takes one `ShadowSpillIndexedProblem` per
  resolved program, derives each one's residency problem from its own program,
  and evaluates every policy for each. A problem whose derivation fails — a
  resolved program that is analytically infeasible at this capacity, say —
  carries that status on its own result, and the others are evaluated together
  as if it were absent; the call's status is the evaluation's, or the first
  refusal's when nothing could be derived. It takes a count, so evaluating a
  single problem is passing one; there is no separate single-problem entry
  point. A candidate of a problem is the unit of work and
  every candidate of every problem competes for the same workers, so **worker
  count and problem count are independent** — asking for eight workers gets
  eight threads whether there is one resolved program or five. The threads
  belong to the call, so concurrent callers do not contend for one another's
  workers, and `options.workers` sizes them (zero for one per logical CPU, one
  to evaluate on the calling thread).

  Evaluating them together is what shares the placement record between them:
  a plan placed under any resolved program bounds the search under every
  other. Results are written one per problem in input order.

  Worker count is scheduling, not an input to the search: it changes neither
  which plans are legal nor how they simulate. It does change how much of the
  search is skipped, since a candidate is skipped when the record already
  holds something it cannot beat, so per-candidate counters like
  `placements_attempted` move with it and so can the choice between plans
  that tie. Each result owns its storage afterwards, including when the call
  reports a failure, since problems that completed still hold theirs.
- `shadowspill_pressurefit_preflight()` fills a
  `ShadowSpillPressureFitPreflightResult` without evaluating candidate
  policies. Its `failure_kind` is a
  `ShadowSpillPressureFitPreflightFailureKind`: workspace capacity, required
  capacity, missing initial residency, or a resident slice that does not fit
  the device on its own.
- `shadowspill_pressurefit_result_destroy()` releases arrays owned by a
  problem result.
- `shadowspill_pressurefit_best_placed_create()`,
  `shadowspill_pressurefit_best_placed_destroy()` and
  `shadowspill_pressurefit_best_placed_read()` are the whole public surface of
  the shared record of the best plan any caller has actually placed; candidates
  offer their placed plans into it internally. `read()` copies out a
  `ShadowSpillPressureFitBestPlacedRecord`: the makespan, the object capacity
  that plan was built against, how much capacity it gave back, the three policy
  codes, and the schedule digest. A `makespan_ns` of zero means nothing has
  been placed. The object also keeps its own copy of the plan, replaced in
  place when a better one arrives, so the record never names a plan nobody
  still holds, but that copy is an internal storage type and does not cross the
  ABI — which is why offering a plan is internal too. Read the schedule a
  search chose from the problem result, not from here: each problem answers
  with its own best plan, while one record may be shared across problems and
  across calls.

  What the record is for is skipping work. Placing a plan is expensive and a
  plan no better than one already placed cannot win even if it places, so the
  search compares against the bound the record holds before paying for a
  measurement. The object knows nothing about candidates, resolved programs
  or calls: passing one object to several concurrent searches shares the gate
  between them, and passing separate objects keeps them independent. It is
  safe to use from several threads at once. The bound is a single atomic word
  and reading it never waits; offering and reading the whole record take a
  spin lock, which is affordable because a placement that succeeds is rare
  next to the checks preceding it. A stale read of the bound costs at most a
  measurement that would have been skipped, never a wrong answer: the best
  plan that will ever be placed is better than everything already placed, so
  it is never refused. Setting `deterministic` takes the shared record out of
  the gate entirely, leaving each candidate to consult only its own placed
  plans.

## Diagnostics

`ShadowSpillPressureFitWorkDiagnostics` counts the work one evaluation did:
schedule emissions/cache hits, simulation calls and cache hits, and admission
calls. What placing cost and bought — `placements_attempted`,
`placements_admitted`, `capacity_refinements` — is per candidate, on the
candidate diagnostic, as is `repairs_at_best`, the repairs spent when the plan
the candidate answers with was placed, and `pressure_escalations` /
`escalations_taken_back`, the pressure repairs that asked for more than the
shortfall because a failure had repeated, and how many of those asks no cut
could meet.

Time is reported separately, as `ShadowSpillPressureFitSectionTiming`. Its
fields are **disjoint sections** rather than overlapping totals: each names
one span of work, an orchestrator opens and closes it, and no two are ever
open at once. `total_ns` is the whole span the orchestrator measured, and
`residual_ns` is what it holds that no named section claimed, so

```text
total_ns = prepare + setup + reduce + emit + simulate + repair
         + digest + place + select + teardown + residual
```

holds exactly, at every level. The one exception is `admit_ns`, which is
nested *inside* `simulate_ns` rather than beside it, because admission runs
as part of simulating; it is excluded from the identity above for that
reason. `prepare_ns` is a problem-level section only — preparing the problem
happens once, before any candidate exists. `reduce_ns` covers a strategy's
base reduction; a reduction a repair forces is charged to `repair_ns`, which
is what pays for it.

What each section covers is walked through stage by stage in
[PressureFit](../architecture/pressurefit.md#current-algorithm).

Each candidate diagnostic carries its own `sections` covering just that
candidate. A problem's sections are the sum of its candidates', totals and
residuals included, which is what keeps the identity above true of the sum;
`prepare_ns` and `teardown_ns` are added by the entry point around them.

Sections measure work, not elapsed time. Once several workers run at once a
problem's total exceeds the time the call took, which is the point of the
workers. Elapsed time is reported separately, as `started_ns` and
`finished_ns` on both the candidate and the problem: nanoseconds from the
start of the call that evaluated them. Every span in one call shares that
origin, so two candidates ran at the same time exactly when their spans
overlap, and a problem spans its candidates. Both are zero for a candidate no
worker reached.

`ShadowSpillPressureFitRepairDiagnostics` categorizes each monotonic repair by
whether it advances or delays a fetch or addresses a pressure boundary, and by
which of the two rejections asked for it: an admission refusal or a simulated
one. Candidate, problem, and aggregate Python diagnostics preserve these
counts.

### Reduction trajectories

With `record_reduction_steps` set, each candidate also records what its
search actually did, one `ShadowSpillPressureFitReductionStep` per plan it
held. A step carries the plan's makespan, the bytes its layout required, the
capacity it was built against, the repair count reaching it, the simulation
status, how many capacity boundaries it came up short at, and a `flags` word
of `ShadowSpillPressureFitReductionStepFlags` saying what happened to it —
whether it was simulated, measured for layout, placed, triggered a refinement,
was at some point the best plan, or is the answer the candidate returned.
`cut_offset`/`cut_count` index the candidate's flat `cut_aliases` array, naming
the objects the reducer cut to reach that step.

Recording is off by default: it costs an allocation per candidate that grows
with the search, which is worth paying when attributing planner time or
explaining a plan and not otherwise.

## Struct sizes

`enum ShadowSpillPressureFitStruct` names this search's own structures for
[`shadowspill_planner_struct_size()`](planner.md#functions):
`SHADOWSPILL_PRESSUREFIT_STRUCT_` then `OPTIONS`, `WORK_DIAGNOSTICS`,
`CANDIDATE_DIAGNOSTIC`, `SECTION_TIMING`, `REDUCTION_STEP`,
`BEST_PLACED_RECORD` or `RESULT`. The values continue
`enum ShadowSpillPlannerStruct` rather than restarting, so one call answers for
the generic planner and for this search, and a caller mirroring these layouts —
the Python bindings do — compares its sizes at load rather than discovering a
mismatch as corrupted counters.

## Ownership

All problem input arrays are borrowed. Each problem result owns its selected
schedule and candidate array until
`shadowspill_pressurefit_result_destroy()` releases them, including when the
call reports a failure: problems that completed still hold theirs. Calls with
distinct inputs and results are independent; the search performs no I/O and
owns no global mutable state beyond a best-placed record a caller passes it.
