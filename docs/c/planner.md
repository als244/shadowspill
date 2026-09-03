# Planner C API

Include `<shadowspill/planner.h>`. The planner evaluates one
PressureFit candidate problem or a complete predecoded Program problem using
the simulator and exact schedule admission.

The framework-neutral problem formulation and complete algorithm are in the
[PressureFit architecture page](../architecture/pressurefit.md). Training
[graph-pair construction](../architecture/graph-pair-construction.md) and
[complete recomputation selection](../architecture/recomputation-selection.md)
are separate frontend/planner concerns. Exact range placement is documented in
[physical admission](../architecture/physical-admission.md).

## Data model

`ShadowSpillResidencyProblem` contains indexed aliases, boundaries, initial and
final locations, task access, transfer cost, per-boundary capacity, and, for
the aliases the reducer may not cut, `alias_evict_eligible` and the
`fixed_fetch_trigger` their one fetch is issued at.
`ShadowSpillAdmissionFacts` adds exact task allocation/free steps, the
anonymous live-set multiset flattened per task from those steps, fresh outputs,
replacements, handoffs, and task-allocation slots. Executable admission never
constructs allocation steps from scalar workspace or output totals.

`ShadowSpillPressureFitProblem` accepts an already derived residency problem.
`ShadowSpillPressureFitProgramProblem` accepts the schedule-invariant
simulation Program and derives residency inputs internally. One problem is one
resolved program: deciding which resolved programs exist, and in what order to
try them, belongs above this API. The shared best-placed record is the only
object that crosses that boundary.

Both problems carry two independent sets of admission facts. `admission`
switches on the dynamic-pool replay, which rejects a candidate whose schedule
that policy cannot place. `placement` supplies the same topology for measuring
layouts during the search without that filter, because a schedule the dynamic
replay rejects can still have a valid dependency-certified fixed placement; a
search that prefiltered through `admission` would discard plans that would
have run. Either may be null.

Candidate options select residency strategies, fetch rules, coalescing modes,
repair limit, initial placement, how much capacity a plan gives back at a time
when its layout does not fit (`capacity_refinement_bytes`, zero for the whole
shortfall), whether each candidate records its reduction trajectory
(`record_reduction_steps`), how many threads the call searches on
(`workers`, zero for one per logical CPU and one for the calling thread), the
shared best-placed record to measure against, and which objects are too small
to be worth cutting (`minimum_object_bytes_evict_eligible`, 1 MiB by default
through the Python request, zero for none):
those stay resident from first to last access, take static homes in a
resident slice whose size is reserved at preparation out of the capacity
handed to the reducer, and are fetched at a trigger chosen once. The three
policy axes are
supplied as explicit arrays with counts, so a caller evaluates exactly the
combinations it asks for; their product is the candidate count per problem.

A plan that comes up short of capacity is never finished, whatever its
makespan: the waiting is time it pays, and the shortfall behind the waiting is
what reduction relieves. A candidate reaching such a plan keeps going and
answers with the best plan it found, not the first that ran.

Results contain the selected indexed schedule, every candidate status, exact
repair counters, per-candidate placement counters, work counters, the
sections its time went to, when it ran (`started_ns`/`finished_ns`), the
objects kept resident (`evict_ineligible_aliases`, their bytes, the
resident slice reserved for them as `resident_slice_bytes` per device, and
which they are as `alias_evict_eligible` per alias, both owned by the
result), and failure boundary. A candidate
reports `SHADOWSPILL_CANDIDATE_UNPLACEABLE` when every plan it reached needed
more contiguous pool than the pool has.

`ShadowSpillAdmissionOperations` is parallel arrays in two families. Arrays
indexed by operation hold `operation_capacity` entries — an operation's
sequence is its index — and carry its lease, the completion a reuse of that
lease's address must await, the bytes and alignment it reserves, its kind, why
the lease exists, where in the step it sits, which task or action that
boundary names, and the allocation step behind it when it is a task
allocation. Arrays indexed by lease hold `lease_capacity` entries and carry
the alias a lease owns plus the operations that create and retire it, so a
reader can go straight to a lease instead of scanning: several operations
touch each lease and most touch none that matters.

Every array is caller-owned and sized from
`shadowspill_admission_operation_bounds`, so the builder allocates nothing the
caller must release. The result also reports how many operations, leases and
dependencies were produced, and the bytes each transfer lane must move.

`ShadowSpillLeaseLifetimeProblem` joins the two: the operations say which
lease each one creates and retires, and the simulated task and transfer
intervals say when. The result is caller-owned columns, one entry per lease:
the four numbers placement reads in `ShadowSpillLeaseLifetime`, and beside
them the `ShadowSpillLeaseIdentity` a certificate needs - lease id, causal
boundaries, purpose, and task, alias and action indices.

Two things about that result are deliberate. **The identity columns are
written on every call and read on almost none**: a measurement wants the bytes
a schedule needs and reads one scalar, so only a certified layout decodes an
identity, and only for the leases it keeps. And **the records come back
partitioned**: fixed leases occupy `[0, fixed_count)` with the caller-owned
dynamic ones after, so placement runs on the prefix without a copy and neither
function has to know about the other.

`ShadowSpillPlacementProblem` is independent of the rest of the data model. It
is one array of `ShadowSpillLeaseLifetime`, each holding the four numbers
placement reads — size, alignment, and the half-open interval the lease is
live over — and `ShadowSpillPlacementResult` receives one offset per lease, in
input order, plus the total bytes the slice requires. The offsets array is
caller-owned, so placement allocates nothing the caller must release.

Placement is never told which lease a record belongs to. It has no use for
identity beyond breaking ties between records that are equal in every key, and
the input index does that, so the result is a function of the records and the
order they arrive in.

## Functions

- `shadowspill_select_plan()` selects from an explicitly supplied candidate
  set.
- `shadowspill_reduce_residency()` solves the indexed residency problem.
- `shadowspill_evaluate_pressurefit_problems()` evaluates all policies for one
  or more already-derived problems.
  `shadowspill_evaluate_pressurefit_program_problems()` derives those problems
  from schedule-invariant inputs first, then does the same. Both take a count,
  so evaluating a single problem is passing one; there is no separate
  single-problem entry point. A candidate of a problem is
  the unit of work and every candidate of every problem competes for the same
  workers, so **worker count and problem count are independent** — asking for
  eight workers gets eight threads whether there is one resolved program or
  five. The threads belong to the call, so concurrent callers do not contend
  for one another's workers, and `options.workers` sizes them (zero for one
  per logical CPU, one to evaluate on the calling thread).

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
- `shadowspill_validate_pressurefit_program_problem()` returns the structured
  workspace, required-capacity, resident-slice, or missing-initial-residency
  preflight result
  without evaluating candidate policies.
- `shadowspill_evaluate_schedule_admission()` checks one selected schedule
  against the exact admission topology.
- `shadowspill_pressurefit_problem_result_destroy()` releases arrays owned by
  a problem result.
- `shadowspill_admission_operation_bounds()` reports how many operations and
  leases a schedule will produce, so the caller can size the arrays the
  builder fills. It is pure arithmetic over the topology and schedule and
  allocates nothing.
- `shadowspill_build_admission_operations()` derives the pool operations a
  schedule implies, with the provenance a fixed layout needs: where each
  operation sits, why each lease exists, and which allocation step produced
  it. It also reports the bytes each transfer lane must move, which bound the
  schedule's makespan without simulating. The rules it follows are specified
  in [from a resolved program to leases](../architecture/admission-leases.md).
- `shadowspill_build_lease_lifetimes()` resolves every lease a schedule
  creates to the interval it is live over and the identity it carries, and
  moves the caller-owned terminal aliases named in `dynamic_aliases` out of
  the fixed prefix. It also reports the lease each allocation step used and
  the lease each alias ends the step holding, which are what a certificate's
  lookup tables are built from. It allocates only scratch it frees before
  returning. The rules it follows are specified in
  [from a resolved program to leases](../architecture/admission-leases.md).
- `shadowspill_place_lifetimes()` assigns each lease a fixed offset within
  one execution-pool slice and reports the bytes required. Leases are placed
  largest first, longest-lived first among equals, and each takes the lowest
  aligned offset clearing every lease it overlaps in time; lifetimes are
  half-open, so leases that merely touch may share an offset. A lease marked
  `excluded` is left out, unplaced and outside the span, which is how the
  leases given static homes in the resident slice stay out of the main
  assignment. Where no two
  records tie, the layout depends on the records alone and not on the order
  they were listed in. The assignment and the structure behind it are
  specified in [fixed-offset placement](../architecture/fixed-placement.md).
- `shadowspill_best_placed_create()`, `shadowspill_best_placed_destroy()`
  and `shadowspill_best_placed_read()` share the best plan any caller has
  actually placed; candidates offer their placed plans into it internally. The record carries the
  makespan, the object capacity that plan was built against, how much
  capacity it gave back, the caller's selection index, the candidate policy
  and the schedule digest; the object also keeps its own copy of the plan,
  replaced in place when a better one arrives, so the record never names a
  plan nobody still holds. Whatever it holds at the end is the plan the
  search selected — selection reads it rather than ranking the candidates a
  second time. Placing a plan is expensive and a plan no better than one
  already placed cannot win even if it places, so a search calls `admits()`
  before paying. `shadowspill_best_placed_offer()` is internal, since the
  plan it keeps is an internal storage type. The object knows
  nothing about candidates, resolved programs or calls: passing one object to
  several concurrent searches shares the gate between them, and passing
  separate objects keeps them independent. It is lock-free and safe to use
  from several threads at once. `admits()` reads a single atomic word so it
  never waits; `offer()` and `read()` take a spin lock over the record, which
  is affordable because a placement that succeeds is rare next to the checks
  preceding it. A stale `admits()` read costs at most a measurement that would
  have been skipped, never a wrong answer: the best plan that will ever be
  placed is better than everything already placed, so it is never refused.

- `shadowspill_planner_struct_size()` reports the compiled size of one
  planner structure, named by `enum ShadowSpillPlannerStruct`. A caller that
  mirrors these layouts — the Python bindings do — can compare sizes at load
  and refuse a library it does not match. Drift is otherwise silent: the
  mirror reads one field where the library wrote another and reports
  corrupted counters rather than an error.

`shadowspill_abi_version()` and `shadowspill_status_string()` cover loading
and diagnostics for this boundary as for every other; see the
[C API guide](README.md#abi-use).

## Diagnostics

`ShadowSpillPressureFitWorkDiagnostics` counts the work one evaluation did:
schedule emissions/cache hits, simulation calls
and cache hits, and admission calls. What placing cost and bought —
`placements_attempted`, `placements_admitted`, `capacity_refinements` — is
per candidate, on the candidate diagnostic, as is `repairs_at_best`, the
repairs spent when the plan the candidate answers with was placed.

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
whether it advances or delays a fetch or addresses a pressure boundary.
Candidate, problem, and aggregate Python diagnostics preserve these counts.

### Reduction trajectories

With `record_reduction_steps` set, each candidate also records what its
search actually did, one `ShadowSpillPressureFitReductionStep` per plan it
held. A step carries the plan's makespan, the bytes its layout required, the
capacity it was built against, the repair count reaching it, the simulation
status, how many capacity boundaries it came up short at, and a `flags` word
saying what happened to it — whether it was simulated, measured for layout,
placed, triggered a refinement, was at some point the best plan, or is the
answer the candidate returned. `cut_offset`/`cut_count` index the candidate's
flat `cut_aliases` array, naming the objects the reducer cut to reach that
step.

Recording is off by default: it costs an allocation per candidate that grows
with the search, which is worth paying when attributing planner time or
explaining a plan and not otherwise.

## Concurrency and ownership

All problem input arrays are borrowed. Each problem result owns its selected
schedule and candidate array until destroyed. Calls with distinct inputs and
results are independent; the API performs no I/O and does not own global
mutable planning state.

`shadowspill_place_lifetimes()` follows the same rules: it borrows the problem
arrays, writes only the caller's offsets array, and keeps all scratch state on
the stack or in allocations it frees before returning. Concurrent calls on
distinct problems and results are safe.
