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
final locations, task access, transfer cost, and per-boundary capacity.
`ShadowSpillAdmissionFacts` adds exact task allocation/free steps, the
anonymous live-set peak derived from those steps, fresh outputs,
replacements, handoffs, and task-allocation slots. Executable admission never
constructs allocation steps from scalar workspace or output totals.

`ShadowSpillPressureFitProblem` accepts an already derived residency problem.
`ShadowSpillPressureFitProgramProblem` accepts the schedule-invariant
simulation Program and derives residency inputs internally.

Candidate options select residency strategy, fetch rule, coalescing, repair
limit, and initial placement. Results contain the selected indexed schedule,
every candidate status, exact repair counters, component work counters,
timings, and failure boundary.

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
- `shadowspill_evaluate_pressurefit_problem()` evaluates all policies for one
  derived problem.
- `shadowspill_evaluate_pressurefit_program_problem()` derives and evaluates a
  complete problem from schedule-invariant inputs.
- `shadowspill_validate_pressurefit_program_problem()` returns the structured
  workspace, required-capacity, or missing-initial-residency preflight result
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
  half-open, so leases that merely touch may share an offset. Where no two
  records tie, the layout depends on the records alone and not on the order
  they were listed in. The assignment and the structure behind it are
  specified in [fixed-offset placement](../architecture/fixed-placement.md).
- `shadowspill_best_placed_create()`, `shadowspill_best_placed_destroy()`,
  `shadowspill_best_placed_admits()`, `shadowspill_best_placed_offer()` and
  `shadowspill_best_placed_read()` share the best plan any caller has actually
  placed, as a `ShadowSpillBestPlacedRecord` carrying the makespan, the object
  capacity that plan was built against, the caller's selection index, the
  candidate policy and the schedule digest. Whatever holds the record at the
  end is the plan the search selected, so "is this worth measuring" and "what
  won" are answered by one object. Placing a plan is expensive and a plan no better than one
  already placed cannot win even if it places, so a search calls `admits()`
  before paying and `offer()` once a placement succeeds. The object knows
  nothing about candidates, resolved programs or calls: passing one object to
  several concurrent searches shares the gate between them, and passing
  separate objects keeps them independent. It is lock-free and safe to use
  from several threads at once. `admits()` reads a single atomic word so it
  never waits; `offer()` and `read()` take a spin lock over the record, which
  is affordable because a placement that succeeds is rare next to the checks
  preceding it. A stale `admits()` read costs at most a measurement that would
  have been skipped, never a wrong answer: the best plan that will ever be
  placed is better than everything already placed, so it is never refused.

`shadowspill_abi_version()` and `shadowspill_status_string()` cover loading
and diagnostics for this boundary as for every other; see the
[C API guide](README.md#abi-use).

## Diagnostics

`ShadowSpillPressureFitWorkDiagnostics` separately counts residency cache
hits/misses, schedule emissions/cache hits, simulation calls/cache hits,
admission calls, and time spent in residency, schedule construction,
simulation, admission, and digesting.

`ShadowSpillPressureFitRepairDiagnostics` categorizes each monotonic repair by
whether it advances or delays a fetch or addresses a pressure boundary.
Candidate, problem, and aggregate Python diagnostics preserve these counts.

## Concurrency and ownership

All problem input arrays are borrowed. Each problem result owns its selected
schedule and candidate array until destroyed. Calls with distinct inputs and
results are independent; the API performs no I/O and does not own global
mutable planning state.

`shadowspill_place_lifetimes()` follows the same rules: it borrows the problem
arrays, writes only the caller's offsets array, and keeps all scratch state on
the stack or in allocations it frees before returning. Concurrent calls on
distinct problems and results are safe.
