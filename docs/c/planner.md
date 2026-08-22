# Planner C API

Include `<shadowspill/planner.h>`. The compiled planner evaluates one
PressureFit candidate context or a complete predecoded Program context using
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
`ShadowSpillAdmissionTopology` adds exact task allocation/free steps, the
anonymous live-set peak derived from those steps, fresh outputs,
replacements, handoffs, and task-allocation slots. Executable admission never
constructs allocation steps from scalar workspace or output totals.

`ShadowSpillPressureFitContext` accepts an already derived residency problem.
`ShadowSpillPressureFitProgramContext` accepts the schedule-invariant
simulation Program and derives residency inputs internally.

Candidate options select residency strategy, fetch rule, coalescing, repair
limit, and initial placement. Results contain the selected indexed schedule,
every candidate status, exact repair counters, component work counters,
timings, and failure boundary.

`ShadowSpillPlacementProblem` is independent of the rest of the data model:
it is five parallel arrays describing one lease each — size, alignment,
predicted lifetime, and lease id — and `ShadowSpillPlacementResult` receives
one offset per lease plus the total bytes the slice requires. The offsets
array is caller-owned, so placement allocates nothing the caller must release.

## Functions

- `shadowspill_select_plan()` selects from an explicitly supplied candidate
  set.
- `shadowspill_reduce_residency()` solves the indexed residency problem.
- `shadowspill_evaluate_pressurefit_context()` evaluates all policies for one
  derived context.
- `shadowspill_evaluate_pressurefit_program_context()` derives and evaluates a
  complete context from schedule-invariant inputs.
- `shadowspill_validate_pressurefit_program_context()` returns the structured
  workspace, required-capacity, or missing-initial-residency preflight result
  without evaluating candidate policies.
- `shadowspill_evaluate_schedule_admission()` checks one selected schedule
  against the exact admission topology.
- `shadowspill_pressurefit_context_result_destroy()` releases arrays owned by
  a context result.
- `shadowspill_place_lifetimes()` assigns each lease a fixed offset within
  one execution-pool slice and reports the bytes required. Leases are placed
  largest first, longest-lived first among equals, and each takes the lowest
  aligned offset clearing every lease it overlaps in time; lifetimes are
  half-open, so leases that merely touch may share an offset. The result is a
  function of the lease set alone, independent of input order.
- `shadowspill_planner_abi_version()` and
  `shadowspill_planner_status_string()` support loading and diagnostics.

## Diagnostics

`ShadowSpillPressureFitWorkDiagnostics` separately counts residency cache
hits/misses, schedule emissions/cache hits, simulation calls/cache hits,
admission calls, and time spent in residency, schedule construction,
simulation, admission, and digesting.

`ShadowSpillPressureFitRepairDiagnostics` categorizes each monotonic repair by
whether it advances or delays a fetch or addresses a pressure boundary.
Candidate, context, and aggregate Python diagnostics preserve these counts.

## Concurrency and ownership

All context input arrays are borrowed. Each context result owns its selected
schedule and candidate array until destroyed. Calls with distinct inputs and
results are independent; the API performs no I/O and does not own global
mutable planning state.

`shadowspill_place_lifetimes()` follows the same rules: it borrows the problem
arrays, writes only the caller's offsets array, and keeps all scratch state on
the stack or in allocations it frees before returning. Concurrent calls on
distinct problems and results are safe.
