# Planner C API

Include `<shadowspill/planner.h>`. This is the planning question and the parts
of answering it that do not depend on how the answer was found: a problem in
indexed form, the schedule a search returns, and the certification a schedule
has to pass — exact admission against the pool policy, the lease lifetimes a
schedule implies, and the fixed-offset placement those resolve to.

Nothing here names a search. The search that ships is PressureFit, whose
options, diagnostics and entry points are in
[`<shadowspill/pressurefit/pressurefit.h>`](pressurefit.md); a second search
would ship its own header beside it and reuse everything on this page
unchanged. A caller that only wants to certify a schedule or place leases
needs this header alone.

Training [graph-pair construction](../architecture/graph-pair-construction.md)
and [complete graph-pair selection](../architecture/graph-pair-selection.md)
are separate frontend/planner concerns. Exact range placement is documented in
[physical admission](../architecture/physical-admission.md).

## Data model

A `ShadowSpillIndexedSchedule` is a schedule in indexed form: every identifier
is the contiguous task or alias index in the simulation program it was placed
over. A search returns one inside its own result, which owns the arrays and
says when they are released.

`ShadowSpillAdmissionFacts` carries exact task allocation/free steps, the
anonymous live-set multiset flattened per task from those steps, fresh outputs,
replacements, handoffs, and task-allocation slots. Executable admission never
constructs allocation steps from scalar workspace or output totals.

A `ShadowSpillIndexedProblem` is the question a search is handed, before any
schedule exists for it: a `ShadowSpillScheduleContext`, the `device_priority`
to plan against, an `abi_version`, and optionally an incumbent. A search
resolves this into whatever it evaluates; deciding which resolved programs
exist, and in what order to try them, belongs to the search and not to this API.

`incumbent` is the plan to beat: an indexed schedule for this problem already
in hand — found at a smaller capacity, say — or NULL. A search given one
measures it at this capacity before any candidate runs, exactly as it measures
a candidate's plan (simulated, admitted, placed against the pool), and answers
with it unless a candidate does strictly better, so a search handed one never
answers worse than it.

`ShadowSpillScheduleContext` is the part of a problem that is not about how it
is searched: the `ShadowSpillSimulationProgram` its schedules run on, the
ownership facts they must fit, and the JSON-escaped alias and task names they
are written under. Certifying a schedule, digesting it, and replaying it through
the pool are the same questions whichever search produced the schedule, so a
context is what every problem embeds and what the library's own generic code
works on. The entry points below take the two pieces they use directly — a
simulation program and one set of admission facts — so a caller holding neither
a problem nor a search can still certify or place a schedule.

The context carries two independent sets of admission facts, and either may be
null. `admission` switches on the dynamic-pool replay, which rejects a
candidate whose schedule that policy cannot place; null skips the replay and
leaves the schedule judged by simulation alone. `placement` supplies the same
facts for measuring layouts during the search without that filter, because a
schedule the dynamic replay rejects can still have a valid dependency-certified
fixed placement; a search that prefiltered through `admission` would discard
plans that would have run. Null leaves plans unplaced, which is how a caller
opts out of measuring layouts during the search.

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
intervals say when. `ShadowSpillLeaseLifetimeResult` is caller-owned columns,
one entry per lease: the four numbers placement reads in
`ShadowSpillLeaseLifetime`, and beside them the `ShadowSpillLeaseIdentity` a
certificate needs — lease id, causal boundaries, purpose, and task, alias and
action indices. Two more columns come back with them, one entry per flattened
allocation step and one per alias, naming the lease each holds when the step
ends.

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

Every call returns a `ShadowSpillStatus` except the last, whose return is the
answer. Input pointers are borrowed; output structs are the caller's.

| Call | Arguments |
|---|---|
| `shadowspill_evaluate_schedule_admission` | `const ShadowSpillSimulationProgram *simulation`, `const ShadowSpillAdmissionFacts *admission`, `const ShadowSpillIndexedSchedule *schedule`, `ShadowSpillScheduleAdmissionResult *result` |
| `shadowspill_admission_operation_bounds` | the same first three, then `uint64_t *operation_capacity` and `uint64_t *lease_capacity`, the two array sizes the builder needs |
| `shadowspill_build_admission_operations` | the same first three, then `ShadowSpillAdmissionOperations *result`, whose arrays the caller has sized from those bounds |
| `shadowspill_build_lease_lifetimes` | `const ShadowSpillLeaseLifetimeProblem *problem`, `ShadowSpillLeaseLifetimeResult *result` |
| `shadowspill_place_lifetimes` | `const ShadowSpillPlacementProblem *problem`, `ShadowSpillPlacementResult *result` |
| `shadowspill_planner_struct_size` | `uint32_t which`, a `ShadowSpillPlannerStruct`; returns `uint64_t` |

- `shadowspill_evaluate_schedule_admission()` checks one selected schedule
  against the exact admission facts, reporting the decision digest, the
  allocation, reservation and fragmentation peaks, the physical byte delta
  each task and action causes, and the reuse dependencies the schedule
  implies. Every array on the `ShadowSpillScheduleAdmissionResult` is
  caller-owned.
- `shadowspill_admission_operation_bounds()` reports how many operations and
  leases a schedule will produce, so the caller can size the arrays the
  builder fills. It is pure arithmetic over the facts and schedule and
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
  leases given static homes in a resident slice stay out of the main
  assignment. Where no two records tie, the layout depends on the records
  alone and not on the order they were listed in. The assignment and the
  structure behind it are specified in
  [fixed-offset placement](../architecture/fixed-placement.md).
- `shadowspill_planner_struct_size()` reports the compiled size of one
  planner structure, named by `enum ShadowSpillPlannerStruct`, and zero for a
  value it does not know. The enum spans this header and whichever searches
  ship, since a caller mirroring either set of layouts — the Python bindings
  mirror both — can then compare sizes at load and refuse a library it does
  not match. Drift is otherwise silent: the mirror reads one field where the
  library wrote another and reports corrupted counters rather than an error.

`shadowspill_abi_version()` and `shadowspill_status_string()` cover loading
and diagnostics for this boundary as for every other; see the
[C API guide](README.md#abi-use).

## Constants

`SHADOWSPILL_PLANNER_NO_INDEX` is the absent-index sentinel every `uint32_t`
index field uses — no task, no alias, no action, no allocation step.
`SHADOWSPILL_PLANNER_DIGEST_BYTES` is the length of a schedule digest, 32
bytes. `SHADOWSPILL_ADMISSION_NO_DEPENDENCY`, `_NO_OPERATION` and `_NO_LEASE`
are the `uint64_t` equivalents for an operation that publishes no dependency, a
lease that outlives the step, and an allocation step or alias holding no lease.

## Concurrency and ownership

All problem input arrays are borrowed. Calls with distinct inputs and results
are independent; the API performs no I/O and does not own global mutable
planning state.

`shadowspill_place_lifetimes()` follows the same rules: it borrows the problem
arrays, writes only the caller's offsets array, and keeps all scratch state on
the stack or in allocations it frees before returning. Concurrent calls on
distinct problems and results are safe.
