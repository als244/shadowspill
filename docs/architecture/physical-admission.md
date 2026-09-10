# Physical admission and offset handling

Physical admission proves that one logical schedule can be assigned real ranges
inside its configured memory pools without unsafe overlap. It runs after a
search has chosen tasks, residency and memory actions, and before an executable
callable is published.

Admission is search-agnostic. It takes a schedule and the facts about the
machine it must fit, and certifies that pair; which search produced the
schedule, and how, is not part of the question. [PressureFit](pressurefit.md)
is the search that ships, and nothing on this page is specific to it.

A search answers:

> Which objects should be resident, and when should release, write-back,
> evict, and fetch actions trigger?

Physical admission answers:

> Which exact execution-pool bytes back every selected object generation,
> task allocation, and fetch destination, and what completion proof protects
> each reused range?

The answer is a fixed layout for schedule-managed execution memory plus bounded
dynamic regions for caller-owned terminal outputs and optional scratch. The
spill pool is dynamically allocated and is checked against its simulated
physical peak.

Runtime-global shared leases lie outside a callable's physical layout. Their
bytes are subtracted before the callable-owned pool slice is constructed, while
diagnostics keep both the shared footprint and residual movable capacity.
Admission never assigns a second offset or transfer destination to a shared
alias.

## Inputs and output

Physical admission consumes:

| Input | Purpose |
|---|---|
| `ProgramPlanResult` | Selected tasks, residency, ordered actions, and logical simulation. |
| `AdmissionFacts` | Pool capacity, task allocation geometry, output/replacement ownership, handoffs, and alignment. |
| `TaskAllocationContract` values | Stable task-local invariant allocation/free identities and geometry. |
| Dynamic-scratch reserve | Bounded capacity for optional allocator operations outside the strict core. |
| Terminal caller-owned aliases | Final execution leases that may outlive a later callable invocation. |

It returns `FixedLayoutAdmission`, which contains:

- a `FixedPhysicalLayout` with placements and causal reuse dependencies;
- a `SimulationAdmission` projection of the physical certificate;
- a new `SimulationResult` that includes the added dependencies;
- stable digests tying the layout to its program, schedule, and facts
  (`program_digest`, `schedule_digest`, `facts_digest`).

### Measuring and certifying are separate steps

`build_fixed_layout_admission` is the entry point that returns the above, and it
composes two halves of very different cost that can also be called on their own.

`measure_fixed_layout` replays the schedule into leases, gives each a
lifetime, and places them. It returns a `FixedLayoutMeasurement` carrying
`required_bytes` and the pool capacity, and it never rejects: a layout larger
than the pool is reported through `fits`, `slack_bytes` and
`shortfall_bytes`, so a caller searching for a schedule that fits can act on
how far it missed by instead of catching an exception to find out.

`certify_fixed_layout` takes that measurement and completes it: it recovers
the reuse dependencies, assembles the certificate, and re-simulates under
those dependencies. This is the half that produces what the runtime enforces,
and it costs about as much again as the measurement.

A caller that only wants an admitted layout or an error calls
`build_fixed_layout_admission`, which is exactly the composition of the two.
A caller searching over schedules measures many and certifies only what it
keeps.

The admitted layout is pointer-free. Runtime materialization translates its
semantic IDs to contiguous task, action, and object indices only after the
execution plan has been resolved.

## Capacity accounting

Let:

- $B_e$ be the public execution-memory budget;
- $F$ be the process-persistent execution bytes excluded before callable
  admission — the driver's baseline and provider-owned state, itemized by
  field under [two things called
  dynamic](admission-leases.md#two-things-called-dynamic);
- $H$ be runtime-global shared execution-resident bytes;
- $P=B_e-F-H$ be the callable's physical execution-pool capacity;
- $C$ be the logical object capacity presented to the search;
- $B_s$ be the public spill-memory budget.

`AdmissionFacts` records both $P$ and $C$. The difference $P-C$ is capacity
leeway (`pytorch.planning.common.capacity_leeway`). A search subtracts each
selected task's actual workspace at that task's boundary; neither it nor the
leeway is a monolithic workspace partition reserved at every boundary.

The leeway exists because the two stages bound different quantities. A search
bounds *instantaneous* occupancy: composing the capacity with the per-boundary
workspace subtraction gives

\[
\text{objects}(b) + \text{workspace}(b) \le C \quad\text{at every boundary } b,
\]

while admission must place a *fixed-offset extent* $L$ that is larger whenever
overlapping lifetimes prevent two leases from sharing an offset. A layout whose
excess $L - C$ fits inside the leeway is admitted with no refinement attempt;
anything larger is what [capacity refinement](#capacity-refinement) resolves
during the search.

`PhysicalAdmission.workspace_reserve_bytes` is a separate quantity: the
contiguous workspace allowance the pool must be able to serve. It is validated
against the slab and reported, but it is not subtracted from $P$ and does not
define $C$. The leeway is derived from that allowance — it is whatever the
allowance holds above the peak task workspace, a quarter of that peak under the
default 5/4 policy, with a 512 MiB floor and 2 MiB rounding that bind instead
of the ratio for small peaks. The excess the leeway absorbs is a property of
lifetime overlap rather than of workspace, so the derivation bounds it only
approximately.

For one fixed layout, let:

- $L$ be the reusable fixed-slice extent;
- $D$ be the simultaneous terminal dynamic-output reserve;
- $S$ be the bounded dynamic-scratch reserve.

Physical execution admission requires

\[
L + D + S \le P.
\]

`fixed_slice_bytes`, `dynamic_reserve_bytes`, `scratch_reserve_bytes`,
`required_bytes`, and `slack_bytes` expose every term directly. The fixed
slice is one parent range, and its tail is the resident slice
(`resident_slice_bytes`) holding the objects the planner kept resident,
described in [fixed-offset placement](fixed-placement.md#the-resident-slice);
$D$ and $S$ remain available in the rest of the execution pool for ordinary
dynamic leases.

Spill admission is separate:

\[
\operatorname{simulated\_spill\_peak} \le B_s.
\]

The spill allocator receives no fixed per-object offsets. Its dynamic pool
enforces the cap while the simulator and runtime action ledger check the
selected spill traffic and residency.

## Admission facts

`AdmissionFacts` is the framework-neutral, immutable physical description
shared by candidate evaluation and layout construction. Each
`TaskAdmissionSpec` describes:

- the simultaneously-live anonymous allocation extents derived from its
  allocation trace;
- fresh persistent output aliases;
- mutation-replacement aliases;
- zero-copy storage handoffs;
- the complete ordered allocation/free trace observed for that executable
  profile.

Allocation steps contain charged bytes, an allocation ordinal, optional output
ownership, and any same-task ordinal reuse. They contain no observed pointer
and no planned offset.

There is no synthetic physical sequence. Creating an executable set of facts
requires an explicit trace for every structural profile, including an
explicitly empty trace for a task that performs no allocation. Workspace
extents are computed from the trace after persistent outputs and replacements
are identified; returned but unretained tensors therefore remain anonymous
workspace for their real lifetime. A nonempty workspace, fresh output, or
replacement without corresponding allocation steps fails before the search
runs.

A hand-authored logical `ShadowSpillProgram` can still be planned and simulated
without an `AdmissionFacts`. It becomes executable only after a frontend
provides complete physical evidence. The serialized facts use the
`shadowspill.admission_facts/v1` schema; anything else is rejected.

### What every problem embeds

Certifying a schedule, digesting it, and replaying it through the pool are the
same questions whichever search produced the schedule, so those entry points
take a `ShadowSpillScheduleContext` rather than any search's own problem
struct. It carries the machine (`simulation`), the facts a schedule must fit
(`admission`), the same facts supplied for placement alone (`placement`), and
the alias and task names identifiers are written under.
`ShadowSpillIndexedProblem` is the generic problem a search is handed, and it
embeds one context, so a search passes the context down and generic code never
sees the search's own structures.

`admission` and `placement` hold the same facts and are deliberately separate
fields: supplying `admission` switches on the dynamic-pool replay, which is a
stricter question that rejects schedules a dependency-certified fixed placement
accepts. A search prefiltering through it would discard plans that would have
run. `placement` is the same description without that prefilter; NULL leaves
plans unplaced, which is how a caller opts out of measuring layouts during a
search.

## From a schedule to physical lifetimes

### Two vocabularies: actions and operations

Two different things are ordered in causal order here, and keeping them apart
matters when reading anything below.

A **memory action** is a decision the *plan* makes about moving an object:
`fetch`, `write_back`, `evict`, or `release`, each triggered at a task
boundary. Actions belong to the `MemorySchedule` and are what a search chooses
([IR](ir.md#memory-schedule)). Admission accounts for the three that move or
free execution memory; a write-back takes a spill destination and frees
nothing, so it changes no execution lease.

A **pool operation** is an allocator call that *executing* the plan implies:
`RESERVE`, `ACQUIRE`, `ACQUIRE_RESERVED`, `BEGIN_RETIREMENT`,
`COMPLETE_RETIREMENT`, `RELEASE`, `PUBLISH_DEPENDENCY`. Operations belong to the
admission script and are what the `MemoryPool` policy replays.

The relationship is one-to-many and the two vocabularies share no kind names.
There are several operations to every action, because a single fetch
reserves its destination, acquires the reserved range, and later begins and
completes a retirement, while a task also acquires and retires leases without
any action at all.

Everything downstream keeps the distinction: `action_index` always identifies a
memory action, a lease is created and retired by operations, and the reuse
diagnostics report an operation's `purpose` beside the `action_kind` that
triggered it. `csrc/src/planner/admission/operations.c` derives the operations;
`candidate.c` replays them.

### The script

How a schedule becomes leases - what is fixed before the walk starts, where
each operation sits, why each lease exists, and the two transitions that
happen without emitting an operation - is specified in
[from a resolved program to leases](admission-leases.md). The summary below is
the shape of the walk.

The admission script applies the complete selected step in causal order:

```text
initial execution objects
    -> validate each task's resident inputs
    -> acquire that task's allocation-core leases
    -> begin task-completion retirements for anonymous temporaries
    -> publish output, mutation-replacement, and handoff ownership
    -> trigger release, write-back, eviction, and fetch actions in schedule order
    -> complete pending task and transfer retirements
    -> validate final execution residency
```

Every acquired or reserved range receives a stable lease ID and a semantic
purpose. [Why a lease exists](admission-leases.md#why-a-lease-exists) is the
complete taxonomy; what matters here is how long each kind occupies bytes:

| Purpose | Birth | End of physical lifetime |
|---|---|---|
| Initial object | Initial residency | Its scheduled release, replacement, or eviction completion |
| Task workspace | Task start | Task completion |
| Task output | Producing task start | Release, replacement, eviction completion, or terminal ownership |
| Mutation replacement | Mutating task start | Later replacement/release/eviction or terminal ownership |
| Fetch destination | Fetch trigger boundary | Later release/replacement/eviction or terminal ownership |

An eviction source remains live through copy completion. A task-local free
begins retirement but does not make bytes physically reusable before the task
completion fence. A fetch destination begins consuming capacity at its action
trigger, even if earlier FIFO work delays wire submission.

`build_lease_lifetimes()` combines this causal script with the selected
simulation's task and transfer intervals. Predicted intervals guide compact
placement; causal operation boundaries prove whether reuse is legal.

## Fixed placement

For every non-dynamic lease $l$, define:

- size $s_l>0$;
- alignment $a_l>0$;
- predicted half-open lifetime $[t_l^0,t_l^1)$;
- causal birth and retirement boundaries $[q_l^0,q_l^1]$;
- relative offset $o_l$ in the callable slice.

Every placement must satisfy

\[
o_l \equiv 0 \pmod {a_l}
\]

and

\[
0 \le o_l, \qquad o_l+s_l \le L.
\]

`shadowspill_place_lifetimes` in the planner library solves this. The
assignment, the structure that chooses each offset, and the two numbers that
judge the result are specified in
[fixed-offset placement](fixed-placement.md).

If two predicted lifetimes overlap, their byte ranges must be disjoint:

\[
[t_i^0,t_i^1)\cap[t_j^0,t_j^1)\ne\varnothing
\Longrightarrow
[o_i,o_i+s_i)\cap[o_j,o_j+s_j)=\varnothing.
\]

The placer is deterministic and never sees lease identity. It assigns the
lowest aligned gap not occupied by an overlapping lifetime, and the maximum
range end is $L$. This is a packing heuristic: admission proves its result
fits but does not claim that $L$ is the minimum possible extent over every
placement.

Predicted time alone is never the safety proof. After placement, admission
examines each physical range in causal birth order. Whenever a successor
overlaps bytes previously owned by another lease, it requires:

1. the predecessor's causal retirement precedes the successor's acquisition;
2. the retirement owns a concrete completion dependency;
3. the dependency can be resolved to a runtime task/action identity.

A missing proof rejects the layout. Runtime timing may therefore be slower or
faster than simulation without making shared-address reuse unsafe.

## Causal reuse dependencies

Most task-allocation reuse is already ordered by the single compute stream.
Cross-lane reuse needs an explicit edge. The physical projection emits
`MemoryReuseDependency` values when an eviction-completion event protects a
range later used by:

- a task allocation; or
- another fetch destination.

The simulator inserts the same predecessor-completion edge before computing
the admitted makespan. The runtime resolves it to the predecessor action's
event. If the event is published but unfinished, it inserts a compute- or
transfer-stream wait; it does not synchronize the host until completion. If
the action has not yet published its event, the foreground path asks the
worker to progress and waits only for that event identity to become available.

For a task allocation the foreground path is the task boundary, which resolves
every dependency the plan pinned to that task before the task is marked
started. The allocator resolves the same dependency again where the allocation
happens and finds it already published. Resolving it at the boundary is what
keeps the wait outside the task's own compute span, where a stall cannot be
told apart from computing; see
[task boundaries](task-boundaries.md).

The dependency is generation-specific. A completion from another invocation
or an old lease cannot authorize reuse.

## Offset vocabulary

Several offsets coexist, but they are not interchangeable:

| Offset | Coordinate system | Meaning |
|---|---|---|
| Runtime slice offset | Execution `MemoryPool` | Where the callable's one parent fixed slice was dynamically reserved. |
| Fixed placement offset | Callable fixed slice | Where one admitted lease lives relative to the parent slice. |
| Absolute slab offset | Execution `MemoryPool` | `slice_offset + placement_offset`; used internally and in allocator telemetry. |
| Tensor view offset | One semantic storage root | Byte/element displacement of a view from its root; defined by `TaskStorageContract`. |
| Spill-pool offset | Spill `MemoryPool` | Dynamic lease position selected by the spill allocator; not fixed by this certificate. |

Physical layout never changes tensor view semantics. All leaves that share one
storage root bind to the same lease, and their view offsets are reconstructed
relative to that lease's base pointer.

The runtime does not assume that the callable slice begins at slab offset
zero. It reserves one compatible parent range from the runtime-owned execution
pool, then adopts borrowed subleases at the certified relative offsets. A
borrowed fixed lease does not independently own or free the parent range.

## Strict core and dynamic exceptions

### Allocation-core slots

Each fixed task allocation is addressed by `(execution_task_id,
allocation_ordinal)`. Runtime allocation callbacks reconcile the observed
operation stream against `TaskAllocationContract`, then validate requested bytes
and alignment before returning the planned subrange. Required output and
mutation allocations cannot be silently omitted or replaced by scratch.

### Dynamic scratch

Optional anonymous/provider operations observed in some allocation paths may
be inserted or omitted at runtime. They receive no fixed offset. Instead they
use the ordinary dynamic range allocator outside the fixed slice and are
bounded per task by:

- maximum individual requested and charged allocation;
- maximum live requested and charged bytes;
- maximum dynamic-scratch allocation;
- maximum live dynamic-scratch bytes.

The global scratch reserve is the maximum admitted need because execution
tasks are sequential. It is derived from profiled path observations with 25%
headroom rounded to 2 MiB, and may be increased by the public planning
request. The request cannot reduce the measured minimum.

### Terminal caller-owned outputs

Only the final execution generation of an alias returned to the caller is
excluded from the reusable fixed slice. It uses a dynamic lease so the caller
may retain that tensor across a later invocation. Earlier generations of the
same alias remain ordinary fixed lifetimes.

### Persistent provider state

Process-persistent provider state is measured and removed from $P$ before the
layout is built. It is neither task workspace nor dynamic scratch. Its
allocator operations may still carry a dynamic task-allocation policy so a
valid observed first-use path can be checked without assigning provider-owned
state a callable-relative address.

## Runtime adoption and validation

Runtime adoption has two cold-path phases:

1. `shadowspill_plan_admit_fixed_layout()` copies the pointer-free certificate
   and reserves the parent execution-pool slice.
2. After objects and immutable task records exist,
   `shadowspill_plan_seal_fixed_layout()` resolves every placement and
   dependency.

Sealing verifies:

- placement identities are unique;
- every fixed range fits the slice and satisfies alignment;
- every strict-contract allocation and fetch action has exactly one fixed or
  dynamic policy;
- task allocation bytes/alignment match the task record;
- action destinations name the expected object and size;
- every dependency names an admitted eviction and a real successor.

At runtime, a missing placement, wrong allocation path, stale invocation,
out-of-envelope scratch request, or unresolved dependency is a plan violation
or task-attributed allocation failure. It is rejected before an invalid
pointer reaches a backend kernel.

## Capacity refinement

A schedule may satisfy logical boundary capacity yet require a fixed extent
larger than $P$: boundary capacity bounds what is live at an instant, while the
extent is what the address assignment spans. The second is the constraint that
decides whether a plan can run, so a search measures it itself rather than
leaving it to a later layer.

With `placement` facts in hand a search, on reaching a plan that could still
win:

1. derives the operations the schedule implies and their lease lifetimes;
2. places those lifetimes and reads back the extent, adding the leases that
   outlive the step, since the certificate charges for those too;
3. if that fits the pool, offers the plan to the shared best-placed record;
4. otherwise gives capacity back and plans again.

Capacity is therefore a property of a plan, not of the search: two plans can
be answered at different capacities in the same call, and neither one's
shortfall costs the other anything. A plan with no layout cannot run whatever
its makespan, so a search that never placed one has no answer to give.

The simulator always runs at the capacity the caller described, because that is
the machine the plan will run on: timed at anything else, its makespan and the
lease lifetimes derived from its timeline belong to a plan that will never
execute, and the certificate would disagree with the search that chose it.
Reduced capacity therefore reaches only the stages that *shape* a plan, never
the ones that *measure* it.

How much a plan gives back at a time is a trade. The extent does not fall byte
for byte with the capacity — a cut in capacity can move it by more — so handing
back everything a layout overran overshoots the capacity that would have fit,
and the plan built below that capacity is materially worse than the one just
under the line. Stepping costs rounds and buys quality; handing back the whole
shortfall converges in the fewest rounds, which is what a caller wants when
planning time matters more than the last of a plan's quality. How much to give
back is the search's own setting.

The shared best-placed record is what keeps this affordable. Placing a plan
costs far more than simulating one, so a plan whose makespan is already no
better than a plan someone else has placed is never measured. The record is
passed in, so its scope is the caller's choice: one call, or every resolved
program in a search.

`resolve_fixed_layout_selection` then certifies a single capacity. It has no
ladder to walk, because the plan it receives has already been measured against
this pool, at the same capacity and from the same timeline; a rejection there is
a disagreement between the search's measurement and the certificate rather than
a capacity to retry.

## Diagnostics

`PlanReport.diagnostics` carries one `PlanPhysicalLayout` per recurrent or
initialization plan: the capacities, the bytes each term of $L + D + S \le P$
required, the placements and their offsets, the reuse dependencies, the task
memory envelopes, the digests, and the logical and physically admitted
simulations side by side. Its `attempts` hold one `PlanFixedLayoutAttempt`,
since the search settles capacity and leaves nothing here to walk, with the
search and physical-admission wall times reported separately. Every field is
defined in [physical-layout
diagnostics](../python/plan-report.md#physical-layout-diagnostics).

When runtime tracing is enabled, `StepDiagnostics` adds actual allocation
offsets, peaks, waits, scratch usage, failures, and execution-task identity.
Tracing is default-off and does not add allocator telemetry work to the normal
critical path.

## Guarantees and limits

Physical admission guarantees for the admitted fixed-shape contract:

- exact execution and spill budget checks;
- no overlapping fixed ranges without causal ordering and a completion proof;
- fixed task/action identities validated before execution;
- trigger-time fetch destination capacity;
- runtime failure before unsafe use when allocation behavior exceeds its contract
  or envelope.

It does not prove an unseen, data-dependent allocator path that was absent
from the admitted core and exceeds dynamic scratch. Fixed-shape guards and
stable required output/mutation allocation behavior remain part of the
supported contract.

## Implementation map

| Module | Responsibility |
|---|---|
| `shadowspill.planner.admission` | Immutable framework-neutral task and pool facts. |
| `shadowspill.planner.admission.setup` | Compile the schedule-invariant half of one resolved program. |
| `shadowspill.planner.admission.admission_replay` | Build the timing-free causal step script and ownership transitions. |
| `shadowspill.planner.admission.layout.lifetimes` | Combine causal operations with selected task/transfer intervals. |
| `shadowspill.planner.admission.placement` | Deterministic aligned interval placement. |
| `shadowspill.planner.admission.layout.dependencies` | Prove shared-range reuse and project cross-lane simulator edges. |
| `shadowspill.planner.admission.layout.build` | Measure a layout, certify it, or both. |
| `shadowspill.planner.admission.refinement` | Certify the fixed layout of the plan the search placed. |
| `shadowspill.pytorch.planning.admission.layout_runtime` | Translate semantic placements to indexed runtime identities. |
| `csrc/src/planner/admission/` | Operations, lifetimes, placement, and the pool replay, over a `ShadowSpillScheduleContext`. |
| `csrc/src/runtime/plan/fixed_layout.c` | Reserve the parent slice, seal identities, adopt subleases, and insert dependency waits. |
| `csrc/src/runtime/memory/memory_pool.c` | Own dynamic ranges outside the fixed slice and enforce physical accounting. |

Previous: [PressureFit](pressurefit.md). Next:
[Planning orchestration](planning.md).
