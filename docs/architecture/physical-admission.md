# Physical admission and offset handling

Physical admission proves that one selected logical schedule can be assigned
real ranges inside its configured memory pools without unsafe overlap. It runs
after [PressureFit](pressurefit.md) has selected tasks, residency, and memory
actions, and before an executable callable is published.

PressureFit answers:

> Which objects should be resident, and when should release, evict, and fetch
> actions trigger?

Physical admission answers:

> Which exact execution-pool bytes back every selected object generation,
> task allocation, and fetch destination, and what completion proof protects
> each reused range?

The current PyTorch path uses a fixed layout for schedule-managed execution
memory plus bounded dynamic regions for caller-owned terminal outputs and
optional scratch. The spill pool remains dynamically allocated and is checked
against its simulated physical peak.

Runtime-global shared leases lie outside a callable's physical layout. Their
bytes are subtracted before constructing the callable-owned pool slice, while
diagnostics preserve both the shared footprint and residual movable capacity.
Admission never assigns a second offset or transfer destination to a shared
alias.

## Inputs and output

Physical admission consumes:

| Input | Purpose |
|---|---|
| `PressureFitResult` | Selected tasks, residency, ordered actions, and logical simulation. |
| `AdmissionTopology` | Pool capacity, task allocation geometry, output/replacement ownership, handoffs, and alignment. |
| `TaskAllocationContract` values | Stable task-local core allocation/free identities and geometry. |
| Dynamic-scratch reserve | Bounded capacity for optional allocator operations outside the strict core. |
| Terminal caller-owned aliases | Final execution leases that may outlive a later callable invocation. |

It returns `FixedLayoutAdmission`, which contains:

- a `FixedPhysicalLayout` with placements and causal reuse dependencies;
- a `SimulationAdmission` projection of the physical certificate;
- a new `SimulationResult` that includes the added dependencies;
- stable digests tying the layout to its Program, schedule, and topology.

The admitted layout is pointer-free. Runtime materialization translates its
semantic IDs to contiguous task, action, and object indices only after the execution
plan has been resolved.

## Capacity accounting

Let:

- $B_e$ be the public execution-memory budget;
- $F$ be process-persistent execution bytes excluded before callable
  admission, including initialized provider state and profiled retained
  provider/custom-operation growth;
- $H$ be runtime-global shared execution-resident bytes;
- $P=B_e-F-H$ be the callable's physical execution-pool capacity;
- $C$ be the logical object capacity presented to PressureFit;
- $B_s$ be the public spill-memory budget.

The `AdmissionTopology` records both $P$ and $C$. The difference $P-C$ is
logical allocator/workspace leeway. PressureFit subtracts each selected task's
actual workspace only at that task boundary; it does not reserve a monolithic
workspace partition at every boundary.

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
slice is one parent range; $D$ and $S$ remain available in the rest of the
execution pool for ordinary dynamic leases.

Spill admission is separate:

\[
\operatorname{simulated\_spill\_peak} \le B_s.
\]

The current spill allocator does not receive fixed per-object offsets. Its
dynamic pool enforces the cap while the simulator and runtime action ledger
check the selected spill traffic and residency.

## Admission topology

`AdmissionTopology` is the framework-neutral, immutable physical description
shared by candidate evaluation and layout construction. Each
`TaskAdmissionSpec` describes:

- the exact anonymous live-set peak derived from its allocation trace;
- fresh persistent output aliases;
- mutation-replacement aliases;
- zero-copy storage handoffs;
- the complete ordered allocation/free trace observed for that executable
  profile.

Allocation steps contain charged bytes, an allocation ordinal, optional output
ownership, and any same-task ordinal reuse. They contain no observed pointer
and no planned offset.

There is no synthetic physical sequence. Creating an executable topology
requires an explicit trace for every structural profile, including an
explicitly empty trace for a task that performs no allocation. Workspace
extents are computed from the trace after persistent outputs and replacements
are identified; returned but unretained tensors therefore remain anonymous
workspace for their real lifetime. A nonempty workspace, fresh output, or
replacement without corresponding allocation steps fails before PressureFit.

A hand-authored logical `Program` can still use PressureFit and the simulator
without an `AdmissionTopology`. It becomes executable only after a frontend
provides complete physical evidence. The serialized topology uses only the
current `shadowspill.admission_topology/v3` schema; older synthetic forms are
rejected rather than migrated.

## From a schedule to physical lifetimes

The admission script applies the complete selected step in causal order:

```text
initial execution objects
    -> validate each task's resident inputs
    -> acquire that task's allocation-core leases
    -> begin task-completion retirements for anonymous temporaries
    -> publish output, mutation-replacement, and handoff ownership
    -> trigger release, eviction, and fetch actions in schedule order
    -> complete pending task and transfer retirements
    -> validate final execution residency
```

Every acquired or reserved range receives a stable lease ID and semantic
purpose:

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

If two predicted lifetimes overlap, their byte ranges must be disjoint:

\[
[t_i^0,t_i^1)\cap[t_j^0,t_j^1)\ne\varnothing
\Longrightarrow
[o_i,o_i+s_i)\cap[o_j,o_j+s_j)=\varnothing.
\]

The deterministic placer orders leases by decreasing size, decreasing count
of distinct simulated timeline boundaries spanned, earlier start, and stable
lease ID. It assigns the lowest aligned gap not occupied by an overlapping
lifetime. The maximum range end is $L$. This is a deterministic packing
heuristic; admission proves its result fits but does not claim that $L$ is the
minimum possible extent over every placement.

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
Cross-lane reuse needs an explicit edge. The current physical projection emits
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

A PressureFit schedule may satisfy logical boundary capacity yet require a
fixed extent larger than $P$. The PyTorch planning orchestrator handles this
monotonically:

```text
object-capacity reductions:
    0 MiB
    256 MiB, 512 MiB, 768 MiB, 1024 MiB
    1536 MiB, 2048 MiB, ...
```

For each reduction it:

1. reruns or restores PressureFit at the lower logical object capacity;
2. rebuilds the complete fixed layout against the unchanged physical pool;
3. re-simulates with physical reuse dependencies;
4. accepts the first layout that fits.

Reducing logical capacity changes the selected residency/actions and leaves
more physical slack; it does not shrink the runtime pool. Every rejected and
accepted attempt records requested/effective object capacity, required bytes,
physical capacity, PressureFit wall time, physical-admission wall time, and
candidate diagnostics.

The framework-neutral `pressurefit()` API can alternatively receive an
`AdmissionTopology` and evaluate the production dynamic-pool policy inside
each candidate. The current PyTorch callable path deliberately uses the fixed
layout builder as its final physical authority: a dynamic best-fit rejection
must not discard a schedule that has a valid dependency-certified fixed
placement.

## Pseudocode

```text
AdmitSelectedSchedule(selected, topology, scratch):
    operations, ownership = build_causal_admission_script(
        selected.tasks,
        selected.schedule,
        topology.task_allocation_geometry,
    )

    lifetimes = build_lifetimes(
        operations,
        selected.simulated_task_intervals,
        selected.simulated_transfer_intervals,
    )

    terminal_dynamic = final_caller_owned_generations(lifetimes)
    fixed = lifetimes - terminal_dynamic

    placements, fixed_extent = deterministic_interval_place(fixed)
    dependencies = prove_every_shared_range_reuse(
        operations,
        placements,
    )

    required = (
        fixed_extent
        + sum(bytes(l) for l in terminal_dynamic)
        + scratch
    )
    if required > topology.pool_capacity:
        reject(required - topology.pool_capacity)

    admitted_simulation = simulate(
        selected.schedule,
        reuse_dependencies=project_cross_lane_edges(dependencies),
    )

    return FixedLayoutAdmission(
        placements,
        dependencies,
        terminal_dynamic,
        scratch,
        admitted_simulation,
    )
```

## Diagnostics

`PlanReport.diagnostics` retains, per recurrent or initialization plan:

- every fixed-layout refinement attempt;
- pool, fixed-slice, terminal-dynamic, scratch, required, and slack bytes;
- placement counts and bytes by semantic purpose;
- complete placements and their relative offsets;
- causal reuse dependencies;
- task memory envelopes and allocation-contract digests;
- layout, Program, schedule, and topology digests;
- logical versus physically admitted simulation results;
- PressureFit and physical-admission wall times separately.

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
| `shadowspill.planner.admission` | Immutable framework-neutral task and pool topology. |
| `shadowspill.pytorch.planning.admission.admission_replay` | Build the timing-free causal step script and ownership transitions. |
| `shadowspill.pytorch.planning.admission.layout.lifetimes` | Combine causal operations with selected task/transfer intervals. |
| `shadowspill.pytorch.planning.admission.layout.placement` | Deterministic aligned interval placement. |
| `shadowspill.pytorch.planning.admission.layout.dependencies` | Prove shared-range reuse and project cross-lane simulator edges. |
| `shadowspill.pytorch.planning.admission.refinement` | Rerun PressureFit at monotonically lower logical capacity until a layout fits. |
| `shadowspill.pytorch.planning.admission.layout.runtime` | Translate semantic placements to indexed runtime identities. |
| `csrc/runtime/src/fixed_layout.c` | Reserve the parent slice, seal identities, adopt subleases, and insert dependency waits. |
| `csrc/runtime/src/memory_pool.c` | Own dynamic ranges outside the fixed slice and enforce physical accounting. |

Previous: [PressureFit](pressurefit.md). Next:
[Planning orchestration](planning.md).
