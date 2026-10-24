# Memory inputs and external profiles

The user clarified that external profiles may include explicitly supplied
estimates. Coverage must be complete, but values need not all come from GPU
measurement. There is no implicit fallback that invents missing values.

Static graph/storage structure belongs to `Program`; task timings and dynamic
memory evidence belong to `Profiles`. The user selected one combined record per
compatible task for timing and missing memory information, preserving the source
of each supplied value. Measured timing may coexist with explicitly estimated
memory. Timing and memory fields use plain numbers, as requested by the user;
source/classification information uses separate plain metadata rather than value
wrapper classes. When combining compatible records, later supplied values
overwrite earlier conflicts by default and fields omitted by the later record
are preserved. Each collection describes one hardware/runtime setup, while
allowing many tasks, shapes, variants, ranks, and measurement/import runs. Reuse
and merging still require compatible tasks and setup assumptions. Exact fields,
metadata, and execution-status reporting below remain to be agreed. The user
clarified that memory estimates are for simulation only; a plan relying on them
cannot execute.

## Information needed for a memory-aware simulation

| Information | Source/responsibility |
| --- | --- |
| Sizes and identities of tensor storage | Build derives shapes, dtypes, strides, aliases, inputs, outputs, and mutations. Use storage extents, not a sum over tensor views that may share storage. |
| State and lifetime constraints | Build identifies weights, persistent gradients, optimizer/master state where present, saved activations, dependencies, and required entry/exit state. Planning then determines placement and transfer timing. |
| Shard ownership | The parallelism description determines full execution storage versus local spill storage and required communication. |
| Temporary workspace for each unique task/variant/shape | Build derives compiler-visible allocation sizes/lifetimes. Provider declarations, measurements, or explicit estimates fill remaining gaps beyond program-managed objects. Forward and backward need separate values; save and recompute can differ. |
| Physical storage refinements | Compiler manifests and, where necessary, profiles describe padding, alignment, actual output allocations, aliases, or off-device outputs that affect memory accounting. |
| Additional persistent/fixed memory | Provider caches, communication buffers, runtime overhead, and bounded scratch reserves must be counted in the appropriate capacity accounting, without charging shared storage once per task. Exact public ownership remains open. |

The largest missing quantity in the isolated GLM traces is task-specific
workspace and physical allocation evidence. The static model/state sizes and
saved-value structure should be derived by build, rather than entered manually
for every stage. Omitted stage/optimizer timing entries also need supplied values.

## Workspace is not the whole-device peak

A useful workspace value counts temporary memory beyond the inputs, persistent
state, and outputs already represented in the program. It must not charge those
same objects a second time. Temporary gradient contributions or functional
mutation results can also increase the live set, depending on whether they are
retained as program objects or consumed into existing state.

The current implementation distinguishes requested bytes from allocator-charged
bytes and records allocation order. It uses that order to account for transient
outputs whose lifetimes do not necessarily overlap the anonymous workspace peak.
Adding the sizes of all temporary tensors overstates peak live memory.

A whole forward/backward pair peak cannot by itself establish these quantities.
It combines inputs, saved values, gradients, outputs, and temporaries, and does
not locate their individual peaks. Allocator-reserved memory is also different
from live tensor memory. In particular, pair peak minus input sizes is not a
reliable backward-workspace measurement.

## Memory while standalone DP communication overlaps compute

Scope clarification: the user prioritizes representative single-GPU DP profiling
without live groups. That does not require independent collective/compute
overlap. The API draft recommends initially using ordered communication
boundaries and retaining the following requirements for a later overlap
extension. No asynchronous scheduler redesign is selected by this discussion.

The proposed separation of reductions from optimizer code must preserve physical
allocation and completion accounting. An explicit compute order does not imply
that communication launched earlier on another stream has finished. The Plan
must include its buffer lifetimes and completion dependencies, not just its
duration or the size of the final gradient shard.

For a reduction overlapping later backward tasks, account for:

- The full input gradient, retained until the collective has finished reading it
  and any other consumers are done.
- The reduced output shard, allocated before writes begin and retained until its
  optimizer/other consumers complete or it is safely spilled.
- Any communication scratch, packing/redistribution buffers, and provider
  reservations, without charging persistent provider overhead twice.
- Concurrent backward inputs/outputs/workspace and prefetched master/moment
  shards. Peak memory is the union of live physical allocations across lanes.

For example, reducing a 1 GiB gradient over eight ranks into a separate 128 MiB
output requires 1.125 GiB for those two buffers while in flight, before counting
other work or collective scratch. If the output aliases the input allocation,
count its physical backing once, but do not claim the unused portion is freed
unless the allocation/reuse contract actually permits that. A small view alone
does not shrink its backing allocation.

Current source provides part of the needed machinery: simulator eviction
accounting releases device bytes at transfer completion, and physical admission
tracks delayed retirement/reuse. However,
`planner/admission/simulation.py::_simulation_dependency` currently relies on
compute-stream ordering for task-completion reuse and exports explicit simulator
reuse dependencies for eviction predecessors. Standalone asynchronous collective
completion must be represented through admission, placement, simulation, and
runtime; labeling a call as a communication task is not sufficient validation.

The runtime must enforce saved completion/reuse dependencies with actual events,
even when a collective finishes later than its simulated duration. If the
overlapping live set does not fit, planning must select less overlap, different
residency/order, or report failure. It must not admit a schedule by freeing the
input at collective submission or by budgeting only the reduced shard. Tests
for the refactor should include deliberately delayed collectives, overlapping
backward workspace, output/input aliasing, and constrained budgets. This path
has not been implemented or validated; these are required correctness conditions
if standalone collectives are allowed to overlap later compute tasks. They are
not prerequisites for measuring the separated local optimizer computation.

Sources: [simulator transfer completion](../../../../csrc/src/simulator/transfers.c),
[admission dependency projection](../../../../src/shadowspill/planner/admission/simulation.py),
and [runtime allocation reuse](../../../../csrc/src/runtime/memory/allocations/reuse.c).

## Deriving memory from compiled tasks

Yes: once the compiled allocation schedule is fixed, compiler-visible temporary
allocations can often be sized and their lifetimes analyzed on CPU. The analysis
must use compiled buffer allocations/reuse, not sum exported FX intermediates:
fusion can eliminate intermediates, and different logical buffers can reuse the
same physical storage at nonoverlapping times.

The current adapter already captures Inductor's `GraphLowering` and derives
output root extents, views, aliases, and mutations. Its
[`_graph_lowering_contract`](../../../../src/shadowspill/pytorch/compilation/inductor/contract.py)
projects returned buffers, and
[`_build_executable_roots`](../../../../src/shadowspill/pytorch/compilation/inductor/roots.py)
sizes those roots. It does not currently publish the full compiler-visible
temporary allocation/free/reuse schedule in the task manifest.

Extending that extraction can provide a static workspace requirement for the
visible portion. Peak live temporary bytes are the maximum concurrent live
storage after accounting for reuse, with inputs, retained outputs, and separately
accounted persistent state excluded. Allocation alignment and any padding still
matter. Registers and on-chip shared memory are performance/resource constraints,
not additional allocations from the HBM execution pool.

Allocations inside opaque custom operations or library calls may not appear in
that schedule. Such scratch is not necessarily inherently dynamic: a provider
could expose a shape/kernel-dependent workspace requirement. In the absence of
that information, the analysis must identify the unresolved operator instead of
reporting zero scratch or calling a partial bound an exact total. Data-dependent
allocation paths and first-use persistent caches may still need a supplied bound
or observation.

This CPU analysis assumes a compiled schedule/kernel configuration is available.
Whether a particular provider can produce that configuration entirely without
a GPU is a separate build/compilation question.

Agreed responsibility: `Program` retains statically
derived memory requirements and explicit unresolved entries; `Profiles` provides
timing and memory values that remain dynamic or unknown. Measured refinements
and caller-supplied estimates keep their provenance. No compiler extraction has
been implemented in this design discussion.

## Measurement without ShadowSpill pools

The target is for all work before execution, including GPU measurement and CPU
planning/search, to be independent of ShadowSpill runtime pools and allocator
installation. The user subsequently allowed allocator-backed profiling if needed
to preserve admission/correctness information. This exception applies only to
measurement, not to build, profile application, planning, or search. A planning
memory budget describes a capacity; it does not reserve that capacity on a device
or host.

The current profiler does use the runtime: `TaskProfiler` constructs an
`AllocatorBoundary`, timing uses runtime event wrappers, and workspace observation
reads allocation telemetry and maps pointers to slab allocations. Removing its
constructor's runtime argument alone would not satisfy the boundary.

Propose ordinary CUDA events for task timing and an independent allocation
observation path for memory. Compiler manifests remain the first source of static
requirements. PyTorch allocation history is a candidate for observing allocation
sizes and lifetimes through its allocator, while provider declarations or explicit
external observations cover other allocations. The exact backend remains open.
PyTorch documents that its memory snapshots do not see direct CUDA allocations,
including common NCCL allocations; aggregate device usage cannot identify their
task-local lifetimes by itself. See
[PyTorch memory visibility](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html).

Pool-specific padding/placement and execution allocation-path validation must not
be presented as measurements made by the ordinary allocator. Static descriptions
can still support CPU placement calculations. Checks that require the real
ShadowSpill allocator move to execution startup in the fresh process. How those
checks report or resolve a mismatch is an open execution-interface decision.

This also requires bounded per-task input/saved-value materialization instead of
keeping the whole model on GPU or copying all saved tensors into unbounded host
memory. The existing stage trace harness demonstrates ordinary-allocation task
execution, but its pair-level peaks do not replace the required memory analysis.

### Native allocation observation: source-verified feasibility

Della's `shadowspill` environment reports PyTorch `2.13.0+cu132`, source revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`. Its native CUDA caching allocator exposes
`attachAllocatorTraceTracker`. This attaches an observer while retaining the
normal allocator. It does not require installing a replacement allocator or
creating ShadowSpill pools. These are C++/version-specific hooks, not a promised
stable Python interface.

The matching source emits `ALLOC` with the original unrounded request size and
returned address, device, stream, and memory-pool identifier. It emits
`FREE_REQUESTED` for the logical free and `FREE_COMPLETED` when the native cache
can reuse the block. The observer runs even when full stack/history recording is
disabled. Callbacks run under allocator locking and must not call PyTorch or
acquire Python's GIL; a small bounded native event recorder is the proposed
implementation, with explicit overflow detection.

Sources: [observer interface](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.h#L242-L248),
[requested allocation size](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp#L2142-L2153),
[logical free](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp#L2432-L2439),
[completed free](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp#L3505-L3519),
and [observer dispatch](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp#L4436-L4471).

Intercepting only `cudaMalloc` would miss tensor allocations satisfied by cached
blocks. Patching Python tensor constructors would miss allocations inside ATen,
Inductor, and C++ providers. The caching-allocator observation point is therefore
the useful candidate.

| Needed information | Proposed collection/derivation |
| --- | --- |
| Allocation request sequence and exact requested bytes | Native allocator observer; assign task-local ordinals and distinguish generations when addresses are reused. |
| Logical frees and deferred native retirement | Retain both event types and streams. Native retirement behavior is an observation, not a substitute for execution-stream dependencies. |
| Output storage, views, input aliases, and mutations | Inspect input/output storage pointers and offsets while live; reconcile with compiler storage contracts. Replace the slab-specific pointer lookup. |
| Temporary and persistent allocation lifetimes | Replay scoped observations, preserve output lifetimes, and retain warmup/provider-cache observations. Do not infer these from a single peak counter. |
| ShadowSpill charged bytes and alignment | Apply the target allocator's declared charging/alignment rules on CPU, instead of treating native cache block rounding as ShadowSpill's requirement. The current PyTorch adapter requests 256-byte alignment. |
| ShadowSpill offsets, permitted reuse, and physical fit | Compute from the logical allocation contract and plan using ShadowSpill placement rules. Native cache addresses/reuse do not define the execution layout. |
| Direct CUDA/provider allocations outside PyTorch | Provider declarations or separate observations; neither the native hook nor ShadowSpill's PyTorch allocator automatically sees them. Keep unresolved coverage explicit. |

The existing `TaskAllocationContract` already strips process pointers and slab
offsets. However, its steps include charged bytes/alignment, and workspace replay
uses charged extents. Consequently, a native trace cannot be copied unchanged
into the current profiler result. Request observation and backend-specific memory
charging need separate responsibilities before the observer can substitute for
the current telemetry.

No GPU equivalence test or observer implementation has been performed. Propose
qualification against the existing profiler on identical compiled tasks/inputs:
views and shared outputs, mutations, temporary allocations, gradient accumulation,
retained provider state, and stream-dependent lifetimes, followed by representative
GLM KDA/MoE/optimizer tasks. Compare requested allocation contracts, output bindings,
derived target-charged peaks, and actual admitted execution. Include supported
CUDA graph/private-pool behavior where relevant. Keep existing execution checks
enabled; missing events or unsupported observation coverage must be reported.

If this cannot preserve a required contract, the user's authorized fallback is
allocator-backed measurement. The public phase separation and CPU planner must
remain intact either way.

## Selected simulation and execution distinction

For CPU simulation, a caller should be able to supply task durations and
workspace estimates together with the static storage descriptions and any
required explicit physical refinements. The result should retain which inputs
were estimated. Such a call should not require fabricated CUDA allocator traces
or fabricated measurement samples.

The current public planner also performs physical allocation admission. That
coupling needs an explicit design change to support estimated simulation inputs.
Detailed allocation contracts remain relevant when validating a plan for actual
execution, where merely knowing a peak byte count does not establish correct
allocation sizes, alias bindings, offsets, or allocation/free order.

The user explicitly requires memory estimates to remain simulation-only. A plan
whose selected tasks/resources rely on them can be inspected and saved with its
timeline, but execution must reject it. Obtain complete non-estimated memory
information from compiler/provider descriptions or profiling, replace the
estimates, and plan again before execution. Do not add worker-side measurement
or replanning to promote such a plan. Exact reporting fields and worker binding
remain to design; the public object model still uses one Plan type.

This boundary does not require GPU measurement for statically established memory
requirements, nor does it make a sampled peak sufficient for unknown allocation
behavior. Complete allocation details and supported bounds/guards are still
needed. Timing and transfer-cost estimates remain separate performance inputs.

## Existing source references

- [TaskMeasurement](../../../../src/shadowspill/task/profiles.py) contains runtime,
  workspace, output bindings, provider-persistent extents, allocation traces, and
  path observations.
- [ExecutableTaskManifest](../../../../src/shadowspill/task/manifest.py) contains
  compiler-owned storage contracts and root allocation extents.
- [CompiledTaskLayout](../../../../src/shadowspill/task/layout.py) reconciles
  physical layouts and accounts for transient outputs from allocation lifetimes.
- [Lowering profiles](../../../../src/shadowspill/pytorch/lowering/profiles.py)
  constructs planner task profiles and resolves storage contracts.
- [Training program construction](../../../../src/shadowspill/pytorch/planning/training/programs.py)
  currently constructs simulation inputs and physical admission facts together.

No runtime or planner implementation is changed by this document.
