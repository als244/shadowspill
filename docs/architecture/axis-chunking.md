# Axis chunking within a stage

Chunking executes independent positions of a logical tensor axis in bounded
slices. It reduces temporary activations and recomputation workspace. Full
stage inputs, outputs and saved values still need their full storage.

Models express the full computation. The builder generates chunk loops from
caller policies and operator semantics; providers do not implement these loops.
Any supported tensor axis can define the independent domain. Factored axes
preserve its coordinates through flattening, reshaping and permutation.

## Checkpoints, execution regions and chunks

| Control | What it changes | What remains unchanged |
|---|---|---|
| `partition` | Checkpoint-stage boundaries and their derived inputs/outputs | Operator mathematics |
| Graph-pair retention policy | Values retained from forward until backward | Execution-region boundaries |
| Direct `ChunkPolicy` | Chunking of a complete stage | Its checkpoint boundary and retained cut |
| `ChunkRegions` | Internal execution cuts and per-region chunk policies | The parent graph pair's retained cut |
| `ChunkRegions.lifetimes` | Approved ordering, replay or compaction before execution cuts | The parent task's public storage interface |

A checkpoint stage can contain whole-axis operations and several chunked
regions. Execution-region edges live only within one task call, but a cut can
still materialize a full-axis intermediate. A cut alone does not reduce memory.
The [policy reference](../python/api/frontend.md#axis-chunking-policies) defines
the public controls; [operator integration](../python/advanced/operator-integration.md)
explains the declarations required from a library.

### Compose regions inside an outer slice

`AxisChunks.regions` nests execution regions within an outer chunk loop. For
example, an independent group can use an outer bound of 8,192 positions, with
one inner projection using 2,048 and another operation using 8,192. Their
intermediate edge spans at most the outer slice, rather than the full axis.
Each slice finishes its last consumers before the next begins.

The complete group must pass the outer separability checks. An operation that
needs the whole axis stays outside it. Nesting introduces neither checkpoint
saves nor repeated expensive computation in the explicit graph.

## Construction order

1. Capture the full-stage forward and backward graphs, including gradient
   creation and accumulation forms. Resolve retention and precision conversions.
2. Apply selected lifetime transformations to the complete explicit task.
3. Route eligible parameter-gradient contributions to shared destinations.
4. Resolve execution cuts and chunk policies; propagate axis/reduction roles.
5. Specialize full and tail children, then compile the complete composition.

The builder transforms explicit graphs; it does not differentiate a Python
chunk loop. Policy selectors see source paths for the current checkpoint
occurrence, including its derivative and recomputation operations. Structural
reuse rebinds those paths; generated operations receive no guessed provenance.
See [structural deduplication](graph-pair-construction.md#structural-deduplication).

## Axis and reduction semantics

Built-in rules cover supported elementwise operations, feature reductions,
views and matrix products. Opaque operations require versioned rules. Matching
shapes alone cannot prove independence, and an extent equal to an activation
axis does not make a parameter gradient sliceable.

| Value role | Forward assembly | Backward treatment |
|---|---|---|
| Sliced | Write each slice into the full output geometry | Slice its cotangent for each child |
| Shared/axis-independent | Assemble one result | Supply its cotangent to the first child only; later children receive zero |
| Additive statistic | Sum chunk results | Supply the same upstream cotangent to every child |
| Chunk-local scratch | Keep only within the child | Cannot escape through a boundary or retained cut |

Chunk-local scratch may scale with the domain without preserving original axis
order, as in an internal permutation. A provider rule must restore ordered
coordinates before ordinary operators mix it with sliced values. Nonlinear
aggregates of additive statistics run outside the chunked region.

Sliced inputs use bounded dense copies with canonical offsets and the order
required by captured views. Full and tail children use their actual geometry.
Unknown operations, nonadditive reductions, unsupported effects and ambiguous
aliases are refused. Differentiable aliased stage outputs receive independent
roots so their cotangents stay inside the explicit backward graph.

## Shortening temporary lifetimes

Lifetime processing runs before execution partitioning, when it can prevent a
premature full-axis intermediate from becoming a region edge. The default
`lifetimes=None` leaves instruction order unchanged. Each transformation is
explicitly selected:

| Transformation | Effect | Required proof |
|---|---|---|
| Movement | Delay approved readonly work toward its consumer | Preserve dependencies, mutation versions and protected live backing sets |
| Replay | Copy selected work into consumer regions | Preserve primal provenance, effects, aliases and gradient conversions; never infer permission from cost |
| Compaction | Copy internal views into smaller backing storage | Versioned input-layout semantics, a lower protected peak and no increase at another protected point |

The order is movement, replay, compaction. Movement does not repeat work.
Replay copies are shared within a consumer region and are never inserted into
a region containing a protected barrier. Tuple access accompanies its selected
producer unless explicitly excluded.

Dispatcher schemas establish ownership and read/write dependencies; fake
execution provides geometry, not alias proof. Unknown effects and overlapping
writes block movement. Readonly behavior alone does not authorize relayout.
Compaction follows aliases and consumers until a complete fresh dense result
has unchanged geometry; input/external aliases, written backings and
storage-sensitive consumers remain protected.

Checks count complete backing allocations, including sibling views. They also
protect gradient casts, literal scaling, negation and layout changes before
chunk reduction, so moving work cannot silently change accumulation precision.
Semantic backing counts exclude external inputs, opaque workspace and allocator
overhead; only complete-task profiling establishes physical admission.

Resolved choices, used semantic versions and copy origins enter executable
identity. Callbacks run during build, including after structural reuse, and are
absent from saved execution. See the
[lifetime reference](../python/api/frontend.md#execution-lifetimes) for selectors
and refusal conditions.

## Gradient destinations and precision

For a fresh parameter gradient, the first contribution creates its buffer.
Later chunks, regions and microbatches update that destination. If an earlier
microbatch already supplied a gradient, even the first chunk accumulates into
it. A fresh contribution can donate storage only when it has no other readers.
Aliased or externally observed contributions remain explicit sums.

Eligible matrix products write directly into the destination. Opaque providers
without a destination overload, unsupported epilogues and layouts that cannot
be mutated densely use a temporary contribution or explicit assembly. This
routing applies to parameter gradients, not intermediate activation gradients.

`fuse_narrow_gradient_accumulation` permits fused arithmetic at narrow gradient
dtypes. When disabled, narrow contributions retain their explicit rounding and
sum association. A contribution rounded before widening stays rounded; a wide
accumulator cannot restore its lost precision. Chunk geometry can still change
floating-point reduction order.

A completed additive subtotal crossing a region edge enters a downstream chunk
loop once. A mutable destination is shared across all iterations. Provenance
through shared linear subgraphs distinguishes these cases; a nonadditive
aggregation ends that provenance and is not thereby certified for chunking.

## Storage and execution

Both graph-pair variants retain their selected full-axis roots. `recompute`
regenerates local intermediates in backward; chunk size does not change its
retained cut. A policy can restrict a nonterminal stage's variants, subject to
mandatory endpoint requirements.

A composition allocates full result roots once, or adopts a child's result when
its physical storage exactly matches the boundary. Children run in the parent's
allocation scope, write slices or add gradients, then release local results and
input copies. Whole-region children follow the same graph order and release
rules. Explicit gradient mutations preserve destination storage; other mutations
are refused.

Assembly preserves shapes, strides, offsets and aliases. Expanded views retain
zero strides and each distinct physical element is written once. Independent
gradient outputs keep independent storage. Host results stay on the host; full
and tail children must agree on locations, and mixed host/device views of one
root are refused.

Full boundaries, saved values and shared gradients are planner objects.
PressureFit chooses their placement and transfers. Internal chunks are neither
microbatches nor separately scheduled or spillable tasks.

## Reuse, profiling and saved execution

Build-local caches reuse axis transforms, split templates and observed storage
facts after lifetime choices, gradient routing and partition assignments have
resolved. Keys include graph/schema semantics, geometry, aliases, roles, axis
rules and accumulation settings. Reuse rebinds current inputs and provenance;
inner policies still resolve for each occurrence. Unsupported or symbolic
representations use ordinary observation. Caches are released on build exit.
Deferred FX wrapper generation avoids regenerating Python source merely to
inspect a split graph; it does not change runtime selection.

Executable identity includes the graph, logical domain, full extent, chunk
sizes, mappings, semantic versions, child storage and kernel identities.
Creator and accumulator children remain distinct. Only invoked full/tail forms
are finalized and saved, recursively for nested regions.

Profile the **complete parent task**. Its time and allocation evidence include
copies, loops, assembly, full roots and opaque provider workspace. Reusing child
code does not permit reusing a different parent's profile. Changed inner sizes
require measuring the resulting parent before planning and admission.

A saved Plan needs neither model nor policy callbacks nor compiler caches.
[`row_chunks` metadata](../python/api/artifacts.md#axis-task-metadata) records the
resolved axis description; `execution_regions` records child bindings and local
releases. These records do not choose residency. Planning remains framework
independent, and the parent is its unit of timing and memory admission.

## Limits

The supported scope is first-order differentiation with explicit additive
gradients and no untracked random or mutable state. Required discrete choices
must match their own forward during backward; fused arithmetic does not promise
bitwise equality across different geometries.

Attention, recurrence, convolution and global routing statistics cannot be
windowed merely because an input has a sequence axis. They need semantics that
certify the proposed split. Generic terminal losses, random-state replay,
wrapped tensor components and arbitrary mutation protocols need additional
support.

Chunking cannot shrink full boundaries, retained activations, weights or private
workspace inside an unsplit operation. Build finalization and profiling run
without a Plan, so placement cannot spill between their internal child calls.
A complete task that does not fit is refused; it is not silently replaced by an
unmeasured implementation.
