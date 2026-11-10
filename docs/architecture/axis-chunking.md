# Axis chunking within a stage

A stage whose operations act independently on logical axis positions can execute a bounded
slice at a time. This limits transient activations and backward recomputation
workspace. Public inputs and outputs retain their complete shapes, so their
storage remains part of ordinary residency planning.
The declared independent domain can be any supported tensor axis; it need not
be token positions. Factored axes align operands with different numbers of
physical elements per logical position.

The builder's `partition` policy defines checkpoint stages. A direct
`ChunkPolicy` selects whole stages for axis chunking. `ChunkRegions` instead
partitions each explicit task into internal execution regions, then applies a
chunk policy to each region. A checkpoint stage can therefore contain both
whole-sequence operations and chunked regions with different chunk sizes.
The [policy reference](../python/api/frontend.md#axis-chunking-policies) describes
these controls. The library does not select boundaries from a model name.

Execution-region chunk loops belong to ShadowSpill's builder and saved
executable. Models express the full tensor computation. Providers implement
operators and may supply versioned rules describing axis independence, local
scratch and gradient reductions; these rules do not implement the loops.
Recipe policies choose region boundaries and chunk sizes. Provider-internal
kernel tiling and an existing model's chunked loss head are separate mechanisms.

## Capture and differentiation

Training capture constructs explicit full-stage forward and backward graphs,
including creator and accumulating gradient forms. Axis propagation transforms
these graphs after their retention cuts and precision conversions are resolved.
It does not differentiate a Python chunk loop or capture a different autograd
program for each slice.

An optional outer `ChunkRegions.lifetimes` policy first schedules approved
readonly operations on the complete task. It resolves declarative rules and
immutable callback selections before gradient destination routing and execution
partitioning. Moving a producer toward its consumer can therefore avoid a
full-axis execution boundary that the original instruction order would create.
The movement pass does not replay work, alter retained values, or change the
parent storage contract. The default `None` leaves the schedule unchanged.

Dispatcher ownership and alias contracts determine read/write dependencies;
fake execution supplies geometry, not alias proof. Unknown opaque effects and
overlapping writes stop movement. Selected barriers stay in place and every
candidate is checked against all their complete live backing sets, including
siblings of partial views. Parameter-gradient conversions are protected before
chunk reductions, including conversions followed by literal scalar scaling,
negation, or gradient layout changes. Moving such a conversion after a chunk
loop could sum contributions in the compute dtype instead of the parameter's
wider gradient dtype. Semantic backing counts exclude external inputs, opaque
workspace and allocator overhead and are not physical admission estimates.

Normalized resolved choices, used readonly-contract versions and movement
decisions enter the composed executable identity. Source occurrence reports are
separate inspection metadata. Cold execution never reloads a policy callback.
Compilation and profiling see the transformed complete task, and ordinary
planning still owns variant selection, spill and physical admission. See the
[lifetime policy reference](../python/api/frontend.md#execution-lifetimes) for
selectors, contracts and current refusals. Explicit `compact=True` selections
prepare internal aliases and consider escaping sibling views jointly. Separate
versioned input-layout contracts prove logical value preservation; readonly
execution alone is insufficient. Layout changes propagate through alias and
fresh operations until a complete dense fresh result has unchanged geometry.
Input/external aliases, written backings and raw storage-sensitive consumers
are protected. A speculative copy group is accepted only with the same outer
storage contract, a reduction at a protected barrier and no increase elsewhere.
Normalized view/copy decisions and used layout versions also enter identity.
Compilation/profiling measures copies as part of the final composition; the
semantic inventory does not substitute for physical admission.

Explicit `replay=True` selections run between movement and compaction. Captured
primal provenance, original dependency versions, effects, output aliases/writes
and gradient-conversion protection constrain each copy. Structural tuple access
accompanies its selected producer unless excluded. Copies are shared within each
consumer region and never inserted into regions containing protected barriers.
Changed alias relationships require input-value semantics even when strides stay
unchanged. Each proposal preserves the task boundary and cannot increase any
protected backing inventory. Normalized decisions and copy origins enter identity;
callbacks run only for the original operations during build. There are no default
replay selections or implicit permissions based on cost. Profiling includes every
extra operation before ordinary planning makes its admission/selection decision.

Execution policies see source module paths resolved for the current checkpoint
occurrence, including operations emitted by its derivative or recomputation.
The [source-provenance contract](graph-pair-construction.md#structural-deduplication)
keeps these selections consistent across structural reuse and disk restore.
Generated operations without source provenance carry no guessed module path.

A chunk dimension may contain factors before and after the logical axis index.
This preserves axis coordinates through reshaping, permutation, flattened batches,
and feature dimensions. Sliced inputs use bounded dense copies with a canonical
offset, preserving permutation order needed by captured views. Full and tail
child graphs are specialized for their actual
geometry; compatible child contracts share compilation.

During a build, identical resolved axis transforms, split templates and semantic
storage contracts share bounded caches. Outer slicing is separate from inner
region composition: full/tail slice graphs can be reused while inner policies
resolve again for the current occurrence. Keys include explicit graph semantics,
dispatcher schemas, tensor geometry, input aliases/backing sizes, roles,
axis/rule contracts and gradient accumulation settings. Nested slice source
metadata and representative-value provenance bind to the current occurrence;
the template preserves its sliced geometry and transformation overrides.

Split templates are reusable only after lifetime processing, gradient routing
and partition assignments have resolved. Fake observations rebind complete
storage aliases to the current inputs, and child graphs receive current source
metadata. Their already observed outputs also supply the parent boundary facts;
the complete graph need not run a second time just to recover output geometry.
Concrete tensors, unsupported representations and symbolic geometry use the
ordinary observation path. These caches are released on normal or exceptional
build exit. Partition, lifetime and chunk-policy callbacks still run for every
occurrence. Saved-state cuts, executable identities and physical admission remain
unchanged; the complete composed tasks are still compiled and profiled normally.
Split graphs use PyTorch's deferred FX wrapper generation: copying or inspecting
their structure need not regenerate Python code. Calling the graph or reading
its `.code` generates the current wrapper, including after graph edits. This
changes construction cost, not task compilation or runtime kernel selection.

Built-in rules accept supported elementwise operations, feature reductions,
views and matrix products. A shared parameter gradient is a sum of chunk VJPs.
Sliced outputs concatenate into the full boundary geometry. Unknown operations,
nonadditive chunk reductions, unsupported effects and ambiguous aliases fail
capture. A caller-supplied operator rule certifies semantics; validating its
shapes cannot prove that an opaque implementation is independent along the selected axis.

Child computations retain the captured tensor dtypes and precision policy.
Changing matrix shapes and reduction order can change floating-point rounding.
A narrow contribution rounded before conversion to a wider accumulator retains
that rounding; the accumulator's dtype does not increase the precision of the
contribution-producing operation. Numerical qualification must cover the chosen
chunk size and precision combination.

For a fresh shared backward gradient, the first chunk produces its buffer and
subsequent chunks receive that buffer as a mutated input. An existing gradient
input, including one carried from an earlier microbatch, is updated from the
first chunk. Eligible matrix-gradient products add directly into that storage;
the builder's `fuse_narrow_gradient_accumulation` setting also applies here.
An opaque provider gradient or an unsupported product epilogue can still require
a temporary contribution and an in-place addition inside the compiled child.
Broadcast gradient layouts that cannot be mutated as dense tensors use explicit
assembly instead of an accumulating child.

Before execution partitioning, eligible parameter-gradient sums are routed to
one destination. A first contribution may donate its storage when it is fresh
and has no other readers; an existing gradient input can receive multiple
contributions directly. Later execution regions consume that same destination.
This transformation follows gradient assembly and stops at contribution
producers. It does not turn intermediate activation adjoints into parameter
destinations. Narrow sum trees keep their association when narrow fusion is
disabled. Aliased or externally observed contributions remain explicit sums.

An axis-independent forward output is assembled once. Its upstream cotangent
enters only the first child backward; later children receive zeros for that
port. This adds its contribution once when it meets a chunk-summed parameter
gradient. In contrast, a summed forward statistic receives the same upstream
cotangent in every child. Both remain linear VJP contributions.

Opaque scratch can instead be declared chunk local. Its dimensions may scale
with the axis extent without claiming original axis order. It cannot escape in
a boundary or retained cut, and ordinary operators cannot mix it with ordered
axis positions. An explicit provider rule must restore the axis contract.
This permits
internal dispatch permutations while refusing invalid concatenation of local
indices. A provider may expose additive sufficient statistics; a nonlinear
aggregate remains a separate full-domain computation.

Differentiable stage outputs that alias inputs or each other receive independent
roots. This keeps their cotangents inside the explicit backward graph rather
than an autograd wrapper. Downstream boundary geometry is observed again before
capture. Fresh output views retain the offsets required by their consumers.

## Retention and storage

Checkpoint boundaries and execution boundaries have different lifetimes.
Only checkpoint inputs and the graph pair's saved values cross the forward to
backward interval. Internal execution boundaries are local to one task call;
the composition drops their references after their last consumer. Changing
execution partitioning leaves the outer graph pair's retention cut unchanged.
An internal cut can still increase transient workspace by materializing an
intermediate. It is not a promise that a complete task fits its memory budget.

Each stage occurrence retains one graph-pair choice. A `save` variant stores its
full set of saved roots across the full axis; chunking does not make those roots
smaller. A `recompute` variant retains the chosen full-stage cut and rebuilds
chunk-local intermediates during its explicit backward. Chunk size does not
change that cut. A policy may restrict a selected nonterminal stage to either
variant, while mandatory endpoint requirements still apply.

The composition allocates full result roots once, adopting complete first-child
results when their exact physical storage matches the public contract. Each child invocation runs in
the parent's allocation scope, writes its axis slice or adds its shared gradient
contribution, then releases child results and input copies before the next
invocation. Explicit additive gradient mutations preserve their declared prior
storage. Other mutations are refused.

Host output locations reported by child lowering remain host locations in the
parent's assembly and allocation contract. Full and tail children must agree on
those locations; a mixed host/device view of one output root is refused.

Full boundaries, saved values and shared gradients are ordinary planner objects.
PressureFit chooses their residency and transfers. Internal chunks are not
scheduler tasks, microbatches, independently spillable views, or separate
optimizer updates. No planner or runtime special case identifies an attention
architecture.

An execution composition invokes its children in graph order. Whole-region
children and chunk-loop children share the parent's allocation scope. Before an
edge is consumed, its geometry, offsets and aliases match the captured contract.
Compiler-owned outputs can be reused only when their physical allocation matches
that contract; otherwise the composition materializes the declared storage.
Expanded boundary views retain their zero strides. Assembly writes each distinct
physical element once, preserving the captured broadcast aliases.
Explicit gradient materialization changes storage ownership while preserving
the value's sliced, shared, or additive-reduction layout. Independent gradient
outputs retain independent storage after chunk assembly.

An axis composition can contain an execution composition inside each slice.
Each inner region selects its own chunk bound. Its intermediate boundaries
span only the enclosing slice, and their last uses finish before the next
outer slice starts. The complete group passes the outer axis contract before
its children are formed. Whole-axis operations remain outside that group.
This nesting changes execution lifetimes without introducing checkpoint saves
or extra copies of expensive computation in the explicit graph.

Gradient ownership is established before inner execution partitioning. The
first contribution creates a destination; subsequent inner and outer chunks,
and later microbatches, use that destination. A completed additive subtotal
crossing an execution cut enters a downstream chunk loop once. A mutable
gradient destination instead remains shared across every iteration. Shared
linear subgraphs preserve subtotal provenance across their own cuts. An
unchunked aggregation runs once on completed values and ends that provenance
when its operation has no additive contract. This does not certify the
aggregation for chunking; an enclosing axis composition still has to pass its
complete separability checks.

## Compilation, measurement and execution

The executable archive contains a versioned composition and finalized child
payloads. An unsupported composition version requires rebuilding the Program.
Identity includes the original explicit graph, full axis extent, logical domain
name, chunk size, input/output axis mappings, operator-rule versions, child contracts and
physical output storage. Kernel source identities distinguish identically named
kernels from different child specializations.
Creator and accumulating children have distinct contracts. Only specializations
invoked by the complete composition are packaged and finalized.
Nested payloads restore recursively, including full and tail execution
compositions. Their resolved child contracts participate in parent identity;
changing any inner size requires profiling the resulting complete parent.

Measurement restores and runs the complete composition under the ordinary
allocator. Its timing and allocation evidence include the loop, input copies,
assembly, full roots, providers and child workspace. Child measurements alone
cannot certify a parent. Allocation evidence passes through the existing
profile binding, physical admission and planning pipeline. Planning remains
framework independent and does not import the frontend.

A saved Plan executes without the source model, policy callback, capture or
compiler caches. Runtime admission and pool bounds apply to the complete task.
`row_chunks` metadata in the [artifact reference](../python/api/artifacts.md#axis-task-metadata)
records the resolved contract; it does not prescribe residency.
`execution_regions` records nested child bindings, local releases and each
selected child's own axis contract. The complete parent remains the unit of
profiling and planning; child code reuse does not imply profile reuse across
different parent compositions.

## Boundaries and limitations

The supported scope is first-order chunked regions with explicit additive
gradients and no untracked random or mutable state. This is a semantic
requirement, not a bitwise reproducibility guarantee: normal fused execution
can change rounding and near-tied routing or sparse selections across different
chunk geometries. Discrete choices needed by backward must still agree with
their own forward execution through the provider's retention contract.
Generic terminal training losses, random state replay,
wrapped tensor components and arbitrary mutation protocols require additional
contracts. Sequence attention, recurrence, convolution and global routing
statistics must remain outside a selected chunked region unless a provider
supplies a contract that preserves their complete semantics. Merely dividing a sequence
into windows does not satisfy this condition.

Chunking reduces transient workspace. It does not reduce full boundary
storage, aggregate saved activations, resident parameters, or workspace inside
an unsplit sequence operation. Build finalization and profiling run without a
Plan, so residency planning cannot spill between their internal child calls.
If an allowed variant cannot fit as a complete task, build or measurement fails;
no unmeasured implementation or silent unchunked replacement is admitted.
