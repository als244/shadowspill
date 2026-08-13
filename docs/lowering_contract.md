# PyTorch lowering contract

ShadowSpill separates logical PyTorch semantics, the optimized executable ABI,
and measured physical costs. None of these layers is allowed to impersonate
another:

```text
Export / AOTAutograd FX
    │ logical values, public outputs, saved values, mutations
    ▼
AOT semantic TaskStorageContract
    │ the contract presented to Inductor
    ▼
Inductor optimized FX (captured before code generation)
    │ value rewrites and candidate output provenance
    ▼
Inductor GraphLowering output manifest
    │ final buffers, aliases, views, mutations, wrapper allocation extents
    ▼
ExecutableTaskManifest
    │ validated by isolated execution; never inferred from it
    ▼
CompiledTaskLayout
    │ validated leases, charged geometry, workspace, provider growth, timing
    ▼
canonical Program → PressureFit → simulator → runtime
```

The central rule is that allocator telemetry measures physical realization; it
never discovers object identity. Conversely, an AOT graph describes logical
values but cannot be assumed to describe the output-storage ABI of the graph
after Inductor optimization.

## Component contracts

### 1. Export and AOTAutograd: logical program authority

Strict Export and AOTAutograd provide normalized functional FX graphs. This
layer owns:

- tensor and guarded static argument structure;
- public output and saved-value positions;
- forward/backward graph pairs and legal recomputation variants;
- top-level state and user-input mutations from `ExportGraphSignature`;
- logical dependencies between stages.

Every opaque operation must provide correct fake/meta, alias, mutation, and
autograd contracts to PyTorch. ShadowSpill fails closed when one of those
contracts is absent or ambiguous.

Each gradient-accumulation position retains its own objective-output schema.
Tensor metric structure may be structurally identical across positions, but
static metric leaves are values produced for that guarded microbatch and are
never borrowed from another position.

The AOT graph is not the final executable storage ABI. Inductor is allowed to
rewrite it while preserving values. For example, post-gradient optimization
turns `x * ones_like(x)` into `x`; a logically fresh AOT output is therefore an
input alias in the compiled callable.

### 2. AOT semantic contract: compiler-input diagnostics

`GraphArtifact.storage_contract` records the storage semantics presented to
Inductor. A `TaskStorageContract` contains:

- dense roots originating at a task input or fresh producer/result;
- output views with root, shape, stride, dtype, offset, and minimum span;
- functional and dispatcher-schema mutations;
- deterministic provenance and a compatibility digest.

It is extracted offline from FX provenance and operator schemas. A fresh
FakeTensor/meta evaluation supplies geometry only. FakeTensor storage identity,
CUDA execution, allocator tracing, and performance profiling are prohibited.

This contract remains useful for proving what Inductor changed and for
diagnostics. It is not used blindly for runtime bindings.

### 3. Inductor compiler contract: task-boundary storage authority

ShadowSpill's narrow compiler adapter invokes the existing `compile_fx`
boundary and captures two compiler-authored representations. The optimized FX
graph records value-level rewrites such as input passthrough. Inductor's final
`GraphLowering.graph_outputs` records the concrete buffers and layouts that its
generated wrapper returns. ShadowSpill normalizes both offline and stores them
in an immutable `ExecutableTaskManifest` beside the callable.

This layer owns:

- whether a compiled output is fresh or aliases an input;
- whether disjoint pieces of one lazy FX value are materialized as independent
  executable buffers;
- which returned views share one final executable buffer;
- executable output offsets, strides, and extents before physical execution;
- exact requested allocation length for every fresh returned root, using the
  same `GraphLowering.get_allocation_storage_size()` calculation as Inductor's
  generated wrapper;
- the mutation ABI retained by the optimized graph.

The optimized FX contract is diagnostic provenance, not the final allocation
authority. GraphLowering may independently realize disjoint views of one lazy
FX root while scheduling or fusing kernels. Consequently, one optimized-FX
root need not equal one wrapper-visible allocation. The GraphLowering contract
is authoritative for final executable roots and physical-layout
reconciliation.

The adapter executes no CUDA work and consults no allocator telemetry. It is
the only module that imports PyTorch's private Inductor compilation entrypoint
and compatibility helper. Those dependencies are explicitly version-checked
and confined to the PyTorch frontend; the IR, simulator, planner, and runtime
do not depend on Inductor.

PyTorch currently has no public API that returns the final GraphLowering
storage manifest beside a compiled callable. A public custom backend sees the
pre-Inductor FX graph, while the ordinary compiled callable discards the
GraphLowering object after constructing `CompiledFxGraph`. The version-pinned
adapter captures that already-existing object during compilation rather than
copying Inductor's pipeline or reconstructing its result from pointers.

The adapter validates that optimization preserves tensor output leaves,
geometry, and explicit mutation targets. Alias changes are legal and recorded.
An unsupported ABI rewrite fails during compilation with both contracts in the
diagnostic.

Inductor's content-addressed FX cache can return the compiled callable without
reconstructing `GraphLowering`; therefore a cache hit cannot reproduce this
manifest by calling the compiler hook again. ShadowSpill stores a deterministic
manifest sidecar keyed by the exact Inductor FX-cache key and AOT semantic
contract digest. The sidecar records both optimized-FX and executable storage
contracts, the PyTorch and accelerator-runtime versions, and its own validated
compatibility digest. A cache hit is accepted only when every identity agrees.
If an external compiler cache entry has no sidecar, ShadowSpill recompiles that
one task once with compiler caches disabled, captures the authoritative
`GraphLowering` result, and seeds the sidecar; it never guesses the missing
contract from pointers.

Direct `compile_fx` normally creates an internal fixed-shape FakeTensor mode
without a `ShapeEnv`, causing its inner FX cache to report `No shape env` and
bypass caching. The adapter attaches an empty `ShapeEnv` to that exact existing
compiler context. It does not replace task arguments. This distinction is part
of the contract: an earlier experiment converted real task arguments to new
FakeTensors and changed Inductor's physical allocation for two sliced Qwen
matrix-multiply outputs from 156,672 to 221,184 bytes. Preserving the original
exact-stride arguments restores byte-identical executable contracts, compiled
layouts, workspace, and transition accounting while still enabling cache
reuse.

### 4. Compiled physical layout: compiler authority plus measurement

`ExecutableTaskManifest` owns each fresh output root's requested allocation
extent. This value comes directly from Inductor's final buffer layout and is
available offline. Isolated structural profiling then produces
`CompiledTaskLayout` by validating allocator and provider observations against
that immutable ABI:

- output allocation ordinals and actual view offsets;
- exact agreement between the observed allocation request and the compiler
  request, plus the allocator-charged extent;
- allocation/free lifetime and reuse geometry;
- anonymous or opaque workspace high-water;
- bounded persistent provider growth;
- CUDA-event task duration.

Reconciliation uses the GraphLowering executable manifest, not either earlier
FX contract.
An input-root output must be observed at that input; a fresh root must have one
compatible output allocation; distinct simultaneously-live roots cannot be
merged. Any disagreement fails closed.

This distinction prevents a cached telemetry trace from redefining a Program
object. A concrete Qwen regression attributed a 156,672-byte returned buffer
to a later reused 221,184-byte allocation. Its semantic and executable roots
were unchanged, but the stale physical observation made the Program declare
the larger object and fail when the real callable returned the correct smaller
lease. ShadowSpill now rejects and remeasures such a cache entry before Program
construction. Telemetry still owns charged slab geometry because a generic
pool may round a compiler request, but it cannot alter the requested extent.

Opaque custom-operation workspace is necessarily measured unless the provider
declares it. Inductor can expose wrapper-visible temporary buffers, but it
cannot see private allocations inside an opaque CUDA/Triton/library call.
Telemetry therefore remains a physical validation and admission mechanism,
not a semantic-lowering mechanism.

### 5. Program lowering and runtime

`ObjectCatalog` owns cross-task objects and alias bundles.
`TaskBindingResolver` combines one executable storage contract, its compiled
layout, and predecoded task inputs. Forward, backward, recomputation,
inference, and optimizer tasks use this same path.

Program lowering is an offline translation. It never invokes a captured
forward or backward graph to rediscover outputs, residuals, or gradients.
Instead, the resolver binds every task leaf directly from its immutable
storage contract and reconciled compiled layout. Occurrence-specific live
inputs may be rebound to a cached structural graph pair only after their full
geometry and input-alias pattern match; the cached backward contract is shared
directly because its residual and tangent slots are canonical Program objects,
not occurrence-specific Python tensors.

This distinction is both semantic and operational. Executing graph pairs while
constructing the Program made lowering depend on CUDA execution and retained
large temporary result trees long enough to distort planning latency. It also
repeated identical work for the initial and recurrent optimizer phases. The
current path memoizes compiled-layout reconciliation by structural ABI and
constructs both phases from contracts alone. Profiling remains responsible for
timing, charged extents, workspace, and provider growth; none of those
measurements is needed to decide which logical object a task leaf denotes.

Program lowering is responsible for saved values, stage boundaries, gradient
contributions, parameters, buffers, optimizer state, and public outputs.
PressureFit and the simulator consume only the resulting canonical `Program`.
The runtime enforces that plan; it never rediscovers graph semantics.

Residency is always alias-bundle based. Initial residency is also classified
by alias bundle, not individual view object: if any view of an alias is
produced by a task, another view cannot cause that same alias to be treated as
an external input.

An executable input-root output is a zero-copy lease handoff when the
cross-task source and destination are intentionally distinct canonical
objects. The source remains valid through the task-completion fence; the
destination becomes the current owner immediately for subsequent same-stream
dispatch. Several consecutive tasks may therefore form an ordered handoff
chain before the worker observes the first completion. The runtime represents
that chain as an intrusive FIFO on the one memory lease. Completion retires
sources in FIFO order and never copies, synchronizes the host dispatcher, or
changes the PressureFit schedule.

## Mutation semantics without copies

Export normally represents state mutation as a fresh returned replacement.
ShadowSpill does not insert `aten.copy_`. The executable contract associates
that leaf with the canonical state object. At `after_task`, the runtime swaps
the new memory lease into the object, retires the old generation behind the
task fence, and the frontend rebinds every registered view. The simultaneous
old/new extent is charged explicitly as transition workspace.

Inductor may prove a mutation is a no-op and return the target input itself.
That is legal only when the replacement aliases the same canonical input. It
requires no generation swap or transition extent. A replacement that aliases
a different input fails closed because silently merging distinct user objects
would be incorrect.

## Existing approach versus this contract

| Question | Fragile mixed-authority path | Current contract |
|---|---|---|
| What value or state is represented? | AOT plus incidental tensor identity | Export/AOT semantic contract |
| Is the compiled output fresh or an input alias? | Infer from allocator pointers | Optimized FX provenance plus final GraphLowering buffer |
| Do returned leaves share storage? | FakeTensor `_cdata` or allocation ordinal | Final GraphLowering root |
| What is the executable view offset? | Mix captured and observed offsets | GraphLowering layout, physically validated |
| What is the requested output extent? | Timing-sensitive telemetry attribution | Inductor wrapper allocation manifest |
| What is the allocator-charged extent? | Telemetry also influenced identity | Telemetry validated after identity and request are fixed |
| What is opaque workspace? | Allocator telemetry | Allocator telemetry or provider declaration |
| Who creates Program objects? | Capture/profiling mixture | `ObjectCatalog` + `TaskBindingResolver` |
| Can profiling merge roots? | Previously yes in edge cases | Never |

The new path works where the old path fails because it observes both the
compiler's value rewrites and its final wrapper-visible storage ABI before
execution. Pointer equality is now evidence that the physical run honored an
already-known alias or allocation, rather than a heuristic that retroactively
changes graph meaning.

## Root-caused failures

### Repeated Qwen outputs split into multiple objects

An mlops Qwen stage returned the same FX nodes at leaves 1/11 and 2/12. Role-
specific binding classified one occurrence as a stage output and another as a
saved residual, producing separate objects for one value. One semantic root
now binds both leaves regardless of frontend role.

### Backward input passthrough invented an allocation

A Qwen backward task returned task input 16. No allocator callback correctly
occurred, but the prior fallback invented a fresh output from separately
captured FakeTensor storage. The executable contract now records an input root
and physical reconciliation requires the observed input pointer.

### Consecutive cotangent passthroughs exceeded one runtime handoff slot

The formal mlops OLMoE run exposed a runtime contract omission after lowering
correctly identified an input alias. In
`microbatch_0000.stage_0013.backward.recompute`, output leaf 15 is a four-byte
float32 cotangent whose GraphLowering root is exactly backward input 38. It
therefore requires no allocation or copy. The task hands the existing lease
from canonical alias 957 to alias 953.

The preceding backward task had already handed that same lease from an earlier
cotangent to alias 957. PyTorch correctly ran ahead on its ordered compute
stream before the worker observed the preceding task fence. The old runtime
stored only one pending `source -> destination` transition on a lease, so the
second bind failed even though both transitions were causally ordered and the
selected schedule was valid.

The corrected lease stores a FIFO of source objects. Each source records its
destination, trigger task, and next source; the lease stores only the FIFO
head and tail. The worker retires a source only after its own task fence and
same-task release action complete. Thus `A -> B -> C` may be submitted without
waiting for `A` to retire, while physical ownership cannot be released out of
order. An admitted-runtime regression holds the first mock completion for 50
ms, submits the second handoff, and proves that `A` and `B` retire while `C`
retains the original address. The pressured OLMoE schedule then completed five
steps and bitwise checkpoint replay without adding a copy.

### Synthetic sharing differed from compiled sharing

Pure Qwen showed one 104,448-byte FakeTensor storage containing several
26,112-byte values. Inductor allocated the simultaneously-live results
separately. Producer/result provenance defines distinct roots; profiling then
attaches one physical extent to each.

### Inductor compacted a returned view

A captured 39,936-byte view began at byte 39,936 in a 79,872-byte synthetic
storage. Inductor omitted the unreachable half and returned a compact
39,936-byte allocation at offset zero. The executable contract/layout records
the compact boundary while the AOT contract remains diagnostic evidence of the
pre-optimization value.

### Stale metadata reported the wrong slice offset

The first extractor read `node.meta["val"]`; `value[2:10]` was reported at
offset zero instead of eight bytes. Re-executing the FX graph with fresh
symbolic inputs produced the correct geometry. Node metadata is now provenance
only.

### Post-gradient simplification changed a fresh root into an input root

The grouped mlops-Llama backward graph contained a 525,336,576-byte
`getitem_11 * ones_like(loss)` output. AOT correctly classified it as fresh,
but Inductor simplified the expression and its generated wrapper returned task
input 35 without allocation. Treating this as runtime buffer donation caused
mixed initial/produced residency and incorrect cotangent outputs.

A minimal control established the exact boundary: direct Inductor and ordinary
`torch.compile(..., backend="inductor")` returned the input storage, while
`aot_eager` returned fresh storage. The `inner_compile` graph already contained
the input passthrough. The executable manifest now captures that fact offline.

### View-level initial residency violated alias ownership

After introducing executable input roots, one produced stage-boundary alias
also contained later view objects used as task inputs. Initial-residency logic
checked produced object IDs rather than produced alias IDs and incorrectly
declared the bundle external. Classification is now entirely alias-bundle
based.

### One optimized FX root became three executable allocations

Pure Qwen produced a 2,048-byte optimized-FX root whose three returned views
covered disjoint byte ranges `[0,512)`, `[512,1024)`, and `[1024,2048)`.
Inductor's final wrapper returned three independently materialized buffers of
512, 512, and 1,024 bytes. Treating the earlier FX root as one executable
allocation made physical reconciliation report one semantic root across
allocation ordinals 8, 10, and 17.

This is legal compiler behavior: a lazy FX value is not an allocation
manifest. `GraphLowering.graph_outputs` identifies the three final buffers, so
the executable contract now preserves them as three roots. Allocator telemetry
only confirms their extents and lifetimes.

### Zero-length outputs were assigned a fake physical lifecycle

An mlops Qwen stage returned two zero-length CUDA tensors. They have semantic
identity and downstream dependency positions, but no allocation callback and
a null `data_ptr()`. Slab admission first demanded nonexistent output extents;
after that was corrected, runtime adoption rejected their null addresses.

Zero-byte alias groups are now explicitly nonphysical throughout planning,
simulation, spatial replay, and runtime bridging. They remain tensor values in
the frontend, are always ready for dependency purposes, and never receive an
initial residency, memory action, lease, transfer, rebind, or retirement.

### Program lowering re-executed structural graph pairs

The first robust-contract implementation still called every captured
forward/backward pair while building the canonical Program. Those calls did
not determine the final executable ABI—the semantic and GraphLowering
contracts already did—but they manufactured Python tensors used by the old
binding path. Repeated blocks therefore paid repeated CUDA execution, output
flattening, and tensor-lifetime costs during what should have been an offline
translation. Pure Qwen spent 7.414 seconds in Program lowering in a controlled
pre-change run.

`TaskBindingResolver.bind_contract()` now consumes `OutputView` records
directly, and `ObjectCatalog` derives logical extents from view geometry.
Compiled-layout reconciliation is cached once per structural ABI. With the
same profile cache and identical Program digest, pure Qwen Program lowering
fell to 0.373 seconds. Five-step pressured execution and checkpoint replay
then passed, proving that the removed graph executions supplied no required
runtime semantics.

## Genericity and limits

The implementation contains no Llama, Qwen, OLMoE, or mlops-specific policy.
The supported claim is:

> Any fixed-shape graph accepted by ShadowSpill's strict
> Export/AOTAutograd/Inductor capture and whose operations provide correct
> fake/meta, alias, mutation, and autograd contracts is lowered without model-
> or operator-specific policy.

A decomposed custom operation exposes its internal graph. An opaque custom
operation exposes its declared output/alias/mutation ABI and fake geometry;
its private workspace is measured. Graph breaks, unbounded data-dependent
geometry, incorrect custom schemas, or unsupported compiler ABI rewrites are
explicit capture errors rather than heuristic fallbacks.

## Invariants

1. AOT and executable contract extraction performs no CUDA execution or
   allocator tracing.
2. Only the narrow PyTorch compiler adapter imports private Inductor modules.
3. Profiling cannot create, merge, split, or rename executable roots.
4. Repeated leaves and input passthroughs never require duplicate allocations.
5. Every nonempty GraphLowering root reconciles with exactly one physical
   output allocation.
6. Every input-root output physically aliases its declared input and fits its
   storage.
7. Functional mutation replacement is either fresh, or a proven no-op alias of
   the same target input.
8. Alias-bundle identity, residency, and transfer remain consistent across all
   views.
9. Unsupported contracts fail before the user model is mutated.
10. Plan diagnostics retain semantic, optimized-FX, and GraphLowering contract
    digests, roots, views, mutations, physical layout, and task/ABI mappings.

## PyTorch boundary references

- [PyTorch custom backends](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_custom_backends.html)
- [PyTorch Export graph signatures](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)

PyTorch does not currently document a public post-Inductor output-storage
manifest. ShadowSpill's adapter is consequently a small, isolated compatibility
boundary rather than a dependency spread through lowering or runtime code.
