# PyTorch lowering contract

ShadowSpill lowers a compiled PyTorch task in two deliberately separate steps:

```text
Export/AOT FX semantics
    -> TaskStorageContract
    -> compiled physical-layout profile
    -> canonical Program objects
    -> PressureFit / simulator / runtime
```

Graph semantics are the sole authority for object identity, aliases, views,
and mutations. Compilation and profiling describe how those objects are
physically realized and how much time and memory the executable consumes. A
physical observation may reject an inconsistent contract, but it must never
create, merge, or split semantic objects.

## Component contracts

### PyTorch capture

Strict Export and AOTAutograd provide normalized functional FX graphs with
explicit task inputs and outputs. The capture boundary is responsible for:

- tensor and static argument structure;
- operator and output-node provenance;
- output pytrees;
- top-level state and user-input mutations;
- forward/backward graph pairs and legal recomputation alternatives.

Every opaque operation must provide the fake/meta, shape, alias, mutation, and
autograd contracts required by PyTorch compilation. An operation that hides an
alias or mutation from PyTorch cannot be repaired safely by ShadowSpill.

### Semantic lowering

Semantic lowering consumes only the functional graph, representative guarded
geometry, and PyTorch operator contracts. It emits an immutable
`TaskStorageContract` containing:

- dense storage roots that originate at an input or a fresh producer result;
- output views with root, shape, stride, dtype, offset, and minimum span;
- input and state mutations;
- deterministic provenance and a compatibility digest.

It performs no CUDA execution, allocator tracing, or performance profiling and
does not import private Inductor implementation modules. Fresh FakeTensor/meta
evaluation may be used to calculate geometry, but FakeTensor storage identity
is never semantic evidence.

If an alias cannot be proven, lowering fails before model mutation. It does not
fall back to allocator telemetry or coincidental FakeTensor storage.

Export functionalizes state updates as fresh replacement outputs. ShadowSpill
keeps that functional result intact: the semantic contract records both its
fresh root and the signature-declared input object that it replaces. No
`aten.copy_` is inserted. At `after_task`, the runtime atomically installs the
fresh allocation as the canonical object's next execution lease, retires the
old lease behind the task-completion fence, and the frontend rebinds every
registered view of that alias bundle. This preserves registered Tensor
identity while avoiding an extra device-to-device copy and its bandwidth cost.

### Compiled physical layout and profiling

The isolated compiled-task profile reconciles the already-established semantic
roots with the executable. It records:

- output allocation ordinals and actual output offsets;
- requested and allocator-charged physical extents;
- allocation lifetime and reuse information;
- anonymous and opaque workspace high-water;
- task duration and provider growth.

This layer validates that the compiler preserved the graph contract and
provides the physical quantities required by PressureFit and slab admission.
It cannot change root identity or alias relationships.

### Program construction and runtime

Program lowering resolves task-local roots against canonical cross-task
objects, saved values, gradients, parameters, buffers, and optimizer state.
PressureFit and the simulator operate only on that canonical Program. The
runtime executes the selected actions and verifies the admitted physical
layout; it does not rediscover graph semantics.

## Existing approach versus the new contract

The old implementation conflates semantic tensor identity with observed
physical allocation behavior. The replacement separates those concerns.

| Question | Existing approach | New approach |
|---|---|---|
| Is an output newly created or an alias? | Infer from FakeTensor storage identity and allocator traces | Derive from FX provenance and operator alias schemas |
| Does an output alias a task input? | Guess from matching captured tensor identity | Record an input root directly from the graph |
| Do two output leaves share storage? | Match an observed allocation ordinal or FakeTensor storage | Give both leaves the same semantic root |
| What is the semantic view offset? | Reuse incidental FakeTensor metadata | Evaluate fresh symbolic geometry relative to the proven root |
| What is the compiled physical extent? | Treat it as semantic identity evidence | Record it only in the compiled physical layout |
| How large is opaque workspace? | Allocator telemetry | Allocator telemetry, unchanged |
| Who determines Program object identity? | A mixture of capture and profiling | Offline graph semantics only |
| Who validates physical feasibility? | Profiling and admission | Profiling and admission, unchanged |

## Why the existing approach fails

The existing lowering has two competing sources of truth:

1. FakeTensor storage identity such as `tensor.untyped_storage()._cdata`.
2. Allocator telemetry such as allocation ordinal, charged bytes, and returned
   output leaves.

That mixture has produced four concrete failures.

### One value split into two Program objects

In the first mlops Qwen forward stage, flattened output leaves 1 and 11 are the
same FX node, while leaves 2 and 12 are another repeated FX node. The first
pair occupies one 32-byte allocation.

One occurrence was classified as a canonical stage output and the other as a
saved residual. The former received identity from stage-boundary capture while
the latter received identity from compiled allocation telemetry. They could
therefore become separate alias bundles even though the graph returned the
same value twice. The planner charged two objects while the executable created
one allocation.

Under the new contract both leaves reference one fresh storage root. Their
frontend roles do not affect identity.

### A returned input has no allocation event

One Qwen backward task returns an existing task input as output leaf 16. This
is a normal passthrough result, so no `malloc()` occurs. The old fallback
consulted FakeTensor storage produced by a separately captured graph and could
invent a new output object. Spatial admission then waited for an allocation
that correctly never occurred.

The new contract declares the output view against its input root. No output
allocation is required or expected.

### FakeTensor sharing does not predict compiled sharing

Pure Qwen exposed a 104,448-byte synthetic FakeTensor storage containing
several 26,112-byte outputs. Inductor compiled those simultaneously-live
outputs as separate allocations. FakeTensor described functional example
geometry; it did not promise a compiled physical allocation group.

The new contract assigns distinct non-aliasing producers distinct roots even
when their synthetic examples happen to share a storage. The compiled layout
then attaches each root to its actual 26,112-byte allocation.

### Inductor may compact a returned view

Another Qwen value appeared in capture as a 39,936-byte view at byte offset
39,936 inside a 79,872-byte synthetic storage. The unreturned half was not
observable, and Inductor emitted one compact 39,936-byte output allocation at
offset zero.

The semantic contract retains the returned view relationship and minimum
span. The physical-layout profile records the compact compiled offset and
extent. Neither representation is mistaken for the other.

### Fresh-symbolic geometry incident

The first draft of the replacement trusted `node.meta["val"]` for view
geometry. A basic `value[2:10]` graph then reported byte offset zero instead of
eight: the `make_fx` node metadata contained a canonicalized FakeTensor even
though evaluating the graph produced the correct offset-two view. This is why
the final extractor evaluates every contract with fresh FakeTensor/meta inputs
and uses node metadata only as descriptive provenance.

## Why the replacement is generic

The lowering rules depend on graph structure and PyTorch contracts, not model
or operation-provider names:

- exact node identity handles duplicate outputs;
- placeholders handle input passthroughs;
- dispatcher schemas handle aliases, views, and mutations;
- producer/result identity distinguishes fresh values;
- fresh symbolic execution supplies guarded geometry;
- compiled profiling supplies provider-specific physical layout and workspace.

This applies equally to native ATen operations and registered custom
operations. A decomposed custom operation exposes its internal graph. An
opaque custom operation exposes declared inputs, outputs, aliases, mutations,
and fake geometry while its private workspace remains a measured physical
quantity.

The supported claim is intentionally precise:

> Any fixed-shape graph accepted by ShadowSpill's strict
> Export/AOTAutograd/Inductor capture and whose operations provide correct
> fake/meta, alias, mutation, and autograd contracts is lowered without model-
> or operator-specific policy.

Graph breaks, unbounded data-dependent output geometry, or incorrect hidden
custom-operation semantics fail explicitly rather than activating a heuristic
fallback.

## PyTorch API boundary

PyTorch publicly documents a custom compiler backend that receives an FX
`GraphModule` and representative inputs. `ExportedProgram.graph_signature`
also publicly models lifted state and mutations. PyTorch does not expose a
supported post-Inductor allocation manifest containing
`GraphLowering.graph_outputs`, scheduler lifetimes, and generated
allocation/free statements.

ShadowSpill therefore does not introduce a version-pinned Inductor adapter for
correctness. A future optional compiler-layout extractor may enrich physical
profiling, but its absence cannot alter semantic lowering.

References:

- [PyTorch custom backends](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_custom_backends.html)
- [PyTorch Export graph signatures](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)

## Invariants

The implementation and qualification suites enforce these invariants:

1. Semantic capture performs no CUDA allocation, task profiling, or allocator
   trace inspection.
2. Physical profiling cannot change a semantic root or mutation.
3. Distinct fresh roots remain distinct unless graph semantics prove aliasing.
4. Repeated leaves and passthrough values never require duplicate allocations.
5. Every physical output is reconciled with exactly one fresh root, except
   explicit zero-byte and input-root outputs.
6. All physical views fit the admitted allocation and all workspace is charged.
7. Unsupported contracts fail before model mutation.
8. Production lowering contains no Llama, Qwen, OLMoE, or mlops-specific
   behavior.
9. Every Export state replacement remains a fresh compiled result and is
   installed as a new object generation without an inserted numerical copy.
10. The old and replacement generations overlap until the task fence; that
    exact compiled extent is charged as transition workspace in simulation and
    spatial admission.
