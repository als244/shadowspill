# Graph-pair construction

Graph-pair construction turns one differentiable, partitioned PyTorch stage
into every executable forward/backward alternative that ShadowSpill is willing
to expose to planning. It is a PyTorch frontend operation, and it is neither
[graph-pair selection](graph-pair-selection.md) nor part of the
[search](search.md).

The three layers have deliberately different outputs:

| Layer | Unit of work | Output |
|---|---|---|
| Graph-pair construction | One structural stage contract | A `TaskGraphPairs` containing named forward/backward variants |
| [Graph-pair selection](graph-pair-selection.md) | All occurrence-level `TaskAlternativeGroup` values in one program | A bounded tuple of complete `TaskAlternativeChoice` assignments |
| [Plan search](search.md) | One complete assignment plus the program and machine model | A residency/action schedule with simulated cost |

This separation lets the frontend add another legal graph-pair variant without
changing the framework-neutral selection, search, simulator, or runtime
contracts.

## Vocabulary

| Term | Meaning |
|---|---|
| Stage occurrence | One ordered partition in one captured input geometry or accumulation round. |
| Structural contract | The deterministic graph/input/storage identity shared by equivalent stage occurrences. |
| Graph pair | One mutually compatible AOTAutograd forward graph and backward graph. |
| Variant | A named graph pair produced under one partition budget, such as `save` or `recompute`. |
| Task graph pairs | Every configured legal variant for one structural contract. |
| Graph-pair group | The occurrence-level program choice whose options activate one variant's forward and backward tasks. |

The rest of the family — choice, selection, and problem — is named in
[Graph-pair selection](graph-pair-selection.md).

A graph pair is not a pair of chronological execution IDs. Construction
happens before execution tasks receive their final program identities. During
lowering, each stage occurrence gets one alternative forward task and one
alternative backward task per variant.

## Inputs and output

Construction consumes:

- a `StageExample` from model partitioning;
- the stage's FX graph, explicit inputs, input provenance, and mutation
  contract;
- the flattened stage output produced by the representative example;
- the differentiable output positions that seed the vector-Jacobian product;
- whether terminal unit cotangents may be specialized away;
- the retention policy that says what the `save` variant regenerates;
- the builder that defines the variant set.

It returns an immutable `TaskGraphPairs`:

```text
TaskGraphPairs
├── structural_contract
├── root_output_indices
├── reference_option_id
└── variants
    ├── GraphPairVariant("save", memory_budget=1.0, pair=...)
    └── GraphPairVariant("recompute", memory_budget=0.0, pair=...)
```

The record supports any number of uniquely named variants, and the default
builder defines two. “Every legal variant” therefore means every variant that
builder defines, not every mathematically possible cut of the AOT joint graph.

## Accumulating onto gradients that already exist

For each stage, the first backward the step's walk reaches creates the
gradient and every later one contributes to it. So every variant has a second
form, `accumulating()`, which takes those gradients as further arguments and
adds into them -- the addition happens inside the backward task instead of
after it, where no plan accounts for it. `options(accumulates=...)` returns
the forms one microbatch may choose between. Only parameter gradients outlive
a microbatch; a cotangent belongs to the microbatch that produced it, so those
are left alone.

The addition is in place, which is why the task declares a mutation of the
argument rather than a fresh output: the running gradient keeps its storage.
The compiler folds the add into the kernels it generates -- a reduction reads
the running gradient and writes it back as it finishes -- but not into a call
it only makes. A matrix multiply is such a call, writing its result before
anything can read it, so a gradient one computes, moved at most by views on
its way out, is added by the multiply itself: `shadowspill::accumulate_matmul_`
is `C = A @ B + C` in one call, with the running gradient as `C`. The product
is never written out and read back. That is decided from the graph's
operations, where the device has an in-place kernel that accepts the dtypes;
a device without one adds after. An opaque operation's result -- a custom
kernel's -- is still written and then added.

Adding inside the multiply avoids a separate product buffer and addition.
The BLAS kernel controls rounding: it may round only the final sum, or round
the product before adding the running gradient inside the same kernel.
`round_accumulation_once` permits this fusion for gradients narrower than
the multiply's accumulator, such as BF16; it does not guarantee a particular
rounding sequence. With the option off, those gradients are added after the
multiply, preserving that explicit sequence. FP32 gradients use fusion
either way, subject to the usual reduction-order differences.

Which form a microbatch's stage runs follows from the step's data ordering
(`StepDataOrdering.creates`), not from planning, so both forms share one
option ID and every microbatch is offered the same graph-pair choices. Under
the depth-first walk the first microbatch creates everything; under a
reversed walk a pass's last microbatch creates every stage but the paired
last one. The accumulating form is derived on demand rather than captured:
every microbatch of an accumulating step carries both forms, derived once
per structural contract, and a step with a single microbatch never builds,
compiles, or profiles a form it would not run.

## Differentiation roots

For a nonterminal stage, every floating or complex output leaf that requires a
gradient is a differentiation root. For the terminal stage, construction uses
the exported objective's loss position. A missing, nontensor, or
nondifferentiable root is a capture error.

The terminal loss cotangent is structurally known to be one. ShadowSpill may
specialize that unit seed out of the backward task's public object set. This
specialization applies only to the terminal unit seed; intermediate
cotangents remain real task inputs because they carry activation gradients.

### What a cotangent is laid out like

A backward is captured for one tangent layout and is then called directly:
nothing stands between a plan and the compiled artifact. Left to itself the
compiler assumes a tangent arrives strided exactly like the forward output it
belongs to, and restrides any gradient that disagrees on the way in. A stage
boundary has no such step -- the gradient one task publishes is the gradient
the next task is handed -- so a layout assumed rather than known would reach
the compiled kernel as a wrong stride.

Restriding at the boundary is not available either: geometry sizes objects,
alias extents and offsets before any task is compiled, so a copy nobody
planned has nowhere to live. Construction therefore fixes a tangent's contract
as the canonical memory format of the output it belongs to, which is a
function of the graph's structure rather than of the strides one capture
happened to produce. It is the same reasoning that keeps the compiler from
choosing an output's geometry through shape padding.

## Structural deduplication

Before AOT capture, ShadowSpill computes the stage structural contract from:

- the normalized FX graph, with static arguments specialized into it;
- each tensor argument's geometry and position;
- which tensor arguments share storage;
- explicit mutation declarations;
- normalized input provenance, which fixes each argument's role;
- the framework and provider versions the graph would be built under.

`GraphPairStore` keys one task's graph pairs by:

```text
(structural_contract, differentiable_root_positions, specialize_unit_tangents,
 retention_digest)
```

`retention_digest` is the retention policy together with the class it gives
every custom operator in the stage ([below](#save)). PyTorch's own operators
class the same way under one PyTorch build, which the structural contract
already names; a custom operator's class moves when its library registers or
corrects a flop formula, and the digest moves with it, so a policy changed by
the caller or a class changed by a library misses rather than reading back a
pair built under the old one. The manifest beside an entry records the policy
and the classes in the clear.

The first occurrence constructs or restores them. Equivalent later
occurrences reuse its graph code and contracts while rebinding authentic
occurrence-local inputs and input provenance. This makes graph construction
scale with unique structural contracts rather than model positions.

## Constructing one variant

For each configured variant, ShadowSpill calls AOTAutograd with compiler
callbacks that capture the emitted forward and backward FX graphs as
`GraphArtifact` values.

Both variants come from PyTorch's min-cut rematerialization partitioner over
the joint graph, at the two endpoints of its activation-memory budget. Values
the backward is handed are returned by the forward graph and become inputs to
its paired backward graph; the rest of what the backward needs it computes
again. The budget, and the partitioner options the `save` variant sets, are
bound inside the lazy partition callback AOTAutograd invokes, so ambient
Functorch configuration cannot change the generated pair.

### Save

The `save` variant is budget `1.0`: the min-cut over saved bytes, which retains
for the backward what would be expensive to regenerate and regenerates the
rest. It is the reference variant used to establish the canonical public
stage-boundary contract.

What is expensive is decided by arithmetic intensity, in
`shadowspill.pytorch.capture.retention`. Below a device's ridge point an
operator's runtime is set by the bytes it moves rather than by its flops, so
regenerating its result costs one pass over its bytes however many flops it
does: a normalization, an activation, a rotation, a cast, a gather. Above the
ridge the cost per regenerated byte grows with the flops without bound: a
matrix product, an attention, a convolution. Every forward operator is
classed as one or the other by its flops per byte moved against a threshold,
`memory_bound_flops_per_byte`, which the planning request sets and whose
default sits well under any ridge point in use, so the classification does
not depend on the machine and a poor kernel near the boundary still costs
about one pass. The threshold is part of the step's identity in the store.

Flops come from PyTorch's flop counter, which knows every ATen matrix product,
convolution and attention and any custom operator whose library registered a
formula; bytes come from the traced geometry. An ATen or prims operator with
no formula is memory-bound, which is PyTorch's own stance in its denylist. A
custom operator with no formula is *unknown*, and unknown is treated as
compute-bound: a library that declares nothing has its values retained, never
regenerated on a guess, and the pair records which operators those were. A
compute-bound or unknown operator is marked as a value to save before the
partition runs; an operator the model itself already tagged with a checkpoint
policy keeps that tag, and the rule fills in only where nothing was said.

The partitioner's own bans on regeneration -- an allowlist of fusible
operators, a ban on values materialized in the backward, on values used far
apart, on long fusible chains -- encode the premise that regeneration is free
only when a compiler fuses it, which the intensity rule replaces, so they are
lifted for the partition. So is the premise that a parameter is free to keep,
since a plan pays a fetch for every parameter a backward reads.

The min-cut solver decides the rest. A regenerated value's inputs have to be
retained instead, and the solver charges for them, so a value is regenerated
only when the bytes it saves exceed the bytes its regeneration keeps: a
normalization's output goes because its input is kept for the normalization's
own backward; a rotation's output stays because its input is the same size
and is kept for nothing else.

### Recompute

The `recompute` variant is budget `0.0`, the full-recompute endpoint: it
retains the stage's inputs alone, and the backward computes everything else
again. It reads no mark.

A budget strictly between zero and one is PyTorch's knapsack between the two
endpoints over the same marks, and a legal variant the representation already
carries, so adding one changes what the builder emits and nothing downstream
of it.

### Captured pair contract

Each `AotGraphPair` records:

- normalized forward and backward `GraphArtifact` values;
- semantic `TaskStorageContract` values for both tasks;
- the number of forward outputs used only as backward saved values;
- the retention summary: the threshold the partition ran under, the forward
  operators the backward regenerates, and the custom operators no formula
  priced;
- the number of specialized terminal unit tangents.

Construction verifies that AOT emits both sides of the pair, that output and
backward argument arities agree, and that the selected roots actually trigger
a differentiable backward graph.

## Saved-value accounting

AOT “saved values” are backward arguments, but not every saved leaf creates a
new retained activation allocation. `saved_value_footprint()` classifies the
saved storage roots:

| Class | Meaning | New retained program bytes |
|---|---|---:|
| Input root | A saved leaf aliases an existing forward input. | 0 |
| Boundary root | A saved leaf aliases a public stage output already needed by the next stage. | 0 |
| Internal root | A fresh, non-public root exists only to feed the paired backward. | Root's physical extent |

Only internal roots become `retained_alias_group_ids` for the occurrence-level
`TaskAlternativeOption`. This prevents repeated leaves, input passthroughs, and
views of public outputs from being double-counted as activation memory.

The reference and every alternative must preserve the same public stage
boundary. Variants may differ in saved internal roots, forward/backward graph
code, runtimes, workspace, allocation paths, and mutation transition bytes.

## Compilation and profiling

Graph-pair construction produces semantic graph artifacts; it does not assign
task runtimes or workspace from AOT heuristics. The profiling pipeline later
compiles and measures the forward and backward artifact of every unique
variant contract.

**A pair is measured as a pair.** A backward is not a task that can be
measured on its own: its saved inputs are what its forward kept -- the
activations it will need again, the statistics a fused operator needs to
rebuild its result, the random state it drew from -- and they mean something
only together. Inventing them one at a time asks a kernel to undo a forward
pass that never happened, and a kernel entitled to assume its saved state
came from somewhere is not obliged to survive state that did not.

So the forward runs on its own representative inputs and the backward runs
on what came out of it, with only its tangents invented. Nothing decides
which saved values may be invented, because none of them may. One forward
run is shared per forward contract, declared metadata and saved arity, which
is the identity a profile already has.

What the forward saved is kept in the plan's spill pool, not beside it: it can
be as large as the step's activations. It is released when planning is done,
and planning fails when the pool has no room for it.

What a saved value is worth belongs to the forward that made it; how it is
laid out belongs to the backward that reads it, and the two need not agree.
The value is written into the geometry its reader declares, through the
locations that geometry actually has, so a broadcast is filled once and read
many times rather than written many times to one place.

Compilation/profiling records, for both halves of every pair:

- semantic and executable storage-contract digests;
- input, output, mutation, and replacement-transition bytes;
- requested and charged workspace, including individual extents;
- invariant allocation path and bounded dynamic-scratch behavior;
- warmed backend-event runtime samples and stability diagnostics;
- representative-input and profiling-metadata provenance.

Equivalent artifact/profile keys are measured once. Different graph-pair
variants remain distinct whenever their graph, input problem, executable
storage, or profiling metadata differs.

## Lowering into program alternatives

For every stage occurrence and every variant, `TaskBindingResolver` binds the
pair to canonical objects and the training task emitter creates:

```text
variant forward task
        +
variant backward task
        +
fresh internal saved-value aliases
        ↓
TaskAlternativeOption
```

One occurrence-level `TaskAlternativeGroup` contains all of those options. Each
option names exactly the forward/backward task IDs it activates and the
internal alias groups it retains. Tasks for unselected variants remain in the
immutable program but are excluded by `ShadowSpillProgram.selected_tasks()`.

Structural deduplication and occurrence-level choice both survive into the
program and the report: the task-alternative groups carry the exact
occurrence-level task and retained-alias identities that graph-pair selection
fixes, and the plan report keeps every legal variant beside the one each
occurrence ran, with structural profiles stored once and referenced rather than
copied per occurrence. See [unique stages and graph
pairs](../python/plan-report.md#unique-stages-and-graph-pairs) for how to read
that evidence.

## Pseudocode

```text
ConstructGraphPairs(partitioned_export):
    repository = structural graph-pair repository
    stages = []

    for occurrence in partitioned_export.stages:
        roots = differentiable_stage_roots(occurrence)
        terminal = occurrence is final stage
        key = structural_contract(occurrence, roots, terminal,
                                  retention.digest(occurrence.graph))

        graph_pairs = repository.lookup(key)
        if graph_pairs is absent:
            variants = []
            for budget in (1.0, 0.0):
                pair = AOTAutograd(
                    occurrence.graph,
                    roots,
                    partition=min_cut(budget, marks=retention),
                )
                validate_public_boundary(pair)
                classify_saved_storage_roots(pair)
                variants.append(named_variant(budget, pair))
            graph_pairs = TaskGraphPairs(key, roots, variants)
            repository.store(graph_pairs)
        else:
            graph_pairs = rebind_occurrence_values(graph_pairs, occurrence)

        stages.append(DifferentiatedStage(occurrence, graph_pairs))

    return stages

CompileAndProfileGraphPairs(stages):
    artifacts = stable_unique(
        pair.forward and pair.backward
        for every structural variant
    )
    return compile_and_profile_each_unique_artifact(artifacts)

LowerOccurrence(stage, profiles):
    options = []
    for variant in stage.graph_pairs:
        forward = bind_and_emit_forward_task(variant, profiles)
        backward = bind_and_emit_backward_task(variant, profiles)
        retained = fresh_internal_saved_aliases(variant)
        options.append(TaskAlternativeOption(variant.name, [forward, backward], retained))
    return TaskAlternativeGroup(stage.occurrence_id, options)
```

## Fail-closed conditions

Construction rejects a stage when:

- it has no valid differentiable root;
- AOTAutograd does not emit a complete pair;
- a variant changes the public stage-boundary arity or incompatible storage
  semantics;
- alias or mutation relationships cannot be normalized;
- a cached task's structural key or serialized artifact is invalid;
- one variant cannot compile or produce a valid physical profile.

There is no model-family branch and no fallback that infers semantic identity
from allocator pointers or FakeTensor storage identity.

## Implementation map

| Module | Responsibility |
|---|---|
| `shadowspill.pytorch.partition` | Produce ordered stages and authentic examples. |
| `shadowspill.pytorch.graph_pairs.capture` | Choose differentiation roots and bind occurrences to their graph pairs. |
| `shadowspill.pytorch.capture.retention` | Class every operator by arithmetic intensity, mark what a partition retains, and record what it regenerated. |
| `shadowspill.pytorch.graph_pairs.build` | Define the default variant set and invoke AOT capture at each budget. |
| `shadowspill.pytorch.graph_pairs.artifacts` | Immutable pair, variant, task-graph-pairs, and differentiated-stage records. |
| `shadowspill.pytorch.graph_pairs.footprint` | Classify saved input, boundary, and internal storage roots. |
| `shadowspill.pytorch.graph_pairs.store` | Structural cache identity, persistence, and occurrence rebinding. |
| `shadowspill.pytorch.profiling` | Compile and measure each unique forward/backward artifact. |
| `shadowspill.pytorch.lowering.training` | Bind variants to canonical objects and emit program groups. |

Previous: [PyTorch capture and lowering](lowering.md). Next: [The step
artifacts](step.md).
