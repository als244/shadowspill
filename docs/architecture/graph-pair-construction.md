# Graph-pair construction

Graph-pair construction turns one differentiable, partitioned PyTorch stage
into every executable forward/backward alternative that ShadowSpill is willing
to expose to planning. It is a PyTorch frontend operation. It is not
[recomputation selection](recomputation-selection.md), and it is not part of
[PressureFit](pressurefit.md).

The three layers have deliberately different outputs:

| Layer | Unit of work | Output |
|---|---|---|
| Graph-pair construction | One structural stage contract | A `TaskGraphPairs` containing named forward/backward variants |
| Recomputation selection | All occurrence-level `RecomputationGroup` values in one Program | A bounded tuple of complete `RecomputationSelection` assignments |
| PressureFit | One complete assignment plus the Program and machine model | A residency/action schedule with simulated cost |

This separation lets the frontend add another legal graph-pair variant without
changing the framework-neutral recomputation, PressureFit, simulator, or
runtime contracts.

## Vocabulary

| Term | Meaning |
|---|---|
| Stage occurrence | One ordered partition in one captured input geometry or accumulation round. |
| Structural contract | The deterministic graph/input/storage identity shared by equivalent stage occurrences. |
| Graph pair | One mutually compatible AOTAutograd forward graph and backward graph. |
| Variant | A named graph pair produced by one partitioning policy, such as `save` or `recompute`. |
| Task graph pairs | Every configured legal variant for one structural contract. |
| Recomputation group | The occurrence-level Program choice whose options activate one variant's forward and backward tasks. |

A graph pair is not a pair of chronological execution IDs. Construction
happens before execution tasks receive their final Program identities. During
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
- the configured graph-pair variant builders.

It returns an immutable `TaskGraphPairs`:

```text
TaskGraphPairs
├── structural_contract
├── root_output_indices
├── reference_option_id
└── variants
    ├── GraphPairVariant("save", memory_budget=None, pair=...)
    └── GraphPairVariant("recompute", memory_budget=0.0, pair=...)
```

The record supports any positive number of uniquely named variants.
The current default builder emits exactly two. “Every legal variant” therefore
means every option configured by the builder, not every mathematically
possible cut of the AOT joint graph.

## Differentiation roots

For a nonterminal stage, every floating or complex output leaf that requires a
gradient is a differentiation root. For the terminal stage, construction uses
the exported objective's loss position. A missing, nontensor, or
nondifferentiable root is a capture error.

The terminal loss cotangent is structurally known to be one. ShadowSpill may
specialize that unit seed out of the backward task's public object set. This
specialization applies only to the terminal unit seed; intermediate
cotangents remain real task inputs because they carry activation gradients.

## Structural deduplication

Before AOT capture, ShadowSpill computes the stage structural contract from:

- normalized FX graph semantics;
- tensor geometry and static arguments;
- explicit mutation declarations;
- input roles and provenance;
- alias and storage contracts.

`GraphPairRepository` keys one task's graph pairs by:

```text
(structural_contract, differentiable_root_positions, specialize_unit_tangents)
```

The first occurrence constructs or restores them. Equivalent later
occurrences reuse its graph code and contracts while rebinding authentic
occurrence-local inputs and input provenance. This makes graph construction
scale with unique structural contracts rather than model positions.

## Constructing one variant

For each configured variant, ShadowSpill calls AOTAutograd with compiler
callbacks that capture the emitted forward and backward FX graphs as
`GraphArtifact` values.

### Save

The `save` variant uses AOTAutograd's ordinary partitioning path. Values needed
by backward are returned by the forward graph and become inputs to its paired
backward graph. Its `memory_budget` is `None`, and it is the reference variant
used to establish the canonical public stage-boundary contract.

### Recompute

The `recompute` variant uses PyTorch's min-cut rematerialization partitioner.
The current endpoint sets `activation_memory_budget=0.0`, the full-recompute
endpoint. The budget is bound inside the lazy partition callback so ambient
Functorch configuration cannot change the generated pair.

The representation also supports intermediate budgets strictly
between zero and one. Adding such variants changes the configured construction
inventory, not the downstream type system. Budget `1.0` reproduces the
save-everything endpoint and is not exposed as a recomputation alternative by
the current builder. The current default emits no intermediate choices.

### Captured pair contract

Each `AotGraphPair` records:

- normalized forward and backward `GraphArtifact` values;
- semantic `TaskStorageContract` values for both tasks;
- the number of forward outputs used only as backward saved values;
- whether min-cut recomputation was used;
- the number of specialized terminal unit tangents.

Construction verifies that AOT emits both sides of the pair, that output and
backward argument arities agree, and that the selected roots actually trigger
a differentiable backward graph.

## Saved-value accounting

AOT “saved values” are backward arguments, but not every saved leaf creates a
new retained activation allocation. `saved_value_footprint()` classifies the
saved storage roots:

| Class | Meaning | New retained Program bytes |
|---|---|---:|
| Input root | A saved leaf aliases an existing forward input. | 0 |
| Boundary root | A saved leaf aliases a public stage output already needed by the next stage. | 0 |
| Internal root | A fresh, non-public root exists only to feed the paired backward. | Root's physical extent |

Only internal roots become `retained_alias_group_ids` for the occurrence-level
`RecomputationOption`. This prevents repeated leaves, input passthroughs, and
views of public outputs from being double-counted as activation memory.

The reference and every alternative must preserve the same public stage
boundary. Variants may differ in saved internal roots, forward/backward graph
code, runtimes, workspace, allocation paths, and mutation transition bytes.

## Compilation and profiling

Graph-pair construction produces semantic graph artifacts; it does not assign
task runtimes or workspace from AOT heuristics. The profiling pipeline later
compiles and measures the forward and backward artifact of every unique
variant contract independently.

Compilation/profiling records, for both halves of every pair:

- semantic and executable storage-contract digests;
- input, output, mutation, and replacement-transition bytes;
- requested and charged workspace, including individual extents;
- strict allocation-core and bounded dynamic-scratch behavior;
- warmed backend-event runtime samples and stability diagnostics;
- representative-input and profiling-metadata provenance.

Equivalent artifact/profile keys are measured once. Different graph-pair
variants remain distinct whenever their graph, input problem, executable
storage, or profiling metadata differs.

## Lowering into Program alternatives

For every stage occurrence and every variant, `TaskBindingResolver` binds the
pair to canonical objects and the training task emitter creates:

```text
variant forward task
        +
variant backward task
        +
fresh internal saved-value aliases
        ↓
RecomputationOption
```

One occurrence-level `RecomputationGroup` contains all of those options. Each
option names exactly the forward/backward task IDs it activates and the
internal alias groups it retains. Tasks for unselected variants remain in the
immutable Program but are excluded by `Program.selected_tasks()`.

Structural deduplication and occurrence-level choice are both preserved:

- `PlanReport.diagnostics.unique_stages` describes one structural contract and all
  profiled graph pairs;
- the execution-task map identifies each occurrence and its selected variant;
- the Program's recomputation groups carry the exact occurrence-level task and
  retained-alias identities consumed by recomputation selection.

## PlanReport diagnostics

The report preserves graph-pair evidence at both structural and chronological
levels:

| Record | Directly available information |
|---|---|
| `PlanUniqueStage` | Structural key, semantic module targets, occurrence count, and every legal `PlanGraphPair`. |
| `PlanGraphPair` | Variant name, min-cut memory budget, recomputation flag, saved-value counts/classes/bytes, and paired forward/backward profiles. |
| `PlanGraphProfile` | Profiled runtime; input, mutation, output, replacement-transition, workspace, and persistent extents; storage/layout identities; allocation contract and timeline. |
| Task-to-stage map | Chronological execution ID, semantic task, structural stage, occurrence, and selected graph-pair variant. |

This makes the selected variant a direct task lookup while retaining every
unselected alternative for comparison. Structural profiles are stored once;
occurrence mappings reference them rather than duplicating their measurements.

## Pseudocode

```text
ConstructGraphPairs(partitioned_export):
    repository = structural graph-pair repository
    stages = []

    for occurrence in partitioned_export.stages:
        roots = differentiable_stage_roots(occurrence)
        terminal = occurrence is final stage
        key = structural_contract(occurrence, roots, terminal)

        graph_pairs = repository.lookup(key)
        if graph_pairs is absent:
            variants = []
            for policy in configured_variant_policies:
                pair = AOTAutograd(
                    occurrence.graph,
                    roots,
                    partition_policy=policy,
                )
                validate_public_boundary(pair)
                classify_saved_storage_roots(pair)
                variants.append(named_variant(policy, pair))
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
        options.append(RecomputationOption(variant.name, [forward, backward], retained))
    return RecomputationGroup(stage.occurrence_id, options)
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
| `shadowspill.pytorch.graph_pairs.build` | Define the configured variant set and invoke AOT capture. |
| `shadowspill.pytorch.graph_pairs.artifacts` | Immutable pair, variant, task-graph-pairs, and differentiated-stage records. |
| `shadowspill.pytorch.graph_pairs.footprint` | Classify saved input, boundary, and internal storage roots. |
| `shadowspill.pytorch.graph_pairs.store` | Structural cache identity, persistence, and occurrence rebinding. |
| `shadowspill.pytorch.profiling` | Compile and measure each unique forward/backward artifact. |
| `shadowspill.pytorch.lowering.training` | Bind variants to canonical objects and emit Program groups. |

Previous: [PyTorch capture and lowering](lowering.md). Next:
[Recomputation selection](recomputation-selection.md).
