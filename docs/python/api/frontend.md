# PyTorch build policies

Build policies control captured computation. They do not choose runtime spill
placement or allocate execution pools. See the [phase guide](../phases.md) for
workflow and the [low-level runtime reference](runtime.md) for pool interfaces.

| Control | Changes during build | Separate decision |
| --- | --- | --- |
| `partition` | Checkpoint-stage boundaries | Internal execution regions |
| `chunking` | Independent axes, execution regions and chunk sizes | What crosses the checkpoint boundary |
| `graph_pair_policy` | Saved state in each SAVE/RECOMPUTE variant | Which variant runs and where its state resides |
| `gradient_destination_rules` | Eligible writes into existing gradient buffers | Precision and ownership must still permit the write |
| `ChunkRegions.lifetimes` | Explicitly permitted movement, compaction and local replay | No implicit replay of expensive operations |

Changing these controls can change task identity, memory and timing. Build and
profile each candidate; planning compares the resulting measured alternatives.
Start with the [custom partitioning example](../../examples/custom-partitioning.md).

## Contents

- [Inputs, objectives, and partitioning](#inputs-objectives-and-partitioning)
- [Graph-pair policies](#graph-pair-policies)
- [Custom gradient destinations](#custom-gradient-destinations)
- [PartitionPolicy](#partitionpolicy)
- [Axis chunking policies](#axis-chunking-policies)
- [Build and measurement policies](#build-and-measurement-policies)

## Inputs, objectives, and partitioning

`TensorSpec` is storage-free fixed tensor geometry for planning. It records
shape, dtype, optional stride, `requires_grad`, and layout.

An objective may return a scalar loss tensor, `(loss, metrics)`, or `ObjectiveResult`. A bare
tensor becomes the corresponding `StepResult.losses` entry and has
`metrics=None`. `ObjectiveResult` explicitly names the differentiable `loss`
and arbitrary nondifferentiated `metrics`; each becomes the corresponding
entry in the two per-round `StepResult` tuples. ShadowSpill validates this
contract during capture rather than inferring a loss from model output names.

| `ObjectiveResult` field | Contract |
|---|---|
| `loss` | One floating-point or complex scalar tensor that participates in backward. |
| `metrics` | Optional nondifferentiated metadata returned to the caller for the same accumulation round. |

`metrics` may be a pytree. Tensor leaves are detached task outputs. Static
leaves must be copyable, and the pytree structure must remain fixed across the
captured workload. Metrics never contribute to backward, are not aggregated
across accumulation rounds, and are unrelated to planning reports or runtime-trace
diagnostics. Applications that do not need auxiliary outputs should return the
loss tensor directly.

## Graph-pair policies

Pass `graph_pair_policy=policy` to `build_forward_backward` or `build_step` to
control each checkpoint stage's saved-state cuts independently. `None` means
`GraphPairPolicy()`. Import `GraphPairPolicy`, `RetentionPolicy`, `RetentionRule`
and `OperatorClass` from `shadowspill.pytorch`. Forward-only builds do not take
this argument because they have no backward saved state.

`GraphPairPolicy(save=RetentionPolicy(), recompute=RetentionPolicy(default="recompute"))`
returns an immutable pair of capture policies:

| Argument | Type | Contract |
| --- | --- | --- |
| `save` | `RetentionPolicy` | Policy for the `save` graph pair. The default cost policy permits removals from backward-required forward state. Set `default="retain"` to keep that state without cost-based removals. |
| `recompute` | `RetentionPolicy` | Policy for the `recompute` graph pair. The default retains checkpoint inputs and supported routing choices, regenerating other intermediates. Add retain rules to preserve selected operator results and required auxiliary outputs. |

The policies are independent, with no requirement that one cut save more bytes
than the other. They describe state needed by backward, not every forward
intermediate: unused results are removed. Required inputs, effects and mutations
still constrain valid cuts. Retained values are eligible for planner-controlled
host spill; retention does not require device residency.

`RetentionPolicy(memory_bound_flops_per_byte=16.0, rules=(), default="cost")`
returns an immutable policy for either graph pair:

| Argument | Type | Contract |
| --- | --- | --- |
| `memory_bound_flops_per_byte` | finite nonnegative `float` | Under the cost default, operations at or below this arithmetic intensity may be regenerated. This is not a memory budget or measured execution time. |
| `rules` | `tuple[RetentionRule, ...]` | Exact operator overrides, optionally scoped to source modules. Each target/scope pair may appear once; conflicting matches fail capture. |
| `default` | `"cost"`, `"retain"` or `"recompute"` | `cost` retains compute-bound and unpriced custom operations, permitting memory-bound replay. `retain` keeps backward-required results. `recompute` regenerates unselected intermediates from checkpoint inputs and selected retained results. Correctness-required state can also remain saved. |

`RetentionRule(target, action, *, module_pattern=None)` identifies one exact forward operator overload
name, such as `"aten.mm.default"`. `target` is a nonempty `str`; `action` is
`"retain"` or `"recompute"`. The rule includes tuple extraction, so backward can
receive the selected operation's required auxiliary outputs as well as its
visible result. Unused outputs are removed. The rule does not request saving the
operation's inputs: those can be regenerated from earlier saved state.
Targets absent from an individual stage have no effect there, allowing one
policy to apply across different stages. Inspect captured graphs and the
graph-pair manifest to verify the intended operators and regenerated work.

`retain` prevents the matching forward computation from being replayed in
backward; it does not require a separate saved port containing its raw result.
The cut can save that result, save a sufficient derived value instead, or omit
local backward state when the result is needed only by forward consumers.
For example, a compact count statistic can cross the cut as integer counts or
normalized floating-point frequencies. An existing forward output does not
automatically require a duplicate backward save. Inspect the actual saved
ports and their storage backings to determine retained bytes; the rule name
alone does not determine memory use. Profiling and residency planning still
apply to the resulting graph pair.

`module_pattern` is an optional case-sensitive shell pattern matched against
the operation's source module paths. Any matching ancestor includes its
descendants. For example, `"model.blocks.*.experts"` selects operations in each
block's `experts` module and its children; `*` can match dots. `None` applies
the rule everywhere. Module paths come from export, survive differentiation
through structural source indices, and also identify operations introduced by
decomposing that source operation. The target still names the exact overload
in the differentiated forward graph. Missing required source provenance fails
capture rather than silently turning a scoped rule into a global rule.

Scope avoids removing useful saved cuts in unrelated operations. For example,
if an expert implementation produces chunk-local state through an add:

```python
save_policy = RetentionPolicy(
    rules=(RetentionRule(
        "aten.add.Tensor", "recompute", module_pattern="model.blocks.*.experts"
    ),)
)
```

Other modules keep the cost-based default. A global add-removal rule can force
both large projection operands to remain saved instead of their single sum.
Choose scopes from the captured source paths, and verify the resulting saved
tensors and chunk-local legality. This control does not make an otherwise
invalid saved output legal. Rules have no ordering precedence: overlapping
matches with different actions are errors. Use the policy's `default` for a
broad choice and explicit rules for exceptions.

Explicit rules override the default classification. Model checkpoint annotations
take precedence over the default; contradictory explicit rules fail capture.
Capture also refuses a requested rule the resulting saved cut cannot satisfy,
for example when correctness requires saving a value requested for replay.
Selective cuts retain results of operations tagged as using RNG state or ordered
effects. Explicit replay rules for those operations fail capture: use retained
results or expose replayable state through ordinary tensor inputs. This avoids
implicit generator objects in saved graph signatures.
Rules select computations, not arbitrary views of their storage. An aliased
saved output can retain its complete backing allocation.

For example, compare two candidates with the same save policy and toggle
retention of selected sequence operators in recompute. `attention_operators`
contains the exact dense-attention overload names supplied by the provider:

```python
from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule

save_policy = RetentionPolicy()
graph_pair_policies = {
    "inputs_only": GraphPairPolicy(save=save_policy),
    "attention_retained": GraphPairPolicy(
        save=save_policy,
        recompute=RetentionPolicy(
            default="recompute",
            rules=tuple(RetentionRule(op, "retain") for op in attention_operators),
        ),
    ),
}
```

Each candidate offers its two graph pairs to planning. PressureFit chooses
save/recompute per stage and plans residency for that candidate; program search
compares the resulting admitted Plans across candidates. The chunk policy's
`variants` can restrict which outer variants are offered, independently of the
retention policies. Forward sinks require a `save` choice, so changing only
`recompute` leaves their paired-loss save policy intact.

To let PressureFit choose whether to retain dense attention **within one
Program**, put the selective policy on `save` and leave `recompute` at its
default. Both variants also preserve ATen top-k and
[provider-declared routing choices](../../architecture/routing-retention.md#provider-owned-default):

```python
attention_only_save = GraphPairPolicy(
    save=RetentionPolicy(
        default="recompute",
        rules=tuple(RetentionRule(op, "retain") for op in attention_operators),
    ),
    recompute=RetentionPolicy(default="recompute"),
)
```

Here `attention_operators` contains only the dense-attention provider overloads
whose results should be retained. Both variants must remain enabled. The
`save` name does not force the cost policy: its default is overridden by this
explicit policy. Required auxiliary outputs, checkpoint inputs and effect state
still constrain the cut. Attention inputs such as projected Q/K/V may be
regenerated; inspect the captured backward to verify what replay is avoided.

The three candidate pairs offer different per-stage choices:

| Candidate | SAVE cut | RECOMPUTE cut |
| --- | --- | --- |
| Cost / inputs | Cost-based retention | Checkpoint inputs |
| Cost / attention | Cost-based retention | Selected attention state and checkpoint inputs |
| Attention / inputs | Selected attention state and checkpoint inputs | Checkpoint inputs |

All three pairs additionally retain supported [routing choices](../../architecture/routing-retention.md)
by default, including in cuts labeled "inputs" above. An explicit `RetentionRule`
can override a routing operator independently for each variant. These saved
tensors use normal placement and spill planning.

The last pair removes other cost-retained results from its SAVE alternative.
In stages without selected attention operators its two cuts may coincide. It
can reduce saved state, but may repeat more non-attention work than the cost
alternatives. Compare their measured admitted Plans rather than assuming an
ordering. Changing SAVE also applies to the mandatory terminal SAVE choice;
terminal chunking and backward requirements still need to validate. A separate
Program candidate does not add a third per-stage variant to another Program.

| Phase/control | Responsibility |
| --- | --- |
| `partition` | Chooses checkpoint boundaries and their inputs. |
| `graph_pair_policy` during build | Constructs each variant's saved-state cut and backward graph. Both policy identities, operator classifications, resolved source scopes and graph contracts key capture reuse. |
| `chunking` during build | Chooses execution regions, axes, sizes and permitted outer variants. It does not change the selected outer saved-state cut. |
| Profiling | Measures each final complete task, including its chunk loop, assembly and provider workspace. |
| Planning/PressureFit | Chooses among measured allowed stage variants and decides residency, transfers and execution order under device and spill budgets. Retained does not mean device-resident. |
| Candidate search | Plans the declared measured candidates at each budget and selects the lowest admitted simulated update time. |
| Execution | Uses the exact selected saved Plan without invoking retention callbacks or resizing chunks. |

Changing either policy requires building and profiling the resulting task forms;
CPU replanning can reuse saved candidates for different budgets and transfer
costs. Equal final task contracts can share code/profiles; a changed saved cut
cannot borrow another cut's memory evidence. See
[graph-pair construction](../../architecture/graph-pair-construction.md#save),
[artifact metadata](artifacts.md#aot-graph-pair-cache), and the
[preparation settings](../../../docs/python/advanced/service-client.md).

For unusual cost classification, subclass `RetentionPolicy.classify(node)`.
It receives a traced FX node and returns `OperatorClass.MEMORY_BOUND`,
`OperatorClass.COMPUTE_BOUND`, `OperatorClass.UNKNOWN`, or `None` for a nonoperator.
Explicit rules and the retain/recompute defaults resolve before cost classification.
Override `identity()` with a deterministic JSON-compatible mapping containing
every extra configuration value; include `super().identity()`. Classification
must depend only on traced metadata and that configuration, never live payloads
or runtime memory pressure. `digest(graph_module)` combines this identity with
custom-operator classifications for capture reuse.

## Custom gradient destinations

`build_step` and `build_forward_backward` accept `gradient_destination_rules`,
an empty tuple by default. These rules let an opaque backward operator add its
contribution directly into existing parameter-gradient storage. The first
contribution creates the gradient; later uses across checkpoint regions,
execution regions, chunks and microbatches reuse that destination. One replacement
call handles its selected outputs together; it must not repeat the backward
computation for each output.

The public types are exported from `shadowspill.pytorch`:

| Type | Contract |
| --- | --- |
| `GradientDestinationRule` | Names `target` and `replacement` dispatcher overloads, eligible `output_indices`, a positive `version`, and an optional build-time `mapper`. The replacement receives the original arguments plus mutable `destinations`, `output_indices`, and `contract_version`. |
| `GradientDestinationContext` | The mapper's input: the FX `producer`, eligible `destinations`, and the `fuse_narrow_gradient_accumulation` precision permission. |
| `GradientDestination` | One eligible producer `output_index`, its final `gradient`, existing `destination`, and `operations` from that gradient back to the raw producer output. |
| `GradientDestinationMapping` | The mapper's result: selected `output_indices`, corresponding sequences of destination `views`, and provider-specific `keywords` passed as real operator arguments. Returning `None` declines fusion. |
| `DestinationView` | A storage-preserving `target` operation and its `arguments`, applied to a supplied destination. It must remain writable without a copying reshape or overlapping elements. |

Without a mapper, the rule handles reversible views at unchanged dtype. A custom
mapper must preserve layout, scaling, casts, rounding and padding, honor the
precision permission, and be deterministic without editing the supplied FX graph.
Unmatched or unsupported outputs keep ordinary accumulation. The accumulating
overload preserves unconsumed outputs and may return an empty tensor, `None`, or
the destination at a consumed position, subject to its dispatcher and
functionalization contract. Fixed Tensor schemas with empty consumed outputs
are supported by automatic functionalization.

Rules are build inputs, independent of checkpoint retention and execution axes
or chunk sizes. Replacement calls and contract versions enter artifact identity;
callbacks are not consulted during saved execution. Their kernels still require
normal compilation, profiling and physical admission, including any internal
temporaries. Chunked custom overloads also need `OperatorRule` axis contracts.
See [custom gradient destinations](../../architecture/custom-gradient-destinations.md)
for mutation schemas, ownership restrictions, examples and provider qualification.

## PartitionPolicy

`PartitionSpec` accepts `"auto"`, `"whole"`, or an object implementing the
public `shadowspill.pytorch.PartitionPolicy` protocol:

```python
class MyPartition(PartitionPolicy):
    def assign_stages(self, graph_module, module) -> Mapping[str, int]:
        ...
```

Return every executable FX node name exactly once, excluding `placeholder`,
`get_attr` and `output`. Labels must be nonnegative integers, not booleans;
labels need not be consecutive, but each occupies one contiguous graph interval.
The callback must not modify the graph/model/metadata or execute model kernels.
Use deterministic structural rules and a stable configuration representation.
Subclassing the protocol is optional: implementing the method is sufficient.

Pass the same policy as `partition=policy` to `build_forward`, `build_forward_backward`,
or `build_step`. Direct API calls do not infer policy from a model name.

The policy defines forward-stage boundaries. ShadowSpill derives object inputs,
outputs, explicit mutations and aliases, then constructs backward alternatives
and profiles workspace. It does not let the policy split an opaque custom op,
choose residency, supply byte counts, or select save/recompute behavior.
Malformed mappings fail with `CaptureError` before compilation; later capture
may also reject unsupported differentiation or mutation boundaries.

See the [architecture contract](../../architecture/partitioning.md),
[standalone example](../../examples/custom-partitioning.md), and
[model-scale example](../../examples/custom-partitioning.md#a-model-scale-example).


### IsolateOperations

`IsolateOperations(targets, base="auto")` is a `PartitionPolicy` that gives each
selected operation and its tuple-unpacking nodes a separate stage. Use it to
separate sequence operations from surrounding independent computation. `targets` is a
`tuple[str, ...]` of exact operator overload names in the exported graph; `base`
is a `PartitionSpec` supplying the other cuts. Overlapping intervals merge.
Other nodes inside an interval stay with it to preserve execution order.
`assign_stages(graph_module, module)` returns the normal node-name mapping.

## Axis chunking policies

Pass `chunking=policy` to `build_forward`, `build_forward_backward` or
`build_step` to run selected effective stages in axis slices. The default `None`
leaves stage execution unchanged. The public names below are exported from
`shadowspill.pytorch`, as well as `shadowspill.pytorch.chunking`.

Each region chunks one logical axis at a time. Its tensors can map that axis
to different physical dimensions, including factored dimensions after a
reshape. The axis can represent tokens, independent heads, batch items, or
another independent domain. Operations must preserve chunk independence or
provide a supported additive reduction; this is not unrestricted multidimensional
tiling. Each execution region chooses its own axis and chunk size.

`ChunkRegions(partition, chunking, variants=("save", "recompute"), lifetimes=None)` selects
execution regions **inside** each checkpoint stage. The builder's `partition=`
still defines checkpoint boundaries. The arguments are:

| argument | type | meaning |
|---|---|---|
| `partition` | `PartitionSpec` | Assigns contiguous execution regions on each explicit forward or backward graph. Its `module` argument is that task's graph module; use graph metadata and operator targets, rather than source-model children. |
| `chunking` | `ChunkPolicy` | Receives each region's physical tensor ports and chooses its own chunk size and chunk axes, or `None` for whole-region execution. Backward regions receive separate callbacks. |
| `variants` | `tuple[str, ...]` | Distinct outer training choices from `"save"` and `"recompute"`. Each selected child contract must permit the outer choice. The terminal training stage remains unchanged with its mandatory save choice. |
| `lifetimes` | `ExecutionLifetimes \| None` | Optional operation scheduling on the complete explicit task, before gradient destination routing and execution partitioning. Default `None` preserves the captured schedule. Only outer `ChunkRegions` accepts it. |

The callback can choose different sizes by operator, module path, direction,
input geometry, or `structural_contract`. Equivalent child contracts share
compilation; each different resolved composition receives its own profile key.
Execution-region intermediates are local task workspace. They do not add outer
checkpoint inputs, saved values, or independently scheduled tasks. Cuts must
pass flat tensor ports: keep tuple unpacking with its producer.

```python
from shadowspill.pytorch import ChunkRegions, IsolateOperations

# The caller's axis_policy can return a different AxisChunks for every region.
def execution_policy(axis_policy, sequence_operator_targets):
    return ChunkRegions(
        partition=IsolateOperations(sequence_operator_targets, base="whole"),
        chunking=axis_policy,
        variants=("recompute",),
    )
```

### Execution lifetimes

`ExecutionLifetimes(rules=(), readonly=(), select=None, *, layout=())` controls
operation placement and internal view compaction inside a checkpoint task. It
can move readonly work once, copy selected views into smaller backing storage,
and explicitly repeat selected primal work near consumers. Replay has no default
selections: expensive work repeats only when explicitly selected. These controls
do not change the saved-state cut or request device/host residency. Axis selection
and chunk sizes still belong to `AxisChunks`; retention belongs to
`GraphPairPolicy`; PressureFit chooses variants and residency from measured tasks.

`LifetimeRule(target, *, move=None, barrier=None, compact=None, replay=None, source_target=None,
module_pattern=None)` selects one exact explicit operator overload. Optional
`source_target` matches the original source operation; `module_pattern` matches
any attributed source module path with a case-sensitive shell pattern. An
unattributed operation does not match a source-scoped rule. Each rule must set
at least one of these choices:

| choice | meaning |
|---|---|
| `move=True` | Permit the readonly operation to move once toward its first consumer, subject to effects, alias writes and protected lifetimes. `False` explicitly disables this permission. |
| `barrier=True` | Keep this operation in place and require that every proposed move does not increase the internal backing bytes live immediately before it. `False` leaves it unprotected. A barrier takes precedence over movement and view preparation. |
| `compact=True` | Prepare this internal alias view near its producer and consider a contiguous copy for later execution-region consumers. Related selected views are considered jointly. At least one protected backing set must shrink and none may grow. Unsupported candidates remain un-copied with recorded reasons. |
| `replay=True` | Permit repeating this primal operation in later consumer regions. Inputs must still have their original versions; readonly and value/alias checks must pass. `False` excludes it, including implicit tuple extraction from a selected producer. |
| `None` | Leave that choice unspecified. Conflicting explicit values from matching rules or the callback fail capture. Rule order does not resolve conflicts. |

```python
from shadowspill.pytorch import (
    ChunkRegions, ExecutionLifetimes, IsolateOperations, LifetimeRule,
)

def scheduled_regions(axis_policy, sequence_targets):
    return ChunkRegions(
        IsolateOperations(sequence_targets, base="whole"),
        axis_policy,
        lifetimes=ExecutionLifetimes(rules=(
            LifetimeRule("aten.sin.default", move=True,
                         module_pattern="model.blocks.*"),
            *(LifetimeRule(target, barrier=True) for target in sequence_targets),
        )),
    )
```

For unusual selection logic, `select(operation)` returns
`LifetimeSelection(move=None, barrier=None, compact=None, replay=None)` or `None`. The frozen
`LifetimeOperation` contains `index`, exact `target`, optional `source_target`,
`module_paths`, `effect_fence`, task `kind`, `variant`, `stage_id`, `checkpoint_stage_id`,
`phase` (`"primal"`, `"derivative"` or `"unknown"`), and
flattened `inputs`/`outputs` tensor geometries (`TensorGeometry | None`). It
contains no tensors or mutable graph nodes. The callback must be deterministic
and read-only; do not run kernels or mutate global capture state from it. It
runs once per operation when that task occurrence is built. Profile, plan and
cold execution use the saved result without calling it again.
`effect_fence` explains a mutation, RNG or uncertified opaque effect, or is
`None` for certified readonly work. Absence of an effect fence is not permission
to cross an aliased writer or increase a protected backing set.

Unknown opaque operations are effect fences. A versioned
`ReadonlyOperator(target, version)` in `readonly` certifies deterministic,
readonly execution for that exact overload. It cannot override mutation,
RNG/effect registrations, or unproved alias ownership. An `OperatorRule` for
axis independence or an operator's cost classification is not this effects
contract. Conditional or unknown alias returns are conservatively refused
when lifetime analysis is needed. Parameter-gradient conversion barriers remain
protected regardless of user movement choices.

Provider adapters may supply these contracts while leaving rule/selector choices
to the caller. For example, the `shadowspill.integrations.mlops.fla_execution_lifetimes` factory
certifies specific FLA value operands; the [training client](../client.md)
uses them for movement and internal-view compaction without extra primal replay.
The core `ChunkRegions` default remains `lifetimes=None`; it infers no provider
semantics from a model name or operator name.

Compaction changes storage layout and therefore needs a separate proof from
readonly execution. `LayoutIndependentOperator(target, version, inputs)` in
`layout` certifies value semantics for specified dispatcher argument indices
(including keyword arguments and tensor lists). Replacing those arguments by
equal logical tensors with different strides, offsets, backing spans or alias
relationships must preserve every initialized logical output value, shape and
dtype. The result layout may differ; the pass follows layout changes through
fresh computations as well as alias views. Bitwise equality across different
kernel layouts is not promised by this contract; ordinary numerical qualification
still applies. A provider needs both readonly and layout contracts. Neither can
override mutation, RNG or unsupported ATen storage semantics.

For example, add `LifetimeRule("aten.slice.Tensor", compact=True)` to consider
escaping slices, while marking sequence operators with `barrier=True`. A callback
can select other view/tuple operations. Selected views may be prepared earlier
even when `move=False`; use `barrier=True` to pin an operation. Input/parameter
backings, outer aliases, and written backings cannot compact. With no protected
barriers there is no qualifying memory reduction, so copies are not applied.

The pass conservatively qualifies known value-based ATen operations. Raw
`as_strided`, dtype-reinterpreting views and unknown layout-sensitive consumers
are refused when their input layout changes. An unchanged complete dense fresh
result ends propagation; a fresh result with changed strides does not. A failed
candidate is discarded as a group, and the task's outer storage contract must
remain identical. Execution partition rules are resolved before considering
cross-region uses, then resolved again after transformation and gradient routing;
they must cover inserted `aten.clone.default` operations too.

Replay runs after movement and before view preparation/compaction. It uses the
captured differentiation provenance; operations with missing provenance in a
backward task cannot replay. Explicit forward/inference tasks identify primal
work when no differentiation tag is present. Structural tuple extraction follows
a selected producer automatically unless explicitly excluded. Each selected
operation is copied at most once per consumer region, and barrier-containing
regions never receive copies. `replay=True` does not authorize a barrier itself,
a gradient conversion, an effect, or an externally observed/written result.

Readonly execution alone also cannot justify splitting an alias relationship.
The same input-value contracts protect replay consumers, including an operation
that is itself being replayed with copied dependencies. Dependencies must still
represent the original values at the copied computation; disjoint known tensor
writes may be crossed, but overlapping writes, RNG and unknown effects block it.
Every proposal must preserve the outer storage contract and avoid increases at
all protected memory points. Intermediate proposals need not shrink a live set:
redirecting a later consumer may release the last use of an original backing.
With no selected barriers this local memory check is empty, so physical admission
of the final measured composition remains essential. The profiler measures every
extra computation. A replayed view inherits its original compaction permission;
selectors are never called again for generated nodes.

Resolved operation indices, permissions, used contract versions and scheduling
decisions enter `execution_regions["lifetimes"]` in task metadata and executable
identity. `schedule`, `replay`, `views` and `compaction` records use indices from their
respective input order; movement/view records include their resulting order,
and replay records map original source indices to final copy indices and regions.
Occurrence explanations appear separately in the
`execution_lifetimes` list, including source module paths and semantic backing
counts. These counts exclude task inputs, opaque workspace and allocator
overhead; the complete finalized composition still needs profiling and physical
admission. Different module names with identical resolved behavior can reuse
compilation. Callbacks and occurrence names are not serialized into executable
identity.

Only the outer `ChunkRegions` may specify lifetimes. Setting it inside
`AxisChunks.regions` is refused so scheduling always sees the complete
checkpoint task once. Replay permission is independent of ordinary graph-pair
recomputation and is empty by default.

`ChunkPolicy` implements `configure_stage(stage: StageInfo) -> AxisChunks | None`
to select an
execution contract after partitioning and control-stage folding. Return `None`
to leave a stage intact. The callback must be deterministic and read-only;
it must not run model kernels or mutate capture metadata. Unknown operations
in a selected region fail capture rather than silently running unchunked.

<!-- source-signature: packages/engine/src/shadowspill/pytorch/chunking/policy.py:ChunkPolicy.configure_stage -->
```text
configure_stage(stage: StageInfo) -> AxisChunks | None
```

`StageInfo` provides these read-only fields:

| field | type | meaning |
|---|---|---|
| `stage_id` | `str` | This effective stage occurrence's identifier. |
| `module_paths` | `tuple[str, ...]` | Source module paths present in the current occurrence. Explicit forward/backward tasks resolve these through source-operation provenance; generated operations without provenance add no path. |
| `inputs` | tuple of tensor geometries or `None` | Full input shapes, strides, dtype and storage offsets, indexed by effective stage input position. Static values have no tensor geometry. |
| `input_roles` | `tuple[TaskInputRole, ...]` | Provenance roles such as parameter, buffer, activation or control. |
| `operators` | `tuple[str, ...]` | Exact operator overload names appearing in this stage. |
| `terminal` | `bool` | Whether this is the terminal training stage, which generic axis chunking excludes. Forward-only stages have `False`. |
| `kind` | `str \| None` | For an execution region, the explicit task direction, including `"forward"`, `"backward"` or `"inference"`. A source-stage callback receives `None`. |
| `checkpoint_stage_id` | `str \| None` | The owning checkpoint occurrence for an execution-region callback. |
| `structural_contract` | `str \| None` | Unchunked execution-region structural digest, usable for exact per-contract overrides. It excludes the subsequently selected chunk size. |
| `input_domains` | `tuple[Mapping[str, ChunkAxis], ...]` | Known named axis mappings for each execution-region input, including flattened saved tensors and backward cotangents. Empty for source-stage callbacks. |

`AxisChunks(chunk_size, inputs, variants=("save", "recompute"), rules=(), domain=None, regions=None)` returns
an immutable chunk specification:

| argument | type | meaning |
|---|---|---|
| `chunk_size` | positive `int` | Maximum logical axis positions per invocation. An uneven tail gets its own compiled geometry. |
| `inputs` | `Mapping[int, int \| ChunkAxis]` | Effective stage input positions and their chunk axes. All selected inputs must describe the same positive logical axis extent. Remaining inputs are shared. |
| `variants` | `tuple[str, ...]` | Nonempty distinct training retention choices from `"save"` and `"recompute"`. This restricts stage alternatives, independently of residency. Mandatory endpoint choices must remain available. There is no backward retention choice in forward-only builds. |
| `rules` | `tuple[OperatorRule, ...]` | Caller-certified contracts for opaque operators in this stage. Targets must be unique. |
| `domain` | `str \| None` | Optional nonempty name for the logical coordinates, such as `"tokens"` or `"heads"`. Matching inherited input mappings are included automatically; an explicit conflicting mapping is refused. `None` leaves mappings local to this region. |
| `regions` | `ChunkRegions \| None` | Optional execution partition and per-region policies inside each bounded slice. Requires a named domain and must permit every outer variant. Child callbacks receive slice-sized input geometries and inherited axes. `None` compiles one graph per slice. |

Use nested regions when consecutive computations need different chunk bounds
without a full-axis temporary between them. For example, an outer bound of
32768 can contain projection regions bounded by 8192 and another region bounded
by 32768. The intermediate storage then spans at most the outer bound. Each
inner loop handles its own tail; sizes need not divide one another. Checkpoint
ports and saved values retain their complete shapes. The whole group must pass
the outer axis-separability checks, including opaque operator rules.

```python
from shadowspill.pytorch import AxisChunks, ChunkRegions

def bounded_regions(input_axes, execution_partition, inner_policy):
    return AxisChunks(
        32768, input_axes, domain="positions",
        regions=ChunkRegions(execution_partition, inner_policy),
    )
```

An integer input axis is shorthand for `ChunkAxis(dim)`. Negative dimensions are
resolved against the input rank. `ChunkAxis(dim: int, before: int = 1,
after: int = 1)` describes an ordered dimension with length
`before * logical_extent * after`; both factors must be positive. For example,
flattening a batch and sequence into one dimension uses `ChunkAxis(0, before=B)`
when each invocation should take the same sequence slice from every batch item.
The factors specify axis coordinates, not independent microbatches.

Policies should prefer `stage.input_domains[i][domain]` when available over
inferring an axis from equal dimension lengths. For example, a saved statistic
can flatten multiple dimensions and still carry the same logical axis. Named
mappings propagate through chunked children, whole regions with supported
operator contracts, and the outer graph pair's saved values. For example, a
whole normalization can produce flattened `[batch * sequence * heads]`
statistics that a later chunked consumer must slice with its token inputs.
Executing the normalization whole does not make those statistics shared.
Gradient destinations also need proven coordinates: a parameter-gradient
feature dimension can equal the sequence length without representing tokens.
The text execution policy slices a gradient destination only when it inherits
the corresponding named domain, including activation-gradient destinations.
Named domains carry their declared `OperatorRule` contracts from forward into
backward and into nested compositions. Conflicting contracts for the same
overload and domain are refused. This propagation does not select chunking for
the whole region or change checkpoint retention. An operation with unknown
semantics does not establish an output mapping; the policy must declare it
where execution resumes.

```python
from shadowspill.pytorch import AxisChunks
from shadowspill.task.inputs import TaskInputRole

class FeedForwardChunks:
    def configure_stage(self, stage):
        if stage.terminal or not any(p.endswith(".ffn") for p in stage.module_paths):
            return None
        axes = {
            i: 1
            for i, (value, role) in enumerate(zip(stage.inputs, stage.input_roles))
            if value is not None and len(value.shape) == 3
            and role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
        }
        return AxisChunks(8192, axes, variants=("recompute",)) if axes else None

# Supply this alongside the partition policy to any public builder.
chunking = FeedForwardChunks()
```

For a provider-oriented walkthrough and qualification checklist, see
[Integrating an operator library](../advanced/operator-integration.md).

`OperatorRule(target, version, output_axes, reductions=())` certifies one opaque
operator overload's semantics. `target: str` names that exact overload;
`version: str` is a nonempty semantic version included in executable identity.
`output_axes` is a tuple of `int | ChunkAxis | AxisFrom | ChunkLocal | None`,
with one entry per flattened
output: an integer declares a dimension of exactly the logical axis extent,
`ChunkAxis` declares a factored dimension, `AxisFrom` derives its layout, and
`None` declares a shared output.
`reductions: tuple[int, ...]` instead marks listed `None` outputs as additive
chunk contributions. Shared outputs must be invariant across chunks. Reduced
outputs must sum to the full result; an average or normalized statistic does
not satisfy that requirement. Separate forward and backward overloads need
separate contracts.

`AxisFrom(input_index: int, trailing_dims: int = 0)` derives an output's ordered
chunk factor from an operator's positional input geometry. `input_index` is
nonnegative; `trailing_dims` removes that many trailing feature dimensions
before reshaping to the output. Normalization statistics flattened from
`[batch, rows, features]` use `AxisFrom(0, 1)`. This declares a shape relation;
it cannot certify a data-dependent permutation or global routing statistic.

`ChunkLocal(dim: int | None = None)` declares opaque scratch whose coordinates
belong to one child invocation, such as a dispatch permutation. An integer
`dim` names an extent proportional to the logical axis extent; `None` means fixed
geometry. These values may flow through the child, but cannot escape as saved
values or public outputs. Mixing local coordinates with ordered axis values
requires an explicit opaque restoration rule. A SAVE or RECOMPUTE cut that
retains this scratch is refused; use a retention policy that regenerates it
inside each chunk. For example, expert-sorted rows from two token chunks have two
separate routing orders; concatenating them does not produce the full-domain
expert order. Retaining them would require segmented per-chunk saved state and
its metadata, which this contract does not represent. A selective SAVE policy
can regenerate just those intermediates while retaining other backward state,
including the route choices. This contract does not make capacity-limited routing or a global
auxiliary loss independent along the selected axis. Additive sufficient statistics may escape as
reductions, with nonlinear aggregation in a separate stage.

Supported built-in rules validate axis propagation, additive parameter VJPs and
explicit backward gradient mutations. Opaque rules remain the caller's semantic
responsibility. Generic terminal training stages, RNG operations, wrapped tensor
component slicing and unsupported mutations are refused. Wrapped weights may
remain shared: capture lowers their payloads and scales to ordinary physical
inputs, and chunking preserves that mapping. This does not infer how to split
a logical quantized weight or its scales. Attention, recurrence and
convolution must retain their complete semantics outside regions chunked along dependent axes. The [architecture contract](../../architecture/axis-chunking.md)
explains retention, complete-task profiling and automatic boundary residency.


## Build and measurement policies

`PartitionPolicy` supplies custom stage boundaries to the public builders;
`PartitionSpec` is the accepted partition policy type. `measure_program` accepts
public `ProfilingOptions` for conditioning/timing duration floors, maximum
counts, wall limits and stability thresholds. The two duration shortcuts use
the same policy. These phases do not create runtime pools. See the
[profiling API](profiling.md) and [build phase](../phases.md).
