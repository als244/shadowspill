# Integrating an operator library

ShadowSpill captures a model's computation through PyTorch and compiles explicit
forward/backward tasks. Providers expose ordinary operators; the compiler owns execution-region
chunk loops. Ordinary supported PyTorch operations use
the frontend's built-in contracts. Opaque operators need explicit additional
contracts for transformations whose semantics the frontend cannot inspect.

This guide separates basic integration from optional optimizations. See the
[frontend reference](../api/frontend.md#axis-chunking-policies) for exact chunk-policy
types and the [axis-chunking architecture](../../architecture/axis-chunking.md) for
transformation and storage details.

## What an operator must provide

| Capability | Provider responsibility | When needed |
| --- | --- | --- |
| Capture and lowering | A graph-visible, registered operator or a supported PyTorch decomposition; accurate schema, fake/meta implementation, output geometry, dtype, device, alias and mutation behavior | Every operation in a captured task |
| Differentiation | A capture-compatible derivative for each differentiable input; opaque backward calls must themselves support fake execution and lowering | Training through that operation |
| Cost-based retention | A PyTorch flop-counter formula for the custom operation | Optional classification under the cost retention policy; an unpriced opaque operation is conservatively treated as expensive |
| Axis chunking | Versioned `OperatorRule` declarations for every opaque overload encountered in the selected region, including backward and accumulating forms | Only when splitting a region containing that operation |
| Saved branch decisions | A saved-state default or explicit retention rule for backward-required discrete choices | Routing or other choices that backward must reuse |
| Moving or compacting operations | Separate readonly/effect and input-layout contracts | Optional execution-lifetime transformations |
| Fused gradient accumulation | A mutable accumulating overload and `GradientDestinationRule` | Optional direct accumulation into an existing gradient |
| Saved external kernels | An `OperatorPreparation` factory that captures, validates and loads selected implementations | External JIT/tuning providers that need prepared binaries in saved artifacts |
| Non-tensor resources | An importable resource factory and explicit lifecycle | Operators requiring shared handles or buffers outside ordinary tensor state |

Fake execution describes geometry; it does not prove independence, absence of
effects, or safe alias ownership. Declare aliasing and mutation accurately in the
dispatcher contract. Operators must use the runtime's device and stream and
expose allocations to its supported allocator/accounting path. Hidden device
allocations or process-global state do not become safe merely by supplying a
shape rule. Qualify the complete task under physical admission.

An operation can work in an unchunked task without an axis rule. Unknown operations
inside a selected chunked region are refused; ShadowSpill does not silently run
the requested region unchunked. A rule cannot split work hidden inside an opaque
call. To chunk projections around a whole-sequence attention operation, expose
them as distinct graph operations and choose execution regions accordingly.

The flop formula helps choose a saved-state cut; it is not a task timing or
workspace measurement. Explicit retention rules can override that cost choice.
See [graph-pair construction](../../architecture/graph-pair-construction.md) for
the cost policy and its interaction with saved-state declarations.

## Define the independent domain

The workload's `ChunkPolicy.configure_stage(stage)` chooses the region, its input
axes and chunk size by returning `AxisChunks`. Providers supply semantic facts;
workloads choose where to use them. Keep reusable provider facts independent of
any execution engine. Put the adapter that constructs ShadowSpill rules in this
repository or the caller's integration package. For example,
[`shadowspill.integrations.mlops`](../../integrations/mlops.md) converts standalone `mlops.operator_metadata`
records without duplicating their operator semantics. The engine does not
discover libraries or select policies by architecture name.

For each selected input, `AxisChunks.inputs` identifies an **effective region
input position** and its logical axis. Unlisted inputs are shared. Selected axes
must have the same logical extent. A dimension of size `B * S` can represent the
sequence domain using `ChunkAxis(0, before=B)`; each slice then takes the same
sequence interval from every batch item. A dimension of size `S * H` uses
`ChunkAxis(dim, after=H)`. These factors preserve coordinate order through
flattening; they do not create additional microbatches.

Use a named `domain` and inherited `stage.input_domains` where possible. Equal
dimension lengths alone do not establish a shared domain: a weight-gradient
dimension can happen to equal a sequence length without representing tokens.
Axis selection is not restricted to token positions, but the operator must be
independent along the particular axis selected.

## Declare each opaque output

`OperatorRule(target, version, output_axes, reductions=())` names one exact
dispatcher overload, for example `acme.affine_fwd.default`. The nonempty string
`version` identifies its semantic contract. Provide one entry per flattened
output, in captured output order.

| Output declaration | Promise to the compiler | Assembly across chunks |
| --- | --- | --- |
| `int` or `ChunkAxis` | Ordered coordinates along the specified dimension, optionally factored | Write into the corresponding full output slice |
| `AxisFrom(input_index, trailing_dims=0)` | Ordered coordinates derived from an operator input's axis mapping and geometry | Slice assembly using the derived mapping |
| `None`, absent from `reductions` | The same shared result for every chunk | Produce the shared result once |
| `None`, listed in `reductions` | Each chunk computes an additive contribution to the complete result | Sum contributions |
| `ChunkLocal(dim)` | Scratch with coordinates private to one chunk; the indicated extent scales with that chunk | Consume locally; never concatenate into an outer boundary |

`AxisFrom.input_index` indexes the **operator's positional inputs**, not the
enclosing region's inputs. For normalization of `[B, S, D]`, a flattened inverse
norm with shape `[B * S]` can use `AxisFrom(0, 1)`: remove one trailing feature
dimension before deriving the output layout. `ChunkLocal(None)` describes
scratch with fixed geometry.

A sum is additive; a mean, softmax over the split axis, normalized routing
statistic, or maximum is not certified by this sum contract. Expose additive
sufficient statistics and perform nonlinear aggregation outside the split
region when the algorithm permits it. Batch-dependent capacities or global
expert ordering must not be declared independent simply because output shapes
match.

### Example: affine outputs and parameter gradients

Suppose a provider exposes `affine_fwd(x, weight, bias) -> y`, with
`y = x * weight + bias`, and
`affine_bwd(x, weight, dy) -> (dx, dweight, dbias)`. The last dimension contains
features and the other dimensions contain independent positions. Weight and
bias are shared; their backward outputs sum over those positions.

The caller-side integration can construct these rules without creating kernels
or tensors. The operation library itself need not import ShadowSpill:

```python
def affine_chunk_rules():
    from shadowspill.pytorch import AxisFrom, OperatorRule

    return (
        OperatorRule(
            "acme.affine_fwd.default", "1", (AxisFrom(0),),
        ),
        OperatorRule(
            "acme.affine_bwd.default", "1",
            (AxisFrom(0), None, None), reductions=(1, 2),
        ),
    )
```

A workload can attach those rules to a region containing that affine operation.
For this example only, region inputs zero and one are ordered `[B, S, D]`
activation tensors; any remaining inputs are shared parameters:

```python
from shadowspill.pytorch import AxisChunks

def affine_region():
    return AxisChunks(
        chunk_size=8192,
        inputs={0: 1, 1: 1},
        domain="positions",
        rules=affine_chunk_rules(),
    )
```

Actual forward and backward regions often have different input positions. Select
their mappings from `StageInfo` rather than assuming this example's positions.
Separate backward and accumulating overloads need their own rules. Shape checks
can catch a bad mapping; they cannot prove the provider computed the right sum.
Changing reduction order can change floating-point rounding even with a correct
contract.

## Scratch, retention and gradient destinations are separate

Expert-dispatched rows, sort indices and local offsets are common `ChunkLocal`
values. An explicit operator rule must restore ordered coordinates before an
output escapes the chunk. Saving or returning these local values across the
outer boundary is rejected: concatenating two independently sorted chunks does
not yield the full-domain sort order.

Use the graph pair's retention policies to regenerate such scratch inside each
chunk while retaining any required ordered routing choices. This applies to
SAVE as well as RECOMPUTE. Chunking does not shrink the full saved cut or change
checkpoint boundaries. See [routing retention](../../architecture/routing-retention.md)
and [graph-pair policies](../api/frontend.md#graph-pair-policies).

An additive output declaration permits summing contributions; it does not prove
the provider can write directly into an existing gradient. Ordinary accumulation
may still allocate a contribution and add it in place. To fuse its production,
provide the separate [gradient-destination rules](../../architecture/custom-gradient-destinations.md).
The first contribution creates the buffer; subsequent chunks, regions and
microbatches can accumulate into it. Preserve the declared precision and
mutation/alias behavior of the replacement overload.

Likewise, an axis rule does not authorize scheduling an opaque operation earlier,
replaying it an extra time, or compacting its inputs. Those permissions belong to
the separate [execution-lifetime controls](../api/frontend.md#execution-lifetimes).

## How the declarations enter build, profile, plan and execute

1. **Build:** capture full checkpoint graphs and their SAVE/RECOMPUTE derivatives.
   Resolve retention and execution policies, propagate domains, then transform
   selected regions into bounded loops. Compile the required full/tail and
   creator/accumulator forms. Providers still implement ordinary operators;
   ShadowSpill owns this composition.
2. **Identity and reuse:** resolved axes, rule versions, chunk sizes, geometry,
   aliases and child contracts participate in executable identity. Structurally
   equivalent forms can share compilation. Change contract versions when their
   meaning changes; do not use versions to hide an incompatible implementation.
3. **Profile:** measure the complete composed task, including input copies,
   assembly, loop overhead and internal provider allocations. A child-kernel
   timing or a rule declaration cannot certify the parent memory peak. Reuse
   measurements only through the normal compatibility checks.
4. **Plan:** compare measured candidates and graph-pair variants, select residency
   and spill, and admit physical memory. Chunk loops are internal execution,
   not independently scheduled or spillable tasks. Full boundaries and saved
   tensors remain normal planner objects.
5. **Execute:** restore the saved composition and required operator registrations
   or prepared implementations. Policy callbacks are not rerun. Runtime accounting
   applies to the complete task; anonymous workspace inside an opaque operation
   cannot be offloaded between that operation's internal kernels.

External compiler caches, source hashes and autotuning behavior belong to the
provider. See [prepared operators](operator-preparation.md) for packaging
selected implementations and [resources](../../architecture/resources.md) for
non-tensor resource lifecycles. Neither interface substitutes for an axis contract.

## Qualifying a new integration

Compare eager and compiled forward values, input VJPs and parameter updates for
the declared layouts/dtypes. Exercise whole, multiple-chunk and uneven-tail
geometry, more than one batch item, and both fresh and continuing gradient
accumulation. Test SAVE and RECOMPUTE cuts, including retained routing choices
and refusal when local scratch would escape. Check actual memory and timing
for the complete tasks, then load a saved Plan in a fresh process without
relying on warm compiler caches. Qualify distributed behavior when the provider
uses collectives or distributed resources.

Use tolerances appropriate to the arithmetic; bitwise equality across chunk
geometries is not the default contract. Unsupported axes, aliasing, effects or
nonadditive reductions should fail explicitly, rather than produce an apparently
valid Plan. The maintained [axis tests](../../../packages/engine/tests/engine_tests/pytorch/chunking)
and [saved provider regression](../../../packages/workloads/tests/workload_tests/prepared_sonic/README.md)
illustrate these checks; the latter is one implementation, not a required library.
