# Custom gradient destinations

`build_step` and `build_forward_backward` accept `gradient_destination_rules`:
rules that replace an opaque backward operator with a versioned
accumulating overload. The rules apply to shared parameters across checkpoint
regions, execution regions, full/tail chunks and subsequent microbatches. The
first contribution creates the gradient; later contributions write into its
canonical buffer. One replacement call handles all eligible outputs together.

The empty tuple is the default. The built-in matrix-multiply fusion and
ordinary in-place additions apply when no custom rule matches. A rule does not
change checkpoint retention or select chunk axes, region boundaries or chunk
sizes; those remain independent build controls.

## Accumulating overload requirements

One rule names a fresh-output dispatcher overload, its accumulating replacement,
the eligible output indices, and a positive version number. For example:

```python
from shadowspill import build_step
from shadowspill.pytorch import GradientDestinationRule

rule = GradientDestinationRule(
    target="provider.backward.default",
    replacement="provider.backward_accumulating.default",
    output_indices=(1, 2),
    version=1,
)
program = build_step(
    model,
    objective=objective,
    optimizer=optimizer,
    example_args=example_args,
    gradient_destination_rules=(rule,),
)
```

The replacement receives the original arguments and three additional arguments:
`destinations: Tensor[]`, `output_indices: int[]`, and `contract_version: int`.
Its dispatcher schema must mark `destinations` mutable, for example
`Tensor(a!)[] destinations`, and permit mutation of no other argument. The
destination-list argument must be positional-capable in the schema. The rewrite
normalizes positional-capable arguments into schema order, so later axis
rules can interpret the same call.

The operator adds each selected contribution directly to its supplied buffer.
It runs once for all selected outputs, preserves the other outputs, and returns
`None`, a zero-element tensor, or the corresponding destination in consumed
positions. `custom_op` implementations cannot return input aliases. Prefer a
fixed `Tensor` return schema with zero-element tensors for consumed positions:
PyTorch's automatic functionalization rejects mixed optional-Tensor returns on
mutable custom operators. A zero-element placeholder allocates no
tensor storage and its unused extraction is removed. A provider with its own
functionalization may support other permitted forms.
The graph returns the prior buffers in place of the consumed gradients.
The fresh form still creates gradients. The rewrite constructs the form that
receives existing gradients and routes later shared-parameter uses to that same
buffer. It does not allocate zero-filled runtime buffers or repeat the backward
for each parameter. `capture.gradients.accumulate_gradient_outputs` also accepts
the rules directly for callers constructing low-level graph artifacts.

## Layout, scaling and precision

Without a mapper, reversible views at the same dtype can map a destination back
to the producer's output geometry. A reshape that copies cannot be a writable
destination view. Dtype changes, scaling, slicing and non-view permutations
require explicit provider support; unsupported outputs keep their ordinary
accumulation path. Other outputs from the same producer can still fuse.

A rule's optional build-time mapper receives a `GradientDestinationContext`
containing the producer and all eligible destinations. Each destination includes
the operation chain from the final gradient back to the raw output. The mapper
returns `GradientDestinationMapping`: selected output indices, sequences of
storage-preserving `DestinationView` operations, and provider-specific keyword
arguments. Those keywords become real operator arguments. The mapper must
account for every transformation, including scale, dtype rounding, padding and
index order. It cannot silently reinterpret a raw gradient as the final
parameter gradient.

`fuse_narrow_gradient_accumulation` permits accumulating into a destination
narrower than its computation dtype, such as BF16. The default mapper declines
such destinations when it is false. A custom mapper must honor the same
setting, either preserving the original rounding or declining fusion. An explicit mapping must be writable without overlapping
elements; definite overlap is rejected. Layouts whose overlap cannot be decided
statically remain the provider's responsibility.

## Ownership and artifact identity

The pass only consumes single-use gradient expressions. Destinations must exist
before the producer and must not share storage with a different supplied
destination or an input the producer reads. Successive producers may update the
same canonical destination, in graph order; two outputs of one producer cannot
write into that same destination. An externally observed gradient or partial sum
remains intact. Sliced activation cotangents keep their own chunk coordinates.
These conservative checks protect other graph consumers and preserve the
fallback path.

Rules and callbacks are build inputs; executables consult no mutable registry.
Builders snapshot and validate the rules before accelerator preparation. A
`GraphPairStore` owns one fixed rule set: its persistent base AOT pairs are
reusable across rule sets, while each store derives its accumulating forms with
its own rules. Mappers must be deterministic for their context; do not mutate
callback state during a build.
The replacement call, arguments and version enter the graph's compatibility
digest. Change the version when the supported mapping changes. Mutation records
and aliases pass through normal `GraphArtifact` capture and cached restoration.
Actual kernel workspace still needs normal compilation, profiling and physical
admission: a mutation schema does not prove that a provider avoids an internal
temporary. Chunked providers also need `OperatorRule` axis/reduction descriptions
for both their fresh and accumulating overloads. Final replacement graphs enter
ordinary structural compilation and profiling; saved Programs and Plans carry
the operator registration dependency and implementation digest. Rule callbacks
are not needed by the saved executor. Each real provider's fused kernels and
distributed behavior require qualification in addition to generic rewrite tests.

## Integrating an operator library

The operation library owns its accumulating overload, supported layouts/dtypes
and any eligibility checks. A caller-side adapter constructs
`GradientDestinationRule` objects from those facts. It may also supply
`OperatorRule` output axes and reductions for the fresh and accumulating
overloads when the caller selects chunked execution.

The compiler checks destination ownership, rewrites the graph and profiles the
complete task. It does not select rules by library or model name, infer an
opaque kernel's internal allocation behavior, or change dispatch. Keep examples
for individual libraries in their integration guides. The generic
[operator integration guide](../python/advanced/operator-integration.md) explains the
required declarations and validation.

---

Previous: [Axis chunking within a stage](axis-chunking.md) ·
[Architecture index](README.md) ·
Next: [Retaining forward routing choices](routing-retention.md)
