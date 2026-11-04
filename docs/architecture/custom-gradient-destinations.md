# Custom gradient destinations

`capture.gradients.accumulate_gradient_outputs` can replace an opaque backward
operator with a versioned accumulating overload. This is a low-level capture
interface. Public training builds, execution-region composition and chunking
do not yet propagate these rules. The existing ATen multiply fusion and ordinary
in-place additions remain the default when no rule is supplied.

## Operator contract

One rule names a fresh-output dispatcher overload, its accumulating replacement,
the eligible output indices, and a positive contract version. For example:

```python
from shadowspill.pytorch.capture.gradients import (
    GradientDestinationRule,
    accumulate_gradient_outputs,
)

rule = GradientDestinationRule(
    target="provider.backward.default",
    replacement="provider.backward_accumulating.default",
    output_indices=(1, 2),
    version=1,
)
accumulating = accumulate_gradient_outputs(
    backward,
    parameter_gradient_leaves,
    gradient_destination_rules=(rule,),
)
```

The replacement receives the original arguments and three additional arguments:
`destinations: Tensor[]`, `output_indices: int[]`, and `contract_version: int`.
Its dispatcher schema must mark `destinations` mutable, for example
`Tensor(a!)[] destinations`, and permit mutation of no other argument. With
`torch.library.custom_op`, Tensor-list arguments must be positional-capable in
the schema even though the generated call supplies these arguments by keyword.

The operator adds each selected contribution directly to its supplied buffer.
It runs once for all selected outputs, preserves the other outputs, and returns
`None` or the corresponding destination in consumed positions. `custom_op`
implementations should return `None`, because they cannot return input aliases.
The graph returns the prior buffers in place of the consumed gradients.
The fresh form still creates gradients; this transformation only constructs the
form that receives existing gradients. It does not allocate zero-filled runtime
buffers or repeat the backward for each parameter.

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

`fuse_narrow_gradient_accumulation` retains its existing meaning. The default
mapper declines narrow destinations when this permission is false. A custom
mapper must honor the same setting, either preserving the original rounding or
declining fusion. An explicit mapping must be writable without overlapping
elements; definite overlap is rejected. Layouts whose overlap cannot be decided
statically remain the provider's responsibility.

## Ownership and artifact identity

The pass only consumes single-use gradient expressions. Destinations must exist
before the producer and must not share storage with another supplied destination
or an input the producer reads. An externally observed gradient remains intact.
These conservative checks protect other graph consumers and preserve the
fallback path.

Rules and callbacks are build inputs; executables consult no mutable registry.
The replacement call, arguments and version enter the graph's compatibility
digest. Change the version when the mapping contract changes. Mutation records
and aliases pass through normal `GraphArtifact` capture and cached restoration.
Actual kernel workspace still needs normal compilation, profiling and physical
admission: a mutation schema does not prove that a provider avoids an internal
temporary. CPU tests establish grouped rewriting, nonzero LoRA VJPs, full/tail
accumulation, ownership and artifact round trips. GPU provider kernels and the
remaining public build integration require separate qualification.
