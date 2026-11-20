# Retaining forward routing choices

Both graph-pair variants retain backward-required outputs of `aten.topk.default`
and operators whose providers declare the saved-state default described below.
A decomposed router can use this to preserve its selected expert IDs and
routing weights. The default applies even when a
variant uses `RetentionPolicy(default="recompute")`, and when its explicit
rules select only dense-attention retention.

Top-k is a discrete choice. Small numerical differences in regenerated logits
can choose different experts near a tie. Backward must use the branch taken by
forward. Saving the selected IDs and weights avoids rerunning this decision;
the expensive projections and expert computations can still be recomputed.
Other backward-required state remains subject to the selected policy. For
example, a routing derivative may also need regenerated logits for its softmax
or sigmoid normalization. Retaining choices does not freeze all intermediate
values or make a whole training step bitwise deterministic.

These defaults depend on operator behavior, independently of model architecture.
They cover architectures using these graph-visible operators. A provider that
hides routing inside an opaque whole-MoE operation needs to expose a separate
routing operation whose outputs expose that saved state. ShadowSpill cannot
infer internal routing tensors from that opaque call. An additional visible
provider can declare its default or use an explicit `RetentionRule`.

## Provider-owned default

A provider may set the static boolean attribute
`retain_for_backward = True` on its public `torch.ops` overload
after registering the operator:

```python
torch.ops.my_provider.route.default.retain_for_backward = True
```

No ShadowSpill import, provider discovery or engine-side operation list is
needed. Absence or `False` leaves the ordinary retention policy in control;
non-boolean declarations are rejected. Set the declaration when registering
the operator, before capture, and keep it stable during build. The annotation
retains only backward-required outputs. It does not certify effects, layouts,
axis independence, or the correctness of an opaque operator's backward.
It cannot reveal choices hidden inside a whole-MoE kernel.

The resolved declaration, `aten.topk.default` included, enters capture
identity, so a cut is never served from a graph-pair entry built under a
different declaration. Saved execution does not query the declaration again.
The provider owns its declaration; core does not recognize provider namespaces
or router names. Use matching provider sources when rebuilding workloads.

For each operation, a model checkpoint annotation decides first (an explicit
rule that contradicts it is refused), then an explicit rule, then this default,
then the variant's `default`, then cost classification. Every cut but a plain
recompute from the checkpoint inputs retains the results of operations that use
random state or ordered effects; a rule asking to regenerate one fails capture.

## Advanced overrides

An explicit rule takes precedence over the default, separately for each
variant. For example, this deliberately permits routing replay only in the
recompute variant:

```python
from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule

policy = GraphPairPolicy(
    recompute=RetentionPolicy(
        default="recompute",
        rules=(RetentionRule("my_provider.route.default", "recompute"),),
    ),
)
```

Such an override accepts possible branch changes when replayed inputs differ.
There is no additional policy candidate or architecture-specific chunk setting
for routing retention.

## Memory and the offline pipeline

Saved routing tensors are ordinary saved activations: one ID and one weight
per selected assignment. PressureFit chooses their residency and spill
placement; they are not forced to stay on the device. Chunking slices the saved
tensors using their declared output axes; expert-local permutations and offsets
can be regenerated from the saved IDs. The final graphs expose their saved
tensors normally to compilation, profiling, planning, physical admission and
serialization.

---

Previous: [Custom gradient destinations](custom-gradient-destinations.md) ·
[Architecture index](README.md) ·
Next: [Model tensors and initialization](state-import.md)
