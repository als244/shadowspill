# MLOps integration

`shadowspill.integrations.mlops` translates standalone `mlops.operator_metadata`
records into build options. It runs during configuration, performs no tensor
computation and does not initialize CUDA. MLOps owns the operator semantics;
this repository owns their translation and compiler integration tests.

The caller chooses checkpoint boundaries, execution regions, axes, chunk sizes,
retention and spill budgets. These helpers supply facts needed to implement
those choices safely. Other libraries use the same
[operator integration API](../python/advanced/operator-integration.md).

| Helper | Build use |
|---|---|
| `operator_chunk_rules(batch_size=1)` | Common normalization, activation, rotary, basic mHC and builtin LoRA operator rules |
| `sonicmoe_chunk_rules(compute_precision="bf16")` | SonicMoE output axes, local scratch and additive reductions |
| `mhc_chunk_rules(batch_size=1)` | Fused mHC axis/reduction rules |
| `sonicmoe_gradient_destination_rules()` | Eligible backward outputs that can write into an existing gradient |
| `fla_execution_lifetimes(rules=(), select=None)` | FLA readonly and input-layout facts attached to caller scheduling choices |
| `operator_preparations(dtype=..., cache_root=None)` | Available provider preparation factories for ordinary workload builds |
| `sonicmoe_preparation(cache_root=None)` | Explicitly selected BF16 Sonic kernel preparation |

## Attach rules to build policies

The model must already use the intended provider. These options do not change
dispatch. Add chunk rules to the `AxisChunks` values returned by the workload's
chunk policy, and pass destination/preparation options to the builder:

```python
import torch
from shadowspill.integrations.mlops import (
    operator_chunk_rules,
    operator_preparations,
    sonicmoe_chunk_rules,
    sonicmoe_gradient_destination_rules,
)

chunk_rules = (
    *operator_chunk_rules(batch_size=1),
    *sonicmoe_chunk_rules(),
)
# Use rules=chunk_rules in the AxisChunks selected by your chunk policy.
build_options = {
    "gradient_destination_rules": sonicmoe_gradient_destination_rules(),
    "operator_preparation": operator_preparations(dtype=torch.bfloat16),
}
# Pass **build_options to build_step, or use them for every search candidate.
```

Effective input positions can differ between forward and backward regions.
Choose axes from the region's `StageInfo`, as in the
[generic axis example](../python/advanced/operator-integration.md#example-affine-outputs-and-parameter-gradients).
Match `batch_size` to flattened batch/position geometry. It does not select an
axis or authorize arbitrary feature splitting.

## Chunking limits and router retention

Sonic expert-sorted activations, maps and offsets are local to one chunk. They
cannot escape through an outer result or retained cut; concatenating local
sorting maps would not reconstruct a full-domain map. A selective retention
policy can regenerate that scratch while keeping ordered expert selections.
Counts and parameter gradients are additive outputs.

The decomposed router declares `retain_for_backward` on
`torch.ops.mlops.lora_route.default`. Both graph-pair variants honor that default
for backward-required expert IDs and weights; explicit retention rules can
override it. Core has no router-name special case.

The available bundles have distinct limits:

| Bundle | Supported scope |
|---|---|
| Sonic BF16 | Fresh/accumulating up/down operations and routing metadata; no certification of EP collectives |
| Sonic `compute_precision="fp8_block"` | Frozen block-FP8 experts with BF16 activations/LoRA factors; data/scales shared, row-wise activation quantization preserved; no destination fusion, trainable base weights or `fp8_current` |
| Fused mHC | Token-axis pre/post operations; pre uses `[batch * sequence, 4, hidden]`, with 32-token/128-hidden alignment for every full/tail invocation; no stream/feature splitting |

Unsupported retained cuts or chunk geometry fail build rather than falling back
to an unmeasured execution form. Ordinary gradient additions still apply where
destination fusion is unavailable.

## Gradient destination writes

The first contribution creates its gradient buffer. Eligible later chunks,
regions and microbatches write directly into it. One backward invocation handles
all selected destinations, using one GEMM per product with the destination as
both `C` and `D`; empty experts preserve prior contents. Unconsumed outputs,
including activation and routing gradients, keep their ordinary values.

Fusion requires complete aligned contiguous FP32/BF16/FP16 destinations at the
original gradient dtype, supported literal scaling and canonical up-B ordering.
The `fuse_narrow_gradient_accumulation` permission controls narrow epilogues.
Cropped/padded ranks, intermediate narrowing into wider gradients, unknown
transformations and FP8 base containers retain ordinary accumulation. One
unsupported output does not disable other eligible outputs of the same call.

This is a destination-write capability, not qualification of every full-model
dtype. The compiler still proves ownership and preserves casts and rounding;
see [gradient destinations](../architecture/custom-gradient-destinations.md).

## FLA lifetime choices

The helper supplies readonly/layout declarations; its rules select actual
transformations. For example:

```python
from shadowspill.integrations.mlops import fla_execution_lifetimes
from shadowspill.pytorch import LifetimeRule

lifetimes = fla_execution_lifetimes(rules=(
    LifetimeRule("aten.slice.Tensor", compact=True),
    LifetimeRule("mlops.linear_attention_fla_gated_delta_rule_bwd.default",
                 barrier=True),
))
# Supply lifetimes on the outer ChunkRegions policy.
```

Only declared input positions permit relayout. L2 backward's incoming gradient
can move to a new layout; its saved normalized output and statistics retain
their original layout. The compiler still checks aliases, writes, logical values
and complete backing sets. This does not make recurrent sequence windows
independent, select additional replay or change retained values.

## Prepared kernels and saved execution

Use `operator_preparations` for ordinary BF16 builds; the provider checks device
support and optional packages when build opens its factories. Use
`sonicmoe_preparation` when explicitly requiring Sonic. Preparation is independent
of chunk size, retention and gradient destination choices.

Build packages selected BF16 Sonic/Quack kernels and configurations during
normal task finalization, excluding autotuner candidates. Provider factories
own cache formats, source checks, launchers and binary validation. ShadowSpill
stores their opaque manifests and uses the generic
[preparation lifecycle](../python/advanced/operator-preparation.md).

Saved profiling/execution loads those implementations without tuning or compiling
them. Missing or incompatible sources, schemas, specializations or binaries are
refused. Each changed composition still needs complete-task profiling and memory
admission; prepared children do not establish the parent's memory peak.

`cache_root` controls build intermediates. Its default is
`$XDG_CACHE_HOME/mlops/prepared`, or `~/.cache/mlops/prepared`; use node-local
storage in scheduled jobs. Restoring archived binaries does not need that cache.
Prepared loading may still initialize prebuilt runtime libraries, whose external
footprint remains part of admission.

The helper covers BF16 Sonic and records Hopper sparse-forward backend/LSE
selection to avoid a measured first-call probe. It does not package attention
binaries, the cuDNN backward compiler, FP8 or EP kernels. Those need their own
preparation support. Saved execution uses resolved overloads and versions;
it never reruns policy factories or mapping callbacks.

## Validation

CPU capture/metadata checks live in `tests/workloads/test_mlops_integration.py`
and the gradient-destination tests. Real destination epilogues have separate
supported-GPU coverage, including pointer reuse, empty experts, dtypes and
unconsumed outputs. Hardware-ineligible cases skip; that is not model qualification.

For prepared-kernel changes, run the maintained saved-training regression on
supported Sonic hardware with the pinned provider dependencies:

```bash
python -m pytest -m slow tests/workloads/test_prepared_sonic.py
```

Its [test guide](../../tests/workloads/prepared_sonic/README.md) defines the exact
scope. It covers cold saved phases and restore separately from distributed and
full-model qualification; no operation-level check substitutes for those runs.
