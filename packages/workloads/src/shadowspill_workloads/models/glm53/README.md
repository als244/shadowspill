# GLM-5.3

A text decoder with positional sparse MLA, ordinary residual connections and
routed/shared SwiGLU experts. It shares components with
[GLM-5.3-Flash](../glm53_flash/README.md), whose attention and residual architecture
differ.

## Shapes

[config.py](config.py) provides `Config` and the `tiny` and `published` presets.
The published shape has 78 layers, width 6,144, 64 attention heads, three initial
dense MLPs, then 256 routed experts/top-8 and one shared expert per layer.
Architecture settings can be inspected without importing Torch or MLOps.

## Implementation

[implementations/mlops/](implementations/mlops) composes the model from reusable
MLOps operations. Positional sparse attention uses 64-dimensional RoPE query/key
tails. Full indexers select causal keys in layers 0, 1, 2, 6, 10, and so on;
intervening layers reuse their indices. Packed inputs reset positions and keep
attention within sequence boundaries. Actual sequence lengths remain runtime
tensors.

Expert layers use `mlops.moe` or `expert_lora`, with the provider choosing kernels.
The shared expert factory supplies EP construction when selected during setup.

## Checkpoint

[checkpoint.py](checkpoint.py) and [weights.py](weights.py) map local Hugging Face
weights to this model. Bounded loading copies only the required parameter ranges,
including joined projections and expert banks. Native block-FP8 data and scales
retain their checkpoint bytes. The shared initializer fills executor-owned buffers
and initializes LoRA factors separately.

Use `Model.from_pretrained(...)` through the same
[client API](../../../../../../docs/python/client.md) as every other family.
The registered family name is `glm53`; see the
[built-in catalog](../../../../../../docs/python/catalog.md) for model selection.

## LoRA and precision

[lora.py](lora.py) targets the main MLA projections, dense MLP projections and
routed experts. Indexers, routers, normalization and base weights stay frozen;
shared-expert projections and the output head are optional targets.

Sparse attention requires BF16 compute. Compressed dense projections retain their
stored representation and decode for BF16 GEMMs; `expert_storage="fp8_block"`
retains block-scaled expert storage. Family capture preserves intermediate BF16
casts because changing them can alter discrete top-k attention choices.

Optimizer state, gradient storage and advanced parameter overrides use the
[common precision settings](../../../../../../docs/python/advanced/precision.md).
There is no family-specific optimizer or training loop.

## Construction

For direct phase-API use:

```python
from shadowspill_workloads.models.glm53.config import PRESETS
from shadowspill_workloads.models.glm53.implementations.mlops import Model

model = Model(PRESETS["published"])
```

[registration.py](registration.py) declares the presets and checkpoint identifiers;
[training.py](training.py) connects family settings to the shared decoder hooks.
