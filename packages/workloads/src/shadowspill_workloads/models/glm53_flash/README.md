# GLM-5.3-Flash

A hybrid text decoder with Kimi Delta Attention (KDA), NoPE sparse MLA, manifold
hyper-connections (mHC), and routed/shared SwiGLU experts. It shares components
with [GLM-5.3](../glm53/README.md).

## Shapes

[config.py](config.py) provides `Config` and the `tiny` and `published` presets.
The published shape has 45 layers (34 KDA and 11 sparse-MLA), width 4,096, three
initial dense MLPs, then 288 routed experts/top-8 and one shared expert per layer.
Architecture settings can be inspected without importing Torch or MLOps.

## Implementation

[implementations/mlops/](implementations/mlops) owns model composition; MLOps owns
attention, convolution, routing and expert kernels. Routing uses sigmoid scores,
a correction bias for selection, normalized top-k weights and a scale of 2.5.
SwiGLU activations are clamped at the configured limit (10 by default), including
shared experts and LoRA paths.

Packed sequence boundaries remain runtime tensors. KDA supplies separate Q/K/V
projections to the fused causal convolution; sequence resets and convolution
dtypes are preserved. The shared expert factory supplies EP construction when
selected during setup.

## Checkpoint

[checkpoint.py](checkpoint.py) and [weights.py](weights.py) map local Hugging Face
weights to this model. Bounded loading reads only the requested parameter ranges;
joined projections and expert banks need no whole-bank staging copy. Native
block-FP8 data and scales retain their checkpoint bytes. LoRA factors initialize
separately in executor-owned buffers.

Use `Model.from_pretrained(...)` through the same
[client API](../../../../../../docs/python/client.md) as every other family.
The registered family name is `glm53_flash`; see the
[built-in catalog](../../../../../../docs/python/catalog.md) for model selection.

## LoRA and precision

[lora.py](lora.py) targets main KDA/MLA projections, dense MLP projections and
routed experts. KDA control gates, indexers, routers, mHC, convolution,
normalization and base weights stay frozen. Shared-expert projections and the
output head are optional targets.

Sparse attention requires BF16 compute. Compressed dense projections keep their
stored bytes/scales and decode for BF16 GEMMs. Expert storage selects:

| Setting | Frozen expert representation |
| --- | --- |
| `bf16` | BF16 banks |
| `fp8_block` | One E4M3 orientation with a scale per 128×128 block |
| `fp8` | Quack E4M3 orientations with per-row descales |

Checkpoint mHC projections, routers and KDA convolutions retain their stored
dtypes. mHC coefficients and KDA control parameters retain checkpoint FP32;
mHC arithmetic uses FP32 with BF16 residual streams. Optimizer state, gradients
and advanced overrides use the
[common precision settings](../../../../../../docs/python/advanced/precision.md).

## Construction

For direct phase-API use:

```python
from shadowspill_workloads.models.glm53_flash.config import PRESETS
from shadowspill_workloads.models.glm53_flash.implementations.mlops import Model

model = Model(PRESETS["published"])
```

[registration.py](registration.py) declares the presets and checkpoint identifiers;
[training.py](training.py) connects family settings to the shared decoder hooks.

## Vision component

The direct model implementation also contains a vision encoder with axial RoPE,
patch merging and a projector. `GLMCheckpoint.build_model(include_vision=True)`
includes it. `prepare_images` accepts preprocessed pixels, CPU geometry metadata
and expanded image-placeholder tokens; it does not resize or tokenize images.
The public `glm53_flash` model declares text inputs. An image/text model
needs its own image input schema and preparation declarations.
