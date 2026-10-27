# GLM-5.3-Flash

Text architecture: 45 decoder layers (34 KDA, 11 NoPE sparse-MLA), mHC streams, three initial dense MLPs, then 288 routed experts/top-8 plus one shared expert per MoE layer. Operations live in MLOps; this folder owns model composition and Hugging Face naming/layouts.

Each MoE layer is one `mlops.moe` call, or `mlops.lora.experts.expert_lora` when
the experts carry LoRA, so the façade picks the kernels: SonicMoE on Hopper and
Blackwell, the builtin grouped GEMMs elsewhere. Routing is GLM's -- sigmoid
scores, the correction bias for selection only, normalized and scaled by 2.5
-- with the router's GEMM at mlops' precision (bf16 operands, fp32 sums and
logits). Every SwiGLU, the shared expert's too, is clamped at
`swiglu_limit` (10). The routed experts are stacked banks, `w13 [E, 2I, H]`
(gate rows, then up rows) and `w2 [E, H, I]`, held as `expert_storage` says:
`"bf16"`; `"fp8_block"`, frozen `BlockFP8Weight`s (E4M3 once with a scale per
128x128 block, the default for compressed checkpoints); or `"fp8"`, frozen
`QuackFP8Weight`s (two E4M3 orientations with per-row descales, as many bytes
as bf16). Both FP8 forms compute the experts' GEMMs in FP8 on SM90.

## Checkpoints and execution

Use the [checkpoint LoRA recipe](../../recipes/text/glm53_flash_lora/README.md)
or the [generic quickstart](../../../benchmarking/quickstart.md). Both use the
public build → profile → plan → fresh-worker execution pipeline. SSD spill is an
Executor setting; it does not require a model-specific converter or runner.

`GLMCheckpoint` reads safetensors headers, declares a meta model and implements
`copy_chunk(name, shape, offset, destination)` for bounded loading. The destination
is a flat CPU buffer in the declared dtype; offsets count physical storage elements.
Joined Q/K/V rows and expert banks read only intersecting checkpoint ranges.
Direct BF16 values and FP8 data/scales retain their checkpoint bytes.

```python
from workloads.models.glm53_flash.checkpoint import GLMCheckpoint

source = GLMCheckpoint("/models/GLM-5.3-Flash", gemm_precision="bf16")
model = source.build_model(lora_rank=32, lora_alpha=32)
# Build and profile this meta model, then save its selected Plan.
# In the fresh execution worker, the recipe fills Executor-provided buffers
# from the same checkpoint and initializes adapter factors separately.
```

The shared [initializer](../shared/glm/initialization.py) supplies those bounded
fills, including deterministic random initialization. No whole-model or expert-bank
host staging copy is required. Checkpoint and random initialization use the same
shapes and dtypes. The recipe's bounded loader supports direct checkpoint storage;
representation conversion is rejected before build. The reader's ordinary
`load_into(model)` method can explicitly convert one expert at a time for other
callers; it does not depend on ShadowSpill pools.

Dense compressed projections retain their FP8/NVFP4 bytes and scales and decode
for BF16 GEMMs. Routed experts use FP8 GEMMs with `fp8_block` storage on SM90.
Checkpoint BF16 mHC projections, routers and KDA convolution weights stay BF16;
mHC bias/scale and KDA control tensors keep checkpoint FP32. mHC projections
and coefficients compute in FP32, with BF16 residual streams. Random model
construction has separate training dtype defaults.

HF base loading and training checkpoints are separate. The recipe uses generic
bounded checkpoint helpers and records an explicit frozen-base identity for
adapter-only saves. See [Trainer and Executor](../../../docs/python/api/training.md)
for initialization, export and restore.

## KDA projection boundary

The MLOps implementation passes separate Q/K/V projections to
`mlops.causal_conv_silu_qkv`. Packed sequence resets, SiLU and checkpoint
convolution dtypes are unchanged. Frozen convolution weights use a VJP that
reads separate Q/K/V gradients and writes contiguous projection gradients;
full weight training retains the ordinary composition. Layout kernels and
provider dispatch live in MLOps. The family owns projections, Q/K normalization,
decay/beta/output gates and the attention call.

## LoRA construction

The direct importer can declare a frozen compressed base with trainable factors:

```python
model = source.build_model(
    lora_rank=32,
    lora_alpha=32,
    lora_factor_dtype=torch.bfloat16,
    lora_head=False,
)
```

The base, router, shared experts, indexer and normalization weights remain frozen.
Dense attention/MLP projections and routed experts train through LoRA; the output
head is optional. The experts' factors are stacked per layer (`lora_gate_up_a
[E, R, H]`, `lora_gate_up_b [E, 2I, R]`, `lora_down_a [E, R, I]`, `lora_down_b
[E, H, R]`) and train through `expert_lora`; their base GEMMs run at the expert
storage's precision. FP32 factor storage yields FP32 factor gradients. The recipe
initializes factors through bounded Executor buffers. Meta tensors
alone contain no initialized factor values.

Composed training and checkpoint validation remain tracked in the internal agenda.
Full-model LoRA training is not yet qualified.

## Images

The optional vision encoder follows the source checkpoint's 24-block architecture,
axial RoPE, patch merging and projector. Normalization and RoPE arithmetic use
FP32; projections use BF16. Geometry is prepared before capture; attention does
not read GPU metadata back to Python.

Programmatic callers can use their existing HF preprocessing:

```python
from workloads.models.glm53_flash import prepare_images

model = source.build_model(include_vision=True)
images = prepare_images(
    pixel_values,
    image_grid_thw,
    flattened_token_ids,
    image_token_id=source.metadata["image_token_id"],
)
# images is an ordinary pytree input: tensors plus static frame boundaries.
logits = model(flattened_token_ids, boundaries, cumulative, chunks, images=images)
```

`prepare_images` requires CPU metadata and already-expanded image placeholders.
Use the checkpoint's image processor and tokenizer; the function does not resize
images or expand tokens. When LoRA is enabled, vision linear projections also
receive factors; convolution, normalization and base weights remain frozen.

Reduced-encoder tests cover outputs, input/parameter gradients and strict export.
Previous full-checkpoint checks found low-precision fusion differences on random
pixel stress inputs; do not infer full-model numerical parity from small fixtures.
Video preprocessing and cached autoregressive generation are not integrated.

## Bounded training loss

`forward()` returns logits. For training, `loss()` delegates to MLOps'
`LanguageModelHead` or `LoRAHead` and bounds temporary logits by `chunk_size`:

```python
loss_sum = model.loss(
    tokens,
    targets,
    boundaries,
    cumulative,
    chunks,
    images=images,
    chunk_size=256,
    reduction="sum",
)
```

For microbatch accumulation, normalize the summed objective by the intended total
number of valid targets, once. The caller controls target construction, masks and
sequence packing. Output-head LoRA is enabled with `lora_head=True`.

## Partitioning

By default the builder partitions GLM as it does any model. `GLMStages`
([partition.py](../../recipes/text/glm/partition.py)) is an optional `PartitionPolicy`, kept apart
from model mathematics, checkpoint conversion and MLOps kernels; the
[generic architecture contract](../../../docs/architecture/partitioning.md)
defines the terminology. It follows exported module scopes and creates these
**forward stage occurrences**:

| Model region | Boundary rule |
|---|---|
| Input/embedding preparation | Group root-level operations until entering a layer or vision block. |
| Each decoder layer | One stage: attention, mHC/norm work and the MLP -- dense, or the MoE layer's one `mlops.moe` / `expert_lora` call with its shared expert. |
| Final norm and output head/objective | One root-level suffix stage. |
| Optional vision | Patch preparation, each of the 24 blocks, and post-block merging/projection are separated. |

Modules laid out as `...backend.experts.N` would each get a stage, with the
work after them joining the next layer's stage; GLM's stacked experts have no
such modules. Repeated layer occurrences can reuse compiled contracts; stage
count does not equal compilation count.

Pass the policy to any of the three public builders:

```python
from shadowspill import build_forward
from workloads.recipes.text.glm.partition import GLMStages

program = build_forward(
    model,
    example_args=tuple(inputs),
    partition=GLMStages(),
)
program.save("glm-forward.ssp")
```

The quickstart factory accepts `partition="glm_stages"`. A custom model can
provide a different `PartitionPolicy`; there is no architecture-name lookup in
the build API or GLM rule in the neutral planner.

## Quickstart configuration

[quickstart.py](../../recipes/text/glm/quickstart.py) supplies the model to the shared phase runner.
Both GLM families use the same runner as other models. Factory arguments choose
checkpoint, initialization (`"random"` by default or `"checkpoint"`), sequence
length, examples per step, LoRA settings and DP storage. `tiny=True` selects the
small validation fixture. Build and compute profiling need one representative GPU;
communication calibration and execution use the declared ranks.

LoRA defaults are rank/alpha 32, BF16 factors and FP32 accumulated gradients.
Quickstart's optimizer precision is configured separately; the practical training
recipe uses FP32 masters and moments. Family presets target main projections;
auxiliary attention gates, routers, shared experts and indexers remain frozen.

The original FP8 Flash model has about 303 GiB of physical parameter storage
including adapters before optimizer state and activations. Full-size pretrained
32K training, distributed evaluation and checkpoint resume still require target
H100 qualification. Small synthetic-checkpoint and saved-worker tests establish
pipeline behavior, not full-model throughput or learning results.
