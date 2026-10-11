# GLM-5.3-Flash

Text architecture: 45 decoder layers (34 KDA, 11 NoPE sparse-MLA), mHC streams, three initial dense MLPs, then 288 routed experts/top-8 plus one shared expert per MoE layer. Operations live in MLOps; this folder owns model composition and Hugging Face naming/layouts.

## Direct checkpoint import

No permanent converted checkpoint is required. The importer:

1. Reads safetensors headers and quantization metadata.
2. Declares the model on `meta`, validates every required tensor, and records physical storage sizes.
3. Fills already-allocated destination tensors in place, with bounded per-tensor scratch.
4. Preserves FP8/NVFP4 bytes and scales, joining gate/up rows without requantization.
5. Releases consumed mapped source pages. No whole-model host allocation is needed.

Example:

```python
from workloads.mlops.glm53_flash.checkpoint import GLMCheckpoint
from shadowspill.pytorch import import_model_state

with GLMCheckpoint(checkpoint_directory, gemm_precision="bf16") as source:
    model = source.build_model()
    model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        initialize=source.load_into,
    )
```

The initializer also works with ordinary allocated CPU tensors. It has no runtime or pool dependency. A temporary SSD spill pool still contains an execution copy of the imported state; directly using existing safetensors extents as a read-only pool is a separate, unimplemented storage capability.

The original Z.ai release uses block FP8. The RedHatAI release mixes 1x16 NVFP4 routed experts with BF16 dense projections and FP8 auxiliary MTP weights. The importer resolves formats per projection, preserves activation global-scale metadata, and excludes MTP. Vision is optional; text-only construction leaves its tensors unread.

## CLI

Metadata inspection does not load tensor payloads:

```bash
python -m workloads.mlops.glm53_flash.forward \
  --checkpoint /local/models/GLM-5.3-Flash-NVFP4 \
  --outdir /local/results/glm-nvfp4 --inspect
```

Forward validation:

```bash
python -m workloads.mlops.glm53_flash.forward \
  --checkpoint /local/models/GLM-5.3-Flash-NVFP4 \
  --outdir /local/results/glm-nvfp4 \
  --ssd-directory /local/spill --execution-gib 24 --passes 3 \
  --minimum-evict-kib 128 \
  --text "The capital of France is"
```

`--spill-gib` overrides the default of the planner's minimum reservation (including its admission margin), rounded up to GiB plus 4 GiB. `--layer-limit N` selects a decoder prefix for debugging and is explicitly recorded in the inventory. Only BF16 GEMMs are currently validated for the composed checkpoint path; unsupported recipes fail rather than silently selecting another one.

The `--minimum-evict-kib` option configures the ordinary planner's small-object
policy (default 1024 KiB; zero allows every object to be evicted). Objects below
it are retained between uses and currently receive separate reserved physical
slots. For very many small quantization scales, a lower value such as 128 KiB
can recover substantial execution-pool capacity, even in a forward-only plan.

The CLI saves the exact prompt/token IDs, options, metadata, build/plan artifacts,
per-call times/top tokens/log probabilities and final logits. Repeatable finite logits are a smoke check, not a substitute for reference-model numerical parity.

## Development status

Both complete checkpoints run through SSD-backed planned execution with BF16
GEMMs. The tested text prompt and the processed red-image prompt each produce
three bit-identical repeated forwards and the same final answer as the streamed
HF-definition reference.

| Stored weights | Text-only forward | Image/text forward |
|---|---:|---:|
| NVFP4 / BF16 mix | about 27.4 s | about 27.4 s |
| Block FP8 / BF16 mix | about 45.6 s | about 46.2 s |

These short prompts use the fixed expert schedule intended for training.
Even an empty expert declares its weights as task inputs, so a forward fetches
essentially the complete compressed model (190 GB or 321 GB with vision).
These timings do not represent active-expert-only inference.

A matching answer is not full numerical parity. On the image prompt, text-token
logit relative L2 versus the independent reference is 11.7% for NVFP4 and 12.7%
for FP8; image-token positions differ substantially more. The reference uses
the independently evaluated FP32 KDA recurrence. It manually decodes the same
checkpoint weights for BF16 GEMMs and bypasses the Transformers quantized loader.
Thus these are BF16 implementation comparisons, not comparisons with a stock
checkpoint-configured FP8/NVFP4 execution. Component, accumulated-error, and
ordinary torch.compile controls are recorded in
docs/internal/plans/glm_checkpoint_1010/VALIDATION.md.

The supported composed checkpoint recipe remains original compressed storage
plus BF16 GEMMs. Checkpoint-specific low-precision GEMMs and full-size LoRA
training remain unqualified.

## Training checkpoints

HF import and training checkpoint saving are separate operations. Import already
streams the compressed base into the temporary pool. The generic checkpoint writer
can stream pool objects to disk without a whole-model host snapshot. Its default
includes the full model state; an explicit frozen-base identity selects a smaller
checkpoint.

For a checkpoint that omits the frozen base, the planned step supports an
explicit base identity:

```python
base_id = "immutable-source-revision/model-and-precision-config"
step.save("lora.pt", frozen_state_id=base_id)
# Initialize the same compressed base before restoring:
step.load_state_dict(
    torch.load("lora.pt", mmap=True, weights_only=True),
    frozen_state_id=base_id,
)
```

This stores trainable weights, persistent buffers, state mutated by the model,
optimizer state and step count. Unchanged frozen parameter payloads are neither
read nor written. The identity is supplied by the caller; ShadowSpill verifies
the string on restore but does not hash the base checkpoint. Include the immutable
source revision and relevant model/precision configuration. The default still
saves the full model. Master/compute weight selection remains independent.
Training-loop RNG, data progress and schedule state remain the loop's/trainer's
responsibility.

The reduced composed GLM passes three SSD-planned LoRA updates and exact
full-checkpoint replay with BF16, FP8 and NVFP4 expert storage, all using BF16
GEMMs. Selected-state checkpoints now pass the same three precision cases and
exact next-step replay. Their size is approximately 1.76 MB, versus 3.77–4.60 MB
for complete tiny-model checkpoints. Real four-layer checkpoint prefixes execute three LoRA updates and restore the
complete next update exactly. Their selected checkpoints are about 443 MB.
They remain numerical diagnostics: aggregate first-step gradient relative L2
versus eager is 1.15% for NVFP4 and 1.42% for FP8 with the default compiler
behavior, or 0.58% and 1.11% with explicit precision-cast emulation. An all-save
FP8 control retains the same 1.11% discrepancy; recomputation does not explain it.
Near-zero gradients and discrete routing make relative factor error after Adam
larger. This is not a claim of full-size model training or exact reference parity.

The reduced packed image/text model also passes three LoRA updates with an
optional LoRA output head when eager and planned execution use the same bounded
loss. Its maximum per-factor difference is 2.96% under the unchanged 3% test
limit, and the selected checkpoint replays the next step exactly. This integration
check complements the independent operation-level gradient tests.

## Multiple prompts with one plan

Repeat `--text` to reuse one model import and plan:

```bash
python -m workloads.mlops.glm53_flash.forward \
  --checkpoint /local/models/GLM-5.3-Flash-NVFP4 \
  --outdir /local/results/glm-prompts --minimum-evict-kib 128 --passes 2 \
  --text "The capital of France is" --text "The opposite of hot is"
```

The plan uses the longest prompt's geometry. Shorter prompts are right-padded;
returned/logged logits include only real tokens. Each prompt must repeat exactly
across passes. Files `logits-0.pt`, `logits-1.pt`, etc. match argument order.

## LoRA construction

The direct importer can declare a frozen compressed base with trainable factors:

```python
model = source.build_model(
    lora_rank=32,
    lora_alpha=32,
    lora_factor_dtype=torch.float32,
    lora_head=False,
)
```

The base, router, shared experts, indexer and normalization weights remain frozen.
Dense attention/MLP projections and routed experts train through LoRA; the output
head is optional. BF16 activations/GEMMs with FP32 factor storage yield FP32 factor
gradients. Import initializes supplied pool state one projection/factor at a time.
Meta tensors alone contain no initialized factor values.

Composed training and checkpoint validation remain tracked in the internal agenda.
Full-model LoRA training is not yet qualified.

## Images

The optional vision encoder follows the source checkpoint's 24-block architecture,
axial RoPE, patch merging and projector. Normalization and RoPE arithmetic use
FP32; projections use BF16. Geometry is prepared before capture; attention does
not read GPU metadata back to Python.

```bash
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 \
python -m workloads.mlops.glm53_flash.forward \
  --checkpoint /local/models/GLM-5.3-Flash-NVFP4 \
  --image /local/examples/image.png --text "Describe this image." \
  --outdir /local/results/glm-image --minimum-evict-kib 128
```

This CLI accepts one still image and one prompt, using the source HF PIL image
processor and chat template. `--assistant-prefix` optionally supplies text after
the generation prompt. It does not install or require a video-processing backend.
Video preprocessing and cached autoregressive generation are not integrated.

Programmatic callers can use their existing HF preprocessing:

```python
from workloads.mlops.glm53_flash import prepare_images

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

CPU/GPU output, input-gradient, parameter-gradient and strict-Export comparisons
against HF pass on reduced encoders. The complete checkpoint vision encoder is
bit-exact in eager execution on the tested inputs. The processed-image SSD check
also matches exactly with the compiler setting above.

That setting preserves explicit low-precision rounding during Inductor fusion;
it does not change any global ShadowSpill default. A random-pixel stress input
still shows roughly 8% output-relative-L2 drift under both ordinary torch.compile
and ShadowSpill, even with the setting. Keep that limitation separate from the
processed-image result. Complete image/text smoke checks pass for both
checkpoints; the remaining numerical differences are described above.

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

GLM uses the public `PartitionPolicy` protocol. Its implementation is isolated
in [partition.py](partition.py), separate from model mathematics, checkpoint
conversion, and MLOps kernels. The [generic architecture contract](../../../docs/architecture/partitioning.md)
defines the same terminology and requirements for every model.

The policy follows exported module scopes and creates these **forward stage
occurrences**:

| Model region | Boundary rule |
|---|---|
| Input/embedding preparation | Group root-level operations until entering a layer or vision block. |
| Layers 0–2 | One stage per complete KDA + dense-MLP decoder layer. |
| Layers 3–44, before experts | One stage containing attention, mHC/norm work, routing and the shared expert. |
| Each routed expert | One stage for each of the 288 individual expert calls. |
| After those experts | A suffix stage for remaining activation/output handling and residual/mHC combination. |
| Final norm and output head/objective | One root-level suffix stage. |
| Optional vision | Patch preparation, each of the 24 blocks, and post-block merging/projection are separated. |

Sparse-MLA layers are 3, 7, 11, ..., 43; the other 34 layers use KDA. Attention
type affects the stage's graph, but does not introduce a separate attention
boundary. Each MoE decoder layer contributes 290 stage occurrences. Repeated
expert occurrences can reuse compiled contracts; stage count does not equal
compilation count. CPU input preparation can form an additional small stage
when vision separates it from embedding operations.

Use the policy explicitly with any direct planning API:

```python
from workloads.mlops.glm53_flash.partition import GLMStages

# The same partition= argument works for plan_forward, plan_step,
# build_step_programs and plan_step_search.
with plan_forward(
    model, example_inputs=inputs, runtime=runtime,
    execution="execution", spill="spill", partition=GLMStages(),
) as forward:
    outputs = forward(inputs)
```

The direct checkpoint CLI uses this policy too. A custom model can provide a
different `PartitionPolicy`; there is no architecture-name lookup in the
planning API and no GLM rule in the neutral planner.

## Generic quickstart and training sweeps

[quickstart.py](quickstart.py) supplies a text-only LoRA experiment through the
normal quickstart factory interface. Its `plan_options` contains
`partition=GLMStages()`, shared by geometry search and execution planning.
Use `--factory` for this workload; it is not one of the short-name performance
gate presets.

A small random validation model:

```bash
mkdir -p /local/spill
python -m benchmarking.quickstart \
  --factory workloads.mlops.glm53_flash.quickstart:experiment \
  --factory-args '{"tiny":true,"sequence_length":17,"sequences_per_step":2,"sequences_per_microbatch":[1,2],"lora_rank":8}' \
  --search-budget-gib 4 --run-budget-gib 4 --spill-gib 2 \
  --ssd-spill /local/spill --steps 3 --resolution-plans \
  --output-dir /local/results/glm-tiny
```

A local checkpoint uses the same factory and retains bounded, direct-to-spill
initialization. For example, to investigate a four-layer LoRA prefix:

```bash
python -m benchmarking.quickstart \
  --factory workloads.mlops.glm53_flash.quickstart:experiment \
  --factory-args '{"checkpoint":"/local/models/GLM-5.3-Flash-NVFP4","layer_limit":4,"sequence_length":1024,"sequences_per_step":8,"sequences_per_microbatch":[1,2,4],"lora_rank":32}' \
  --search-budget-gib 16,24 --run-budget-gib 24 --spill-gib 64 \
  --ssd-spill /local/spill --steps 3 --resolution-plans \
  --output-dir /local/results/glm-prefix
```

These are configurable experiment examples, not throughput or numerical
qualification claims for a full checkpoint training run. Omit `layer_limit`
for all 45 layers and provision spill capacity for the actual checkpoint plus
trainable state and saved activations. Do not assume the prefix budgets suffice.

Factory arguments: `checkpoint` or `tiny`; `sequence_length`;
`sequences_per_step`; `sequences_per_microbatch`; `lora_rank`;
`lora_alpha`; `lora_dtype`; `lora_head`; `grad_dtype`;
`gemm_precision`; `layer_limit`; `head_chunk_size`; `seed`.
Precision defaults and supported recipes are stated in the function signature.
The checkpoint path rejects unsupported low-precision GEMMs immediately.
The small random fixture additionally permits FP8 expert GEMMs; that is not
equivalent to applying the full checkpoint's quantization recipe.

Each candidate uses the same generated token/target sequence, split into
microbatches. The objective sums token losses and divides by tokens per step.
All candidates therefore describe the same update. This recipe is single-process;
it does not construct distributed groups or imply distributed GLM qualification.

To use `plan_step_search` directly, build/import the factory's model with its
optional `initialize` callback, then pass its `objective`, `optimizer`,
`candidates`, and `**experiment["plan_options"]`. The factory's `context()`
closes any remaining checkpoint readers on failure. Other models can use this
same factory contract without changing quickstart.

## General operation dependencies

The workload composes general MLOps APIs: KDA/decay, sparse MLA/pooled indexing,
mHC, sigmoid routing, and the existing SwiGLU/gated RMSNorm options.
No `mlops.glm` namespace is required. `mlops.bootstrap` registers custom
operations before artifacts are loaded; optional FLA/TileLang code loads only
at execution. Architecture, quantized checkpoint mappings and task partitioning
remain in this workload. SequentialMoE remains the independent
`mlops.sequential_moe` package.
