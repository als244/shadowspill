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

## Direct checkpoint import

The reader also provides the same bounded loading interface as other model
families: `source.copy_chunk(name, shape, offset, destination)`, after declaring
the meta model with `source.build_model(...)`. `destination` is a flat CPU buffer
in the declared dtype; `offset` counts elements of the physical tensor. This
interface is suitable for the shared Executor's tensor-filling callback.
Joined Q/K/V rows and expert banks read only intersecting checkpoint ranges.
BF16 values and FP8 data/scales are preserved without temporary bank assembly.
Adapters require their initializer. Explicit expert-format conversion uses
`load_into`; the bounded direct-copy interface rejects it rather than allocating
a temporary converted bank.

No permanent converted checkpoint is required. The importer:

1. Reads safetensors headers and quantization metadata.
2. Declares the model on `meta`, validates every required tensor, and records physical storage sizes.
3. Fills already-allocated destination tensors in place, writing each destination once. Scratch is one joined projection, or one component of one layer's expert bank.
4. Keeps compressed dense projections' FP8/NVFP4 bytes and scales, for BF16 GEMMs.
5. Stores the routed experts as `expert_storage` says (default: `"bf16"` for a dense checkpoint, `"fp8_block"` -- one copy -- for a compressed one). Dense experts copy into bf16 banks and an FP8 checkpoint's experts into `"fp8_block"` banks byte for byte; every other pairing decodes each expert and quantizes it as the storage does, on the GPU when there is one.
6. Releases consumed mapped source pages. No whole-model host allocation is needed.

Example:

```python
from workloads.models.glm53_flash.checkpoint import GLMCheckpoint
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

For sharded checkpoint loading, quickstart supplies ShadowSpill's
`initialize_storage_` as `load_into(..., write_tensor=...)`. Each direct-copy
physical tensor is read and assembled only by its shard-group leader, then
distributed into owned shards. The loader's default still fills ordinary CPU
tensors independently. LoRA factor initialization retains its RNG sequence;
expert representation conversions retain their per-expert path.

The original Z.ai release uses block FP8. The RedHatAI release mixes 1x16 NVFP4 routed experts with BF16 dense projections and FP8 auxiliary MTP weights. The importer resolves formats per projection and excludes MTP. The experts' activation global scales are read past: the FP8 forms quantize their activations per row per call. Vision is optional; text-only construction leaves its tensors unread.

## CLI

Metadata inspection does not load tensor payloads:

```bash
python -m workloads.models.glm53_flash.forward \
  --checkpoint /local/models/GLM-5.3-Flash-NVFP4 \
  --outdir /local/results/glm-nvfp4 --inspect
```

Forward validation:

```bash
python -m workloads.models.glm53_flash.forward \
  --checkpoint /local/models/GLM-5.3-Flash-NVFP4 \
  --outdir /local/results/glm-nvfp4 \
  --ssd-directory /local/spill --execution-gib 24 --passes 3 \
  --minimum-evict-kib 128 \
  --text "The capital of France is"
```

`--spill-gib` overrides the default of the planner's minimum reservation (including its admission margin), rounded up to GiB plus 4 GiB. `--layer-limit N` selects a decoder prefix for debugging and is explicitly recorded in the inventory. Only BF16 GEMMs are currently validated for the composed checkpoint path; unsupported recipes fail rather than silently selecting another one.

The sparse-attention indexer is mlops's `pooled_topk`, which scores each chunk
of queries against every pooled key in one fused kernel and selects each query's
pools with a second; a 32K-token call is a few dozen allocations and about
256 MiB of scores at a time, within what a profiled task may make.

A checkpoint is imported once into a pool image with
`python -m workloads.models.glm53_flash.convert --checkpoint DIR --image FILE
[--layer-limit N] [--lora-rank R] [--expert-storage S]`: the frozen model as
ShadowSpill lays it out in an SSD pool file, with a manifest beside it. The
frozen tensors are named the same whatever the LoRA rank, so an image serves
every LoRA run with its expert storage. A quickstart given `--ssd-image FILE` binds the
frozen tensors where they lie and writes only LoRA factors, optimizer state and
activations; `GLMCheckpoint.load_into(model, adopted=...)` skips the adopted
copies.

Checkpoint import preserves the source dtype of each direct-copy tensor,
including BF16 mHC projections, routers and KDA convolutions. mHC bias/scale
and KDA control tensors remain checkpoint FP32. The mHC projection and
coefficients still compute in FP32 while residual streams stay BF16. Random
model construction retains its separate FP32 training defaults.

The `--minimum-evict-kib` option configures the ordinary planner's small-object
policy (default 1024 KiB; zero allows every object to be evicted). Objects below
it are retained between uses and currently receive separate reserved physical
slots. For very many small quantization scales, a lower value such as 128 KiB
can recover substantial execution-pool capacity, even in a forward-only plan.

The CLI saves the exact prompt/token IDs, options, metadata, build/plan artifacts,
per-call times/top tokens/log probabilities and final logits. Repeatable finite logits are a smoke check, not a substitute for reference-model numerical parity.

## KDA projection boundary

The MLOps implementation passes separate Q/K/V projections to
`mlops.causal_conv_silu_qkv`. Packed sequence resets, SiLU and checkpoint
convolution dtypes are unchanged. Frozen convolution weights use a VJP that
reads separate Q/K/V gradients and writes contiguous projection gradients;
full weight training retains the ordinary composition. Layout kernels and
provider dispatch live in MLOps. The family owns projections, Q/K normalization,
decay/beta/output gates and the attention call.

## Development status

The importer is tested on synthetic checkpoints -- dense, and FP8 with 128x128
blocks -- in every expert storage (`tests/workloads/test_glm53_checkpoint.py`).
The full checkpoints' planned forwards, their comparison with the streamed
HF-definition reference, and LoRA training on real prefixes are to be repeated
with the stacked experts. The supported checkpoint recipe keeps compressed
dense projections with BF16 GEMMs; full-size LoRA training remains unqualified.

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

## Multiple prompts with one plan

Repeat `--text` to reuse one model import and plan:

```bash
python -m workloads.models.glm53_flash.forward \
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
    lora_factor_dtype=torch.bfloat16,
    lora_head=False,
)
```

The base, router, shared experts, indexer and normalization weights remain frozen.
Dense attention/MLP projections and routed experts train through LoRA; the output
head is optional. The experts' factors are stacked per layer (`lora_gate_up_a
[E, R, H]`, `lora_gate_up_b [E, 2I, R]`, `lora_down_a [E, R, I]`, `lora_down_b
[E, H, R]`) and train through `expert_lora`; their base GEMMs run at the expert
storage's precision. FP32 factor storage yields FP32 factor gradients. Import
initializes supplied pool state one projection/factor at a time. Meta tensors
alone contain no initialized factor values.

Composed training and checkpoint validation remain tracked in the internal agenda.
Full-model LoRA training is not yet qualified.

## Images

The optional vision encoder follows the source checkpoint's 24-block architecture,
axial RoPE, patch merging and projector. Normalization and RoPE arithmetic use
FP32; projections use BF16. Geometry is prepared before capture; attention does
not read GPU metadata back to Python.

```bash
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 \
python -m workloads.models.glm53_flash.forward \
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

By default the planner partitions GLM as it does any model. `GLMStages`
([partition.py](../shared/glm/partition.py)) is an optional `PartitionPolicy`, kept apart
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

Use the policy explicitly with any direct planning API:

```python
from workloads.models.glm53_flash.partition import GLMStages

# The same partition= argument works for plan_forward, plan_step,
# build_step_programs and plan_step_search.
with plan_forward(
    model,
    example_inputs=inputs,
    runtime=runtime,
    execution="execution",
    spill="spill",
    partition=GLMStages(),
) as forward:
    outputs = forward(inputs)
```

The forward CLI takes `--partition glm-stages` and the quickstart factory
`partition="glm_stages"`. A custom model can provide a different
`PartitionPolicy`; there is no architecture-name lookup in the planning API
and no GLM rule in the neutral planner.

## Generic quickstart and training sweeps

[quickstart.py](quickstart.py) supplies model configuration to the
[shared phase runner](../../../benchmarking/quickstart.md). Both GLM families
use the same build, profile, communication calibration, CPU planning and
fresh-worker execution as the other workloads. Family helpers own only model
construction, input metadata, the objective, precision and initialization.

A small random validation model:

```bash
python -m benchmarking.quickstart \
  --factory workloads.models.glm53_flash.quickstart:experiment \
  --factory-args '{"tiny":true,"sequence_length":17,"sequences_per_step":2,"sequences_per_example":[1,2]}' \
  --search-budget-gib 4 --run-budget-gib 4 --spill-gib 2 \
  --steps 3 --output-dir /local/results/glm-tiny
```

With a local checkpoint, `initialization="random"` reads configuration and
safetensors headers only. All models, including the tiny fixture, are declared
on meta. Execution fills Executor-provided CPU chunks in the declared BF16,
FP32 and FP8 dtypes; no full-model or expert-bank staging copy is created.
Generated values use fixed seeded blocks independent of request/shard boundaries,
positive FP8 scales and zero LoRA B factors. This retains the initialization
distributions. Values depend on tensor names and fixed RNG blocks, so they are
independent of initialization chunk boundaries and shard ownership.

`initialization="checkpoint"` reads actual weights into the same bounded chunks.
Direct BF16 and block-FP8 experts preserve checkpoint bytes. Expert format
conversion, including NVFP4 experts and two-orientation FP8 storage, is not yet
supported by this bounded recipe; it fails explicitly before build. The separate
checkpoint reader's existing `load_into` conversion path remains available.
Random initialization supports ordinary dense and block-FP8 storage.

A four-layer checkpoint prefix:

```bash
python -m benchmarking.quickstart \
  --factory workloads.models.glm53_flash.quickstart:experiment \
  --factory-args '{"checkpoint":"/local/models/GLM-5.3-Flash-FP8","initialization":"checkpoint","layer_limit":4,"sequence_length":1024,"sequences_per_step":8,"sequences_per_example":[1,2,4]}' \
  --search-budget-gib 16 24 --run-budget-gib 24 --spill-gib 64 \
  --steps 3 --output-dir /local/results/glm-prefix
```

Defaults are rank-32 BF16 adapters, alpha 32, FP32 accumulated gradients and
BF16 AdamW moments with stochastic rounding. Frozen tensors retain checkpoint
precision; dense compressed projections decode for BF16 GEMMs and block-FP8
experts use FP8 GEMMs. `master_dtype`, `grad_dtype`, `opt_state_dtype` and
`opt_state_rounding` control trainable state independently. LoRA targets are
family-defined main projections; auxiliary gates and indexers remain frozen.

`partition="glm_stages"` selects the family partition policy; otherwise build
uses its automatic policy. `build_options` carries this and the other builder
controls. Every candidate uses the same rank-local token stream. The objective
sums token losses and divides once by the global token count.

For eight ranks with weights, gradients and optimizer data sharded across the
replica, use the same runner and declare `ranks` in the factory arguments:

```bash
python -m benchmarking.quickstart \
  --factory workloads.models.glm53_flash.quickstart:experiment \
  --factory-args '{"checkpoint":"/local/models/GLM-5.3-Flash-FP8","ranks":8,"sequence_length":4096,"sequences_per_step":64,"sequences_per_example":[2]}' \
  --search-budget-gib 70 --run-budget-gib 70 --spill-gib 96 \
  --steps 5 --output-dir /local/results/glm-dp-sharded
```

Build and compute profiling use one GPU. The runner launches all ranks only for
communication calibration and execution. No live process group is consulted by
the factory. `num_replicas` repeats the complete replica; `weights`, `gradients`
and `optimizer` choose storage modes through the common `Parallelism.dp` API.
`sequences_per_step` is per rank; throughput counts global tokens.

The budgets above are examples, not measured feasibility or performance claims.
The original FP8 Flash model occupies about 303 GiB including adapters before
trainable state and activations. Eight 96 GiB spill pools reserve 768 GiB of host
memory plus process/compiler overhead. Each GPU still needs a complete stage's
execution tensors. Full-checkpoint distributed training needs separate validation.

Saved Programs and ProfiledPrograms can be reused by the independent phases.
For example, `--phase plan --output-dir ...` replans without importing Torch or
the model. The optional factory `context()` scopes build settings and closes
checkpoint readers after build/initialization, including on failure. Profiling
and execution load finalized task code. See the shared runner guide for the
complete argument contract and artifact layout.

## General operation dependencies

The workload composes general MLOps APIs: KDA/decay, sparse MLA/pooled indexing,
mHC, sigmoid routing, and the existing SwiGLU/gated RMSNorm options.
No `mlops.glm` namespace is required. `mlops.bootstrap` registers custom
operations before artifacts are loaded; optional FLA/TileLang code loads only
at execution. Architecture, quantized checkpoint mappings and task partitioning
remain in this workload. The checkpoint reader and the compressed dense
projections come from `mlops.sequential_moe.precision`; the MoE itself is
`mlops.moe` / `expert_lora`.

## Real-text LoRA recipe

`workloads.recipes.text.train_glm53_flash_lora` loads a local checkpoint into
sharded backing storage and trains on a corpus tokenized with that checkpoint.
It defaults to 32K sequences, checkpoint FP8 expert storage and GEMMs, BF16
dense GEMMs/adapters, and FP32 adapter masters, gradients and AdamW moments.
Launch with `torchrun --standalone --nproc-per-node=8 -m` and provide
`--checkpoint`, `--data`, `--frozen-state-id`, and `--run-dir`. The corpus
`meta.json` must contain `tokenizer_sha256` for both `tokenizer.json` and
`tokenizer_config.json` from the base checkpoint.

Each rank consumes distinct sequences; objectives sum globally normalized
next-token loss. `global_mean_loss` reports the global token mean, while the
Trainer's ordinary `loss` remains the rank's contribution. Held-out evaluation
and adapter-only checkpoints are enabled. `--resume` restores adapters,
optimizer, schedules/RNG and data progress with the same base identity.
`--layer-limit` is an explicit debugging prefix, never a full-model result.
Full-model 32K training qualification is in progress; see the experiment results
before relying on this as a qualified configuration.

Router weights default to BF16 for random model construction; `Config(router_weight_dtype=torch.float32)` opts into FP32. Checkpoint import preserves source router storage by default (BF16 in GLM-5.3-Flash). `build_model(router_weight_dtype=torch.float32)` or the real-text recipe’s `--router-weight-dtype fp32` explicitly widens only router weights. Quickstart accepts `router_weight_dtype="float32"`. Routing logits remain FP32.


LoRA defaults are declared in [`lora.py`](lora.py), shared by the family's
implementations; explicit caller targets override the defaults. The five KDA
control/output-gate projections are frozen; attention Q/K/V/O and MLA
projections, dense MLP projections, and routed expert factors are adapted.
