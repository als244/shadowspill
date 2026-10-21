# Megatron-Core over the catalog models

This harness trains NVIDIA's own implementation of a catalog model's
architecture with Megatron Bridge's training loop over Megatron-Core, under
the shared objective, geometry, precision contract and result row of
`baselines/common`. Bridge builds the model from the case's Hugging Face
configuration (or from a Bridge recipe of the same architecture) and runs
its `pretrain` loop with the case's settings; the tokens are the ones every
harness trains on.

Every Megatron option a case can set is a setting under Megatron's own
name, left to Bridge's default (or the recipe's value) unless the case names
it. The harness refuses only what it cannot say or measure: a value of the
wrong type or outside Megatron's choices, a precision the contract asks for
that Megatron's options cannot express, the shared settings where they
contradict Megatron's own or a recipe's, and a graph capture during a
measured update. What goes together in Megatron is Megatron's and Bridge's
to decide as they set up; a case they refuse fails there, before training,
with their message. This page gathers their rules, with the combinations
they accept but quietly change or ignore.

**Contents.** [Running a case](#running-a-case) ·
[The model](#the-model) · [Shared settings](#shared-settings) ·
[Megatron's options](#megatrons-options) · [Precision](#precision) ·
[Parallelism](#parallelism) ·
[Data parallelism and communication](#data-parallelism-and-communication) ·
[Recomputation and activation offloading](#recomputation-and-activation-offloading) ·
[Optimizer offloading](#optimizer-offloading) ·
[Mixture of experts](#mixture-of-experts) ·
[Attention, fusions and kernels](#attention-fusions-and-kernels) ·
[Graph capture](#graph-capture) ·
[The process environment](#the-process-environment) ·
[Data](#data) · [Measurement](#measurement) · [What a run writes](#what-a-run-writes) ·
[Hardware limits](#hardware-limits)

## Running a case

The trainer runs in the `nvidia-training` environment, which one command
builds on any machine, with or without a GPU:

```bash
bash baselines/scripts/setup_nvidia_training.sh
```

It creates a Conda environment named `nvidia-training` with Python 3.12 and
clones the pinned checkouts inside it, in `<prefix>/src`: Megatron Bridge with
its Megatron-LM, NeMo AutoModel, APEX and NVIDIA Resiliency Extension.
Megatron-Core, Bridge and AutoModel are installed editable from their
checkouts, Transformer Engine at the revision Bridge pins is built against
PyTorch 2.13, and APEX and NVIDIA Resiliency Extension are built with their
compiled extensions. The versions the pins leave open are preferred from
NVIDIA's own lock files at the pinned revisions (Bridge's `uv.lock`, and
AutoModel's for what Bridge's lacks or AutoModel's requirements exclude), and
the resolution departs from one only where a requirement forces it, listing
what departs in `<prefix>/environment.departures`; PyTorch 2.13 keeps its own
pins (triton and NVIDIA's libraries). NCCL is PyTorch's own 2.29.7 unless 9.0
or 10.0 is a target. There the setup builds Transformer Engine with its NCCL
expert-parallel library (Megatron's `ncclep` dispatcher), with NCCL 2.30.7,
the release that library is validated against, and DeepEP, for Megatron's
`deepep` and `hybridep` dispatchers, as Bridge's CI image builds it: Bridge's
pinned commit and patch, NVSHMEM 3.6.5 over PyTorch's pin, and HybridEP across
nodes where the machine has rdma-core's headers. A build that fails with
either stops nothing else: Transformer Engine is built again without NCCL EP,
and the environment without DeepEP. The compilers are the environment's own,
NVIDIA's toolkit wheels at the release PyTorch's own wheels are built with,
laid out in `<prefix>/cuda-home`; no toolkit the machine has is used.
Transformer Engine and APEX are compiled for the architectures of the GPUs the
machine that builds shows, or for the list `--cuda-archs` names, where the
machine that builds is not the one that trains (a cluster's login node
building for its compute nodes); with no GPU and no list, each build takes its
own list (`--jobs` caps the parallel compilations, `--sources DIR` clones or
finds the checkouts elsewhere, and `--env NAME` or `--python PATH` choose
another environment). What it installed is recorded inside the environment, in
`<prefix>/environment.lock.json`.

One case, through the launcher every harness starts from, run from the
repository root in the environment the sweeps run in:

```bash
python -m baselines.common.launch --harness megatron_core --gpus 1 \
  --output /mnt/storage/research/shadowspill/baselines/dev/megatron-llama3-tiny \
  --model llama3-tiny --sequence-length 64 --tokens-per-step 1024 \
  --tokens-per-microbatch 128 --warmup-steps 1 --steps 2
```

The launcher validates and resolves the case in its own interpreter, writes
`case.json`, sets the environment variables the case decides
([The process environment](#the-process-environment)), and starts torchrun
with the `nvidia-training` environment's interpreter (`--python` names
another).

The tiny qualification matrix:

```bash
python -m baselines.sweep --harness megatron_core \
  --config baselines/megatron_core/configs/qualification.json \
  --output /mnt/storage/research/shadowspill/baselines/dev/megatron-qualification --gpus 1
```

Its cases run each precision, recomputation and offloading mechanism on the
tiny models, the offloading cases at sequences long enough that the values
they move are above Transformer Engine's size floor, and every case's first
update checks its dtypes against the contract. There is no reference
comparison, since the model is NVIDIA's. Its LoRA case adapts a tiny
checkpoint that
`python -m baselines.scripts.write_checkpoint --model llama3-tiny --output runs/checkpoints/llama3-tiny`
writes. A throughput matrix names only Megatron's own options and runs inside
an experiment, which gives every harness the same model, geometry and
precision contract (`../README.md`).

## The model

A case names a catalog model. Its Hugging Face configuration comes from the
named checkpoint's `config.json`, or from the family's `to_huggingface` (each
family's `config.py`) for the catalog entry with the case's `--model-config`.
Bridge's `AutoBridge` turns that configuration into NVIDIA's model provider,
built without weights. `--checkpoint DIR` names a local Hugging Face
checkpoint Bridge imports into the model (`checkpoint.pretrained_checkpoint`);
without it the model trains from Megatron's initialization, seeded by
`--seed` (plus one: Bridge refuses a seed of zero). `--recipe NAME` starts
instead from a recipe in `megatron.bridge.recipes`, whose model must have the
case's architecture (layers, widths, heads, query groups, experts, tied
embeddings). The case's geometry, precision, LoRA and options apply over the
recipe's values (in full mode a recipe's LoRA is dropped), and what the
harness's geometry and rules take as given must be what the recipe leaves: a
recipe that sets the tensor, pipeline, context or virtual-pipeline sizes,
layer-level or fine-grained offloading, graph capture, the optimizer's graph
or Megatron-FSDP otherwise than the case does, by name or by default, is
refused as a different architecture is, and the case names them instead. A
recipe's own environment variables are refused unless the case's
environment already holds them.

The families with a Hugging Face counterpart are Llama 3, Qwen 3, OLMoE,
Qwen 3 MoE, Qwen 3.5 and Qwen 3.5 MoE. Qwen 3.5 has one only with
zero-centered norms (`--model-config '{"zero_centered_norm": true}'`), and
Qwen 3.5 MoE also with its attention gate, since Hugging Face's
implementation has no other form. Bridge's OLMoE layers normalize the keys
over all attention heads, so they take OLMoE as it was released, with as many
key-value heads as heads; the qualification matrix's tiny OLMoE sets
`n_kv_heads` so. A family without `to_huggingface` is refused before launch.

Bridge builds the layers of Llama 3, Qwen 3, OLMoE and Qwen 3 MoE from
Transformer Engine's modules. A layer's norm before attention is part of its
query, key and value projection (one module, `linear_qkv`, that normalizes
and then projects), and in a dense model the norm before the MLP is likewise
part of the MLP's first projection (`linear_fc1`). Of the norms before
attention and before the MLP, only an MoE layer's norm before its router is
a module of its own; Qwen 3, Qwen 3 MoE and OLMoE also normalize the query
and the key with modules of their own (`q_layernorm`, `k_layernorm`). This
decides what the norm options of recomputation and offloading do
([below](#what-a-layer-keeps)). Qwen 3.5's Gated DeltaNet layers are built
otherwise and are not described here.

The row names the implementation `megatron_core`. Its useful work is counted
from the catalog's own model of the same architecture, built on `meta` by the
launcher, with the convention every harness uses, so recomputation, the
optimizer and communication are not credited.

LoRA (`--mode lora`) is Bridge's own layout over the shared definition's
targets (`workloads/lora`): one adapter of the case's rank on each
projection module Megatron builds, fused or not (attention's `linear_qkv`
and `linear_proj`, Gated DeltaNet's `in_proj` and `out_proj` in Qwen 3.5,
a dense MLP's `linear_fc1` and `linear_fc2`), and one on each projection of
every routed expert; the head stays frozen. The shared experts are adapted
only when the case's `shared_experts` entry is true; otherwise the harness
names target patterns that leave `*.shared_experts.*` out. Bridge builds the
factors at the model's parameter dtype, so a `factor_dtype` other than the
model's is refused, and it scales an adapter's output by alpha over the
rank, so `alpha` is passed as given. It adapts a pretrained model, so a LoRA
case names its checkpoint (Bridge refuses one without), and it takes the
`rank`, `alpha`, `factor_dtype`, `head` (frozen) and `shared_experts`
entries of `--lora`.

The layouts differ between frameworks: where the catalog adapts the query,
key and value projections, or the gate and up projections, one by one,
Bridge adapts its fused `linear_qkv` or `linear_fc1` once. Harnesses are
compared by their total trainable parameters, the elements of every
trainable tensor, which each row records from the model Megatron built.

## Shared settings

| Flags | Values (default) | Meaning |
|---|---|---|
| `--model`, `--model-config JSON` | a catalog name; field overrides | The catalog model and overrides of its configuration's fields |
| `--implementation` | `megatron_core` | NVIDIA's implementation, the only one this harness trains |
| `--mode`, `--lora JSON` | `full` or `lora` (`full`); LoRA options | Full or LoRA training ([The model](#the-model)) |
| `--checkpoint DIR`, `--recipe NAME` | a directory; a recipe's name (none) | Weights to import, and a Bridge recipe to start from |
| `--sequence-length`, `--tokens-per-step`, `--tokens-per-microbatch` | tokens | One update's geometry, in whole sequences. A replica is tensor x pipeline x context parallel ranks, and the data-parallel ranks are the replicas (`baselines.common.geometry`) |
| `--model-dtype`, `--master-dtype`, `--grad-dtype`, `--opt-state-dtype`, `--parameter-rounding`, `--opt-state-rounding`, `--[no-]fuse-narrow-gradient-accumulation` | the contract's | The precision contract ([Precision](#precision)) |
| `--data DIR` | a corpus (synthetic tokens) | A corpus `python -m baselines.scripts.prepare_corpus` wrote ([Data](#data)) |
| `--recompute F` | 0 to 1 (none) | Full recomputation of the first share F of each pipeline stage's layers, or of each model chunk's with virtual stages (Megatron's `full` granularity, `block` method). None leaves recomputation to Megatron's own options. A case names `--recompute` or Megatron's own (`--recompute-granularity`, `--recompute-method`, `--recompute-num-layers`, `--recompute-modules`), not both, and `--recompute` is refused beside uneven pipeline stages (the first or last stage's layer count, or the `account_for_*` options), whose stages are not the even split it counts from |
| `--warmup-steps`, `--steps` | updates (3, 5) | Excluded warmups and measured updates |
| `--learning-rate`, `--weight-decay`, `--seed` | (3e-4, 0.01, 0) | AdamW's rate (constant) and decay; betas 0.9 and 0.999, epsilon 1e-8, no clipping |
| `--[no-]nsys-capture` | (off) | Brackets the measured updates with the profiler; not a throughput result |

`--verify-reference` is refused: the reference check compares updates of the
catalog's own implementation, and this harness trains NVIDIA's.

## Megatron's options

Each option below is `--<name-with-dashes> VALUE`; a boolean is
`--<name>` or `--no-<name>`, and a list of modules is comma-separated. The
default in each table is what Bridge builds when the case names nothing:
for Llama 3, and for Qwen 3 MoE where it differs. A model's bridge may set
others, and each rank's request records every option's value in the
configuration Bridge finalized ([What a run writes](#what-a-run-writes)).
The last column gives Megatron's and Bridge's own rules for the option,
which the harness leaves to them; where the harness refuses a value itself,
the column says so.

## Precision

[`../PRECISION.md`](../PRECISION.md) sets this harness beside the others.

### The contract in Megatron's terms

| The contract | In Megatron |
|---|---|
| Model dtype | `bfloat16` (Bridge's bf16 mixed precision, the weights kept at bfloat16) or `float32`, whose masters are what the optimizer steps: under DDP the float32 parameters themselves, under Megatron-FSDP with float32 masters its float32 main weights; `float16` would need loss scaling the contract does not name |
| Masters | `float32` by default: Megatron's mixed-precision optimizer keeps float32 copies of the bfloat16 weights, or the distributed optimizer keeps float32 shards of them. `float16` through the precision-aware optimizer ([below](#the-optimizer)). `none`, stepping the weights themselves, only under Megatron-FSDP (`--use-megatron-fsdp`), whose main weights are then the compute weights (`megatron_fsdp_main_params_dtype` none); Megatron's DDP optimizers always keep masters |
| Gradients | Megatron's DDP accumulates and reduces in one buffer per bucket, at float32 with `grad_reduce_in_fp32` (gradient dtype `float32`) or at the weights' dtype; Megatron-FSDP at any dtype (`megatron_fsdp_main_grads_dtype`). Another dtype under DDP is refused |
| Moments | `float32` by default; `float16` or `bfloat16` through the precision-aware optimizer. With no master (Megatron-FSDP), the case's `--opt-state-dtype`, `float32` or none (Transformer Engine's fused Adam's own float32): other moments need the precision-aware optimizer, which the harness turns on only beside masters, since it holds masters itself (except under Megatron-FSDP version 2) |
| Roundings | Rounding to nearest only (Transformer Engine's fused Adam, PyTorch's) |
| Fused narrow-gradient accumulation | `gradient_accumulation_fusion` for gradients narrower than float32 ([below](#gradient-accumulation)); refused under Megatron-FSDP version 2, which turns the fusion off |

The harness sets Bridge's mixed-precision configuration from the contract
and the case's FP8 or FP4 options. With neither it names no FP8 recipe, since
Bridge's default recipe name, carried onto the optimizer, would keep the
precision-aware optimizer from holding its masters itself. The first update
checks the weights (at the model dtype; a trainable parameter the model
itself builds at another dtype, such as a norm's kept at float32, is counted
apart), the accumulated gradients, the masters (what the optimizer steps)
and the moments against the contract, and a case whose step holds another
dtype fails there.

### Gradient accumulation

Every weight whose gradient a matrix multiply computes (every linear layer's)
can have the multiply write its gradient where the step keeps it. With
`gradient_accumulation_fusion`, Transformer Engine's weight-gradient
multiply adds into the accumulation buffer as it writes it: into a float32
buffer the multiply's float32 sums go unrounded, and into a bfloat16 buffer
the product is added and rounded once, as the BLAS kernel does it. Without
it, the multiply writes the weight's own gradient at the weight's dtype and
Megatron adds that into the buffer after. Bridge turns the fusion on
whenever Transformer Engine imports, as it does in the `nvidia-training`
environment; Megatron-Core's own output layer then fuses through APEX's
`fused_weight_gradient_mlp_cuda` extension, which the environment installs,
and refuses to build without it. Megatron-FSDP version 2 turns the fusion
off for every layer.

For gradients narrower than float32,
`--[no-]fuse-narrow-gradient-accumulation` is the contract's name for this
choice and sets `gradient_accumulation_fusion` (on by default); naming
`gradient_accumulation_fusion` against the contract is refused, and so is
the fused accumulation under Megatron-FSDP version 2. For float32 gradients
the contract asks nothing, and `gradient_accumulation_fusion` is the case's
or Bridge's.
[`../PRECISION.md`](../PRECISION.md#gradients-creation-accumulation-and-reduction)
compares the harnesses.

### The optimizer

**The distributed optimizer.** Without it, every data-parallel rank keeps
the whole optimizer state and all the masters, gradients are all-reduced,
and every rank steps every weight. With `--use-distributed-optimizer`
(ZeRO-1), each rank keeps only its slice of the masters and the moments:
gradients are reduce-scattered so that each rank receives the sum of its
slice, each rank steps its slice, and the updated weights are all-gathered
back. The weights and the gradient buffer stay whole on every rank;
Megatron-FSDP's sharding strategies shard those too
([Megatron-FSDP](#megatron-fsdp)).

**The precision-aware optimizer.** `--use-precision-aware-optimizer` (Adam
only, with the distributed optimizer) steps with Transformer Engine's fused
Adam holding the masters and the moments at chosen dtypes, in place of
Megatron's float32 shards. The harness turns it on, with the distributed
optimizer, whenever the contract keeps masters and asks for masters or
moments other than float32, and sets its dtypes from the contract: masters
at float32 or float16 (Transformer Engine refuses bfloat16 masters),
moments at float32, float16 or bfloat16. Naming it off against such a
contract is refused.

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]use-distributed-optimizer` | (off) | ZeRO-1, as above | Bridge keeps the DDP and optimizer copies of the setting in step; Megatron-FSDP version 1 turns it on, version 2 off |
| `--num-distributed-optimizer-instances N` | integer (1) | Shards the state within N groups of the data-parallel ranks instead of across all of them, each group a full copy | The distributed optimizer |
| `--[no-]overlap-param-gather` | (off) | Gathers the updated weight shards bucket by bucket while the next forward runs | Applies with the distributed optimizer |
| `--[no-]overlap-param-gather-with-optimizer-step` | (off) | Starts the first bucket's gather during the optimizer step | Megatron-LM's command line requires the distributed optimizer, `overlap_param_gather` and virtual stages; Bridge does not check |
| `--[no-]use-precision-aware-optimizer` | (off; on where the contract needs it) | As above | Adam; the distributed optimizer; Megatron-LM's command line also wants it beside `optimizer_cpu_offload`, which Bridge does not check |
| `--[no-]store-param-remainders` | (on) | Keeps a float32 master as the bfloat16 weight plus the master's low 16 bits: the float32 master exactly, in half the memory of a separate copy | Float32 masters held by Transformer Engine |
| `--[no-]optimizer-cuda-graph` | (off) | Captures the optimizer step as a graph (Transformer Engine's capturable Adam), inside update N's step ([Graph capture](#graph-capture)) | Float32 masters; moments both float32 or both bfloat16; `--no-store-param-remainders`; N among the warmup updates |

### FP8 and FP4

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--fp8 F` | `e4m3`, `hybrid` (none: no FP8) | FP8 matrix multiplies through Transformer Engine: `e4m3` for every FP8 value, or `hybrid`, with e5m2 for the gradients | Compute capability 8.9 or later |
| `--fp8-recipe R` | `tensorwise`, `delayed`, `mxfp8`, `blockwise`, `custom` (`tensorwise`) | The scaling: current per-tensor scaling, delayed scaling from an amax history, MXFP8 block scaling, blockwise scaling, or a custom recipe of the quantizers `fp8_quantizer_factory` makes | `mxfp8` needs compute capability 10.0 and is not supported on 12.0; `blockwise` needs 9.0 and a PyTorch built with toolkit 12.9 or later (the environment's is built with 13.0); `custom` needs `fp8_quantizer_factory` |
| `--[no-]fp8-param-gather` | (off) | Keeps the compute weights at FP8 and gathers them at FP8 | Needs `fp8`; with `mxfp8` and no Megatron-FSDP, Bridge also requires `reuse_grad_buf_for_mxfp8_param_ag`, which no harness option sets, so that combination is refused |
| `--[no-]fp8-wgrad` | (on) | Computes the weight gradients at FP8; off, at higher precision | Read only with `fp8` |
| `--fp8-amax-history-len N`, `--fp8-amax-compute-algo A` | integer (1); `most_recent`, `max` (`most_recent`) | Delayed scaling's amax history, and how it reduces it | Delayed scaling |
| `--[no-]fp8-dot-product-attention`, `--[no-]fp8-multi-head-attention` | (off) | FP8 attention | Read only with `fp8` |
| `--[no-]first-last-layers-bf16`, `--num-layers-at-start-in-bf16 N`, `--num-layers-at-end-in-bf16 N` | (off); integers (0) | Keeps the first and last layers at bfloat16 under FP8 | Not with delayed scaling; the counts lie within a stage |
| `--fp4 F`, `--[no-]fp4-param-gather` | `e2m1` (none); (off) | FP4 matrix multiplies, and FP4 compute weights gathered at FP4 | Not with `fp8`; compute capability 10.0 or later |
| `--fp4-recipe R` | `nvfp4`, `custom` (`nvfp4`) | The FP4 scaling: NVFP4 block scaling, or a custom recipe of the quantizers `fp4_quantizer_factory` makes | Read only with `fp4`; `custom` needs `fp4_quantizer_factory` |
| `--fp8-quantizer-factory P`, `--fp4-quantizer-factory P` | an import path (none) | A custom recipe's quantizers: the Python import path (`package.module.factory`) of the callable that makes them, which Megatron imports in the trainer and turns into Transformer Engine's custom recipe | Read only with the `custom` recipe, which needs it |
| `--[no-]activation-func-fp8-input-store` | (off) | Stores the SwiGLU activation's input at FP8 for its backward | SwiGLU |

Delayed scaling also refuses `moe_act`, `layernorm` and
`shortcut_pre_mlp_layernorm` recomputation.

## Parallelism

### Tensor and sequence parallelism

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--tensor-model-parallel-size N` | integer (1) | Splits each layer's weight matrices across N ranks, the attention by heads and the MLP by its hidden width, with the activations between them | The attention heads divide over N, and the key-value groups divide over N or N over them |
| `--[no-]sequence-parallel` | (off) | Splits the activations outside the tensor-parallel multiplies (norms, dropout, residual adds) along the sequence, so each rank keeps 1/N of them | Bridge turns it off without tensor parallelism; Megatron warns that expert parallelism beside tensor parallelism wants it |

### Pipeline parallelism

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--pipeline-model-parallel-size N` | integer (1) | Divides the layers into N stages, one group of ranks each, microbatches flowing between them | The layers divide over the stages, unless the first or last is uneven |
| `--virtual-pipeline-model-parallel-size V` | integer (none) | Gives each pipeline rank V model chunks, interleaved to shrink the pipeline bubble | A stage's layers divide into its chunks; without pipeline parallelism Bridge builds one chunk |
| `--num-layers-in-first-pipeline-stage N`, `--num-layers-in-last-pipeline-stage N` | integers (none: even) | Uneven first or last stages | Not with the `account_for_*` options, nor beside the shared `--recompute` |
| `--[no-]account-for-embedding-in-pipeline-split`, `--[no-]account-for-loss-in-pipeline-split` | (off) | Counts the embedding, or the loss, as a layer when dividing the stages | Not with uneven stages, nor beside the shared `--recompute` |
| `--[no-]overlap-p2p-comm`, `--[no-]batch-p2p-comm` | (off), (on) | Overlaps the pipeline's sends and receives with compute, or batches them into one call | Documented as one or the other; Bridge's overlap configuration derives them so |
| `--[no-]deallocate-pipeline-outputs` | (on) | Frees a stage's output once it is sent to the next stage | |
| `--microbatch-group-size-per-vp-stage N` | integer (the pipeline size) | Microbatches a virtual stage runs at a time | Interleaved pipelines only |
| `--[no-]defer-embedding-wgrad-compute`, `--wgrad-deferral-limit N` | (off); integer (0: every microbatch) | Defers the embedding's weight-gradient multiplies into the pipeline's flush, for this many microbatches | Pipeline parallelism and `gradient_accumulation_fusion`; Bridge's overlap configuration derives deferral off with a limit of -1, and Megatron refuses a negative limit only with deferral on, so a case with `tp_comm_overlap` that turns deferral on names the limit |

### Context parallelism

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--context-parallel-size N` | integer (1) | Splits each sequence across N ranks, attention exchanging keys and values among them | The sequence length divides into 2 x N chunks |
| `--cp-comm-type T` | `p2p`, `all_gather`, `a2a`, `a2a+p2p` (`p2p`) | The exchange: a ring overlapped with attention, an all-gather, an all-to-all over heads (Ulysses-style), or both, hierarchically | `a2a+p2p` needs `hierarchical_context_parallel_sizes` |
| `--hierarchical-context-parallel-sizes A,B` | two integers (none) | The all-to-all and ring degrees of `a2a+p2p` | Their product is the context-parallel size |

### Expert parallelism

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--expert-model-parallel-size N` | integer (1) | Distributes the routed experts over N ranks, out of the data-parallel ranks | MoE models only |
| `--expert-tensor-parallel-size N` | integer (1 under expert parallelism, Bridge's; the tensor-parallel size otherwise) | Splits each expert's weights across N ranks | |

### How a rank runs its microbatches

With one pipeline rank, Megatron runs each microbatch's whole forward, then
its whole backward, one microbatch after another (gradient reduction waits
for the last). Virtual stages and `microbatch_group_size_per_vp_stage` exist
only across pipeline ranks, which are separate processes. The one exception
is expert all-to-all overlap (`overlap_moe_expert_parallel_comm`), whose
one-rank schedule runs one microbatch's forward layer by layer beside the
previous microbatch's backward, to hide the experts' all-to-alls behind
compute.

## Data parallelism and communication

### Megatron's DDP

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]overlap-grad-reduce` | (off) | Reduces each bucket of gradients while backward continues | |
| `--[no-]align-param-gather` | (off) | Launches every pipeline stage's weight gathers together | |
| `--bucket-size N` | parameters (Megatron's: the larger of 40M and 1M per data-parallel rank) | Parameters per gradient-reduction and weight-gather bucket | Read only with `overlap_grad_reduce`: without it, and on every pipeline stage after the first, the parameters form one bucket |
| `--[no-]average-in-collective` | (off) | Averages in the reduction itself instead of dividing first | |
| `--[no-]reduce-scatter-with-fp32-accumulation` | (off) | Sends gradients at their dtype and sums them at float32 | |

### Megatron-FSDP

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]use-megatron-fsdp` | (off) | Megatron-Core's own FSDP in place of its DDP; the one way to step the compute weights without masters | Version 1 turns on the distributed optimizer and refuses `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| `--megatron-fsdp-version V` | `1`, `2` (`1`) | Megatron-FSDP's implementation | Version 2: bfloat16 only; no tensor, pipeline or virtual-pipeline parallelism, no optimizer offload, no FP8 or FP4, and no checkpoint load, so neither `--checkpoint` nor LoRA; it shards the weights, gradients and state, turns the distributed optimizer off, checks no `CUDA_DEVICE_MAX_CONNECTIONS`, and turns off every data-parallel overlap and `gradient_accumulation_fusion`, so the harness refuses the contract's fused narrow-gradient accumulation under it |
| `--data-parallel-sharding-strategy S` | `no_shard`, `optim`, `optim_grads`, `optim_grads_params` (`no_shard`) | What is sharded: nothing; the optimizer state (ZeRO-1); and the gradients (ZeRO-2); and the weights (ZeRO-3) | `gradient_accumulation_fusion` under Megatron-FSDP needs Transformer Engine 2.10 (the environment's is newer); Bridge warns with the sharded-gradient strategies |
| `--expert-data-parallel-sharding-strategy S` | the same (the dense strategy) | The same for the routed experts' parameters | |
| `--outer-dp-sharding-strategy S` | `no_shard`, `optim` (`no_shard`) | Hybrid sharding: the outer data-parallel group replicates, or shards the optimizer state | |
| `--[no-]fsdp-double-buffer` | (off) | Keeps Megatron-FSDP's communication buffers allocated, double buffered | Turned on by `nccl_ub` |
| `--[no-]nccl-ub` | (off) | Registers the weight and gradient buffers as NCCL user buffers | |
| `--[no-]megatron-fsdp-enable-fine-grained-param-gather` | (off) | Gathers the weights in finer units, overlapping more of the gather with forward | |
| `--megatron-fsdp-grad-comm-dtype D` | `bfloat16`, `float16`, `float32` (the main gradients' dtype) | The dtype gradients are sent at | |
| `--suggested-communication-unit-size N` | elements (Megatron-FSDP's own) | Elements communicated at once, which also paces the prefetch | |
| `--[no-]init-model-with-meta-device` | (off) | Builds the model on meta and materializes only each rank's shards | A sharded strategy |

### Tensor-parallel communication overlap

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]tp-comm-overlap` | (off) | Overlaps the tensor-parallel all-gathers and reduce-scatters with the linear layers' multiplies (Transformer Engine's userbuffers) | Bridge turns it off without tensor parallelism of 2 or more, or without sequence parallelism |
| `--[no-]tp-comm-bulk-wgrad`, `--[no-]tp-comm-bulk-dgrad` | (on), (on) | Overlaps the all-gather with the input-gradient multiply, and the reduce-scatter with the weight-gradient multiply | Read only with `tp_comm_overlap` |
| `--[no-]tp-comm-overlap-ag`, `--[no-]tp-comm-overlap-rs`, `--[no-]tp-comm-overlap-rs-dgrad` | (on), (on), (off) | Pipelines each all-gather or reduce-scatter with its multiply, in chunks | Read only with `tp_comm_overlap` |

Naming `tp_comm_overlap` gives the case Bridge's communication-overlap
configuration, which also derives defaults for the data-parallel and
pipeline overlaps: `overlap_grad_reduce`, `overlap_param_gather` and a
bucket of 128M parameters when there are several data-parallel ranks (only
with the distributed optimizer), `align_param_gather` with virtual stages,
and the pipeline's send and receive scheme. A value the case names is kept
over the derived one. Without `tp_comm_overlap` each is Megatron-Core's
default unless the case names it, or a recipe brings its own
communication-overlap configuration, which derives them the same way; a
case's `tp_comm_overlap` replaces a recipe's configuration.

## Recomputation and activation offloading

### What a layer keeps

A layer's forward keeps what its backward will need: the inputs of its
matrix multiplies, the attention's query, key and value with its output and
statistics, the inputs of its norms and activations. Megatron has four ways
to keep less of that on the device between a layer's forward and its
backward: two recompute, two offload, and each comes in a whole-layer and a
per-module form.

| | Whole layers | Named parts of every layer |
|---|---|---|
| **Recompute** (costs compute, no transfer) | Full recomputation: `--recompute-granularity full` | Selective recomputation: `--recompute-granularity selective --recompute-modules ...` |
| **Offload** (costs a transfer to the host and back, overlapped on a side stream) | Layer-level offloading: `--cpu-offloading` (Transformer Engine's) | Fine-grained offloading: `--fine-grained-activation-offloading --offload-modules ...` (Megatron-Core's) |

With Bridge's Transformer Engine layers ([The model](#the-model)), the norm
before attention is part of `linear_qkv`, and in dense models the norm
before the MLP is part of `linear_fc1`. The norm options act only on a norm
that is a module of its own: the `attn_norm` offload does nothing in these
models, and `layernorm` recomputation and the `mlp_norm` offload do nothing
in a dense model and act on an MoE layer's norm before its router. What a
fused norm keeps is part of what its projection keeps.

### Full recomputation

`--recompute-granularity full` wraps whole layers in a checkpoint. The
forward of a checkpointed layer keeps only its input, the hidden states
entering it, and its backward runs the layer's forward again from that input
before running its backward.

- `--recompute-method uniform --recompute-num-layers N` recomputes every
  layer, in runs of N consecutive layers: each run keeps one input, the one
  entering its first layer.
- `--recompute-method block --recompute-num-layers N` recomputes the first N
  layers of each pipeline stage (of each model chunk, with virtual stages)
  one by one, and the rest keep everything they save.
- `--distribute-saved-activations` splits each kept input across the
  tensor-parallel ranks (not with sequence parallelism, which splits it
  already).
- The shared `--recompute F` is `block` over the first share F of each
  stage's (each chunk's) layers, counted from an even split
  ([Shared settings](#shared-settings)).

A checkpointed layer's input stays on the device: layer-level offloading
refuses recomputation, and the fine-grained offload groups lie inside the
layer, so neither can move it.

### Selective recomputation

`--recompute-granularity selective` recomputes the named parts of every
layer. With no modules named, the harness recomputes `core_attn` and
`layernorm`, and `moe_act` in MoE models (Megatron's own default is
`core_attn` alone); a case that names `--recompute-modules` gets exactly its
list.

`core_attn`, `mlp`, `moe` and `shared_experts` are checkpointed: the module
keeps only its input, and its backward runs it again. The others are
checkpointed with their output discarded: the next module has used the
output by then, Megatron frees it, and the backward recomputes it from the
module's input just before it is needed.

| Module | Recomputes | Goes with |
|---|---|---|
| `core_attn` | The attention itself: the softmax of the scores and the weighted values | Its checkpoint keeps the query and key as rotated and the value as its view into the QKV projection's output, which keeps that whole output alive, while `linear_proj` still keeps the attention's output: beside a fused attention, which keeps its own copy of the value, it keeps more than it frees (Megatron warns that recomputing a fused attention gains little) |
| `layernorm` | A norm that is a module of its own; with Bridge's layers, an MoE layer's norm before its router | Not with delayed FP8 scaling |
| `mlp` | A dense layer's MLP | Dense layers |
| `moe` | The whole MoE layer, router to combine | Not with expert offloads, nor with expert all-to-all overlap |
| `moe_act` | The experts' activation function | `moe_grouped_gemm`; not with delayed FP8 scaling |
| `shared_experts` | The shared experts | Not with `moe_shared_expert_overlap` |
| `mla_up_proj` | Multi-latent attention's up projections and rotary embedding | Multi-latent attention models |
| `gdn_norm_out` | Gated DeltaNet's output norm (Qwen 3.5) | Gated DeltaNet models |
| `gdp_in_proj` | Gated DeltaProduct's input projection, with its context-parallel gather and split | Gated DeltaProduct layers, which no catalog model has: elsewhere it recomputes nothing; not beside a `gdp_qkv` offload |
| `gdp_qkv` | Gated DeltaProduct's causal convolution and query, key and value preparation | Gated DeltaProduct layers: elsewhere it recomputes nothing |
| `mhc` | Hyper-connections' intermediate values | `enable_mhc_connections`, which no catalog model has, so Megatron refuses it for these models; not with `mlp`, nor with fine-grained offloading |
| `shortcut_pre_mlp_layernorm` | A shortcut-connected MoE's norm before its shortcut router | `moe_shortcut_connection`, which no catalog model has, so Megatron refuses it for these models; not with delayed FP8 scaling |
| `residual_stream` | A wide residual stream's reads, connected norms and writes, replayed | A wide residual stream (`wide_residual`), which no catalog model has, so Megatron refuses it for these models; no graph capture |

### Layer-level offloading

`--cpu-offloading` is Transformer Engine's. Each of the first
`--cpu-offloading-num-layers` layers runs inside an offload context: every
value the layer keeps for its backward, the layer's own input among them, is
copied to pinned host memory on a side stream as the forward goes, and
copied back before that layer's backward. Parameters, values under 256K
elements and non-contiguous values stay on the device, and so does the last
layer, whose backward comes first. `--cpu-offloading-retain-pinned-cpu-buffers`
keeps the host buffers across updates instead of allocating them every
update. Megatron refuses it beside any recomputation, pipeline parallelism,
per-layer graphs and fine-grained offloading.

### Fine-grained offloading

`--fine-grained-activation-offloading` is Megatron-Core's. It moves to pinned
host memory everything the named parts of every layer keep for their
backward (`--offload-modules`, at least one), on a side stream as the
forward reaches the end of each part, and brings it back before that part's
backward. Parameters, the values Transformer Engine marks as weights, and
values under `--min-offloaded-tensor-size` elements (1M by default) stay.
The first update is a warmup that records each microbatch's groups (each
model chunk's, with virtual stages). From the second on, a microbatch whose
backward runs next keeps the last group of each name on the device, so that
its reload never waits: at one pipeline rank, where each microbatch's
backward follows its forward, every microbatch keeps its last groups; under
pipelining, only one that is next in the backward queue while its forward
runs. `--activation-offload-fraction` moves only the first share of each
microbatch's eligible groups in the order the forward meets them (groups,
not bytes), and `--delta-offload-bytes-across-pp-ranks` keeps more on the
device on each later pipeline stage: rank r keeps about r times the value
in bytes, one budget spent group by group on the microbatches whose
backward comes first.

| Module | Moves | Goes with |
|---|---|---|
| `attn_norm` | What a separate norm before attention keeps; with Bridge's layers that norm is part of `linear_qkv`, whose kept values `qkv_linear` moves, so this moves nothing | |
| `qkv_linear` | What the query, key and value projection keeps (with the norm fused into it, the layer's input among it) | |
| `core_attn` | What the attention keeps: its query, key and value, its output and statistics | |
| `attn_proj` | What the attention's output projection keeps: its input, `core_attn`'s output | Megatron wants `core_attn` named with it, since `core_attn`'s backward needs that output too |
| `mlp_norm` | What a separate norm before the MLP keeps: in an MoE layer, its input, the residual stream after attention; in a dense layer the norm is part of `linear_fc1` and nothing moves | |
| `expert_fc1` | The input of the experts' first projection, the tokens dispatched to them | MoE models |
| `moe_act` | The input of the experts' activation function, their first projection's output | MoE models |
| `fused_group_mlp` | Everything the experts' fused grouped MLP keeps | MoE models; `use_transformer_engine_op_fuser`; not with `expert_fc1` or `moe_act` |
| `gdp_qkv` | What Gated DeltaProduct's causal convolution and query, key and value preparation keep | Gated DeltaProduct layers, which no catalog model has: elsewhere it moves nothing; not beside `gdp_in_proj` recomputation |
| `shortcut_post_norm` | What a shortcut-connected MoE's output norm keeps | `moe_shortcut_connection` with `moe_shortcut_post_norm`, which no catalog model has, so Megatron refuses it for these models |

No module offloads a dense layer's MLP; selective `mlp` recomputation is the
way to keep less of it.

No group dedupes: every value saved inside a group gets its own tag and its
own copy, so a value saved twice moves twice, by two groups or within one
(Megatron only warns, after the warmup, of storage one group copied more
than once). The attention's output, which the attention and `linear_proj`
both keep, moves with `core_attn` and again with `attn_proj`. A
non-contiguous view that covers at least half of its storage moves as the
whole storage (a plain copy and the view's metadata), so the experts'
activation, which keeps one half of `linear_fc1`'s output, moves all of it;
a smaller non-contiguous view is gathered into a copy of its own elements,
and a contiguous view moves just its own elements. A group's commit also
releases values it names outright, when the group is offloaded and the
value passes the size and mark checks: the norm groups the residual once
the residual add has used it, `core_attn` the query, key and value,
`attn_proj` the attention's output, `expert_fc1` the dispatched tokens and
`moe_act` `linear_fc1`'s output. A group that the last-group rule, the
fraction or the delta keeps on the device releases nothing.

### Recomputing and offloading together

Used together, recomputation takes the parts that are cheap to run again
and offloading the ones that are expensive: a part recomputed costs a second
forward of that part, a part offloaded a transfer overlapped with compute.
Megatron's guide lists "fine-grained recomputation" as compatible with
fine-grained offloading: it means selective recomputation, on at the same
time on other parts of the same layer, and the guide pairs them that way,
the norms and the experts' activation recomputed, the attention and the
experts' input offloaded.

What a recomputed part needs is its input, and whether that input can be
offloaded too depends on whether an offload group is open when the part's
checkpoint keeps it:

| Recomputed | The input its recomputation needs | Offloadable |
|---|---|---|
| `layernorm` (an MoE layer's pre-router norm) | The residual stream after attention | Yes, with `mlp_norm`: the norm's checkpoint runs inside that group, so the input goes to the host and comes back before the norm is recomputed |
| `moe_act` | The experts' first projection's output | Yes, with `moe_act`, whose group wraps the activation's checkpoint |
| `core_attn` | Query, key and value | No: a recomputed attention runs outside its offload group, so naming `core_attn` for both moves nothing of it, with no warning (it still has to be named beside `attn_proj`, whose own input does move) |
| `mlp` (dense) | The residual stream after attention | No: no group wraps a dense MLP |
| `moe`, `shared_experts` | The pre-router norm's output | No: the checkpoint takes it after the norm's group has closed. With `layernorm` recomputed too, that output is not kept at all: it is discarded after use and recomputed from the norm's input, which `mlp_norm` can offload |
| A whole layer (full recomputation) | The layer's input | No |

So in Megatron a recomputed region's input reaches the host only at module
granularity, and only for an MoE layer's pre-router norm and the experts'
activation. No rule in Megatron or Bridge refuses fine-grained offloading
beside full recomputation, though a recomputed layer's forward keeps nothing
for the groups; during the warmup, the groups its recomputation enters again
in the backward are offloaded and reloaded at once. The FSDP2 and DeepSpeed
harnesses offload a recomputed block's input whole (`--activation-offload`);
Megatron has no counterpart.

Recomputation and offloading both choose layers only by position: `block`
recomputation and layer-level offloading take the first layers of a stage,
`uniform` and selective recomputation and fine-grained offloading every
layer (fine-grained offloading's fraction then leaves the later groups on
the device). At one pipeline rank both apply the same way to every
microbatch; under pipelining, fine-grained offloading's last-group rule and
its delta budget fall on particular microbatches
([Fine-grained offloading](#fine-grained-offloading)). No option names
particular layers, or particular microbatches: Megatron-Core's
`num_microbatches_with_partial_activation_checkpoints` would hand a
per-microbatch flag to the forward step, but nothing in Megatron-Core's
layers reads it and Bridge's forward step has no parameter for it, so the
harness does not offer it.

Examples, added to a case's flags:

```bash
# MoE, Megatron's pairing: the pre-router norm and the experts' activation
# recomputed, the attention and the experts' input offloaded.
--recompute-granularity selective --recompute-modules layernorm,moe_act \
  --fine-grained-activation-offloading --offload-modules core_attn,attn_proj,expert_fc1

# MoE, with the recomputed parts' inputs offloaded as well.
--recompute-granularity selective --recompute-modules layernorm,moe_act \
  --fine-grained-activation-offloading \
  --offload-modules core_attn,attn_proj,mlp_norm,expert_fc1,moe_act

# Dense: the MLP recomputed, the attention's values offloaded.
--recompute-granularity selective --recompute-modules mlp \
  --fine-grained-activation-offloading --offload-modules qkv_linear,core_attn,attn_proj

# Every layer recomputed from its input, two layers to a checkpoint.
--recompute-granularity full --recompute-method uniform --recompute-num-layers 2

# The first four layers of each stage recomputed, the rest kept whole.
--recompute-granularity full --recompute-method block --recompute-num-layers 4

# Everything the first seven layers keep moved to the host (no
# recomputation, no pipeline parallelism).
--cpu-offloading --cpu-offloading-num-layers 7
```

### What goes together

| | Full recomputation | Selective recomputation | Layer-level offloading | Fine-grained offloading |
|---|---|---|---|---|
| **Full recomputation** | | One granularity per case | Refused [1] | No rule in Megatron: a recomputed layer's forward keeps nothing for the groups; layers outside block recomputation offload as usual |
| **Selective recomputation** | One granularity per case | | Refused [1] | Complementary, as above; recomputing `moe` refuses offloading `expert_fc1`, `moe_act` or `fused_group_mlp`, `mhc` refuses fine-grained offloading, and `gdp_in_proj` a `gdp_qkv` offload [2] |
| **Layer-level offloading** | Refused [1] | Refused [1] | | Refused [3] |
| **Pipeline parallelism** | Per stage (each model chunk's layers) | Yes | Refused [4] | Yes; `delta_offload_bytes_across_pp_ranks` balances the stages |
| **Graph capture** | Full-iteration graphs only [5] | Per-layer or full-iteration graphs, without dropout or MoE input jitter inside a graphed recomputation, and without `moe` recomputation beside Transformer Engine's graph of the router [5] | Full-iteration graphs only [6] | See below |
| **FP8** | Yes | Delayed scaling refuses `moe_act`, `layernorm` and `shortcut_pre_mlp_layernorm` [7] | No rule | Yes |
| **Expert all-to-all overlap** | Refused [8] | Not `moe` [8] | No rule | Yes; not `expert_fc1` with the `ncclep` dispatcher [9] |

Fine-grained offloading under graph capture takes Transformer Engine's
per-layer graphs; Megatron-Core's local graphs only when they capture parts
of a layer and the offloaded modules are the experts' (`expert_fc1`,
`moe_act`, `fused_group_mlp`); and full-iteration graphs only with
`fine_grained_offloading_max_inflight_offloads` set [10]. It never goes
beside a graph of the whole MoE, and it needs at least one graph warmup
update. Megatron turns off, with a warning, `attn_norm` offloading beside a
graph of `attn`, and `mlp_norm` offloading where that norm is a graph's
boundary (in an MoE layer, beside a graph of `attn` or `moe_router`).
`delay_offload_until_cuda_graph` queues the experts' offloads (`expert_fc1`,
`moe_act`) and issues the queue after each graphed layer's replay, before
that layer's eager experts run, so a layer's offloads go out after the next
graphed layer's replay; the queue the update's last graphed layer leaves is
dropped when the offload state resets at the end of the update, and those
groups stay on the device.

[1] "CPU offloading does not work when activation recomputation is enabled"
(Megatron-Core `transformer/transformer_config.py:2527`).
[2] "Cannot offload ... while recomputing the entire MoE layer" (`:2845`),
and the `mhc` and `gdp_qkv` rules (`:2684`, `:2834`).
[3] "fine_grained_activation_offloading cannot be enabled with cpu_offloading"
(`:2806`).
[4] "Currently there is no support for Pipeline parallelism with CPU
offloading" (`:2522`).
[5] Full recomputation only beside full-iteration graphs, and the dropout,
jitter and router rules beside it (`:3656-3708`).
[6] Graphs refused beside layer-level offloading, full-iteration ones aside
(`:3590`).
[7] "Delayed scaling does not support moe_act, layernorm, or
shortcut_pre_mlp_layernorm recompute for fp8" (`:2633`).
[8] `overlap_moe_expert_parallel_comm` refuses full recomputation, a recompute
method or layer count, and `moe` recomputation (`:3825-3836`).
[9] `:3854-3865`. [10] `:3731-3777`.

### Their options

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--recompute-granularity G` | `full`, `selective` (none: nothing recomputed) | Full or selective recomputation, as above | One per case; not beside the shared `--recompute`, which sets `full` itself |
| `--recompute-method M` | `uniform`, `block` (none) | How full recomputation chooses its layers | Full recomputation needs it; not beside the shared `--recompute` |
| `--recompute-num-layers N` | integer (none) | The run length (`uniform`) or the layer count (`block`) | Full recomputation needs it and selective takes none; Megatron checks only that it is set, and the harness refuses a value below 1, with which `uniform` never ends; not beside the shared `--recompute` |
| `--recompute-modules LIST` | `core_attn`, `layernorm`, `mlp`, `moe`, `moe_act`, `shared_experts`, `mla_up_proj`, `gdn_norm_out`, `gdp_in_proj`, `gdp_qkv`, `mhc`, `shortcut_pre_mlp_layernorm`, `residual_stream` (`core_attn,layernorm`, and `moe_act` in MoE models) | The parts selective recomputation recomputes | Read only with `selective`; not beside the shared `--recompute` |
| `--[no-]distribute-saved-activations` | (off) | Splits each kept input across the tensor-parallel ranks | Not with sequence parallelism |
| `--[no-]cpu-offloading` | (off) | Layer-level offloading | Not with recomputation, pipeline parallelism, per-layer graphs or fine-grained offloading |
| `--cpu-offloading-num-layers N` | 0 to one fewer than the model's layers (0) | The layers offloaded, the first ones | 0 moves nothing |
| `--[no-]cpu-offloading-retain-pinned-cpu-buffers` | (off) | Reuses the host buffers across updates | The values' shapes must not change between updates |
| `--[no-]fine-grained-activation-offloading` | (off) | Fine-grained offloading | Not with `cpu_offloading`; Transformer Engine's layers (Bridge's default) |
| `--offload-modules LIST` | `attn_norm`, `qkv_linear`, `core_attn`, `attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`, `fused_group_mlp`, `gdp_qkv`, `shortcut_post_norm` (none) | The parts whose kept values move | At least one; `attn_proj` with `core_attn` |
| `--min-offloaded-tensor-size N` | elements (1048576) | The smallest value moved | |
| `--activation-offload-fraction F` | 0 to 1 (1) | The share of each microbatch's eligible groups moved, the first ones | |
| `--delta-offload-bytes-across-pp-ranks N` | bytes (0) | Bytes more kept on the device per later pipeline stage, which holds fewer microbatches in flight: rank r keeps about r x N, one budget spent on the microbatches whose backward comes first | |
| `--fine-grained-offloading-max-inflight-offloads N` | integer (none: no waits) | Committed offload groups of one name whose copies may still be in flight before the compute stream waits for the oldest (0: after each commit; none: never) | Required with full-iteration graphs |
| `--[no-]delay-offload-until-cuda-graph` | (off) | Queues the experts' offloads and issues them after the next graphed layer's replay, as above | Transformer Engine's per-layer graphs |

Megatron reads a mechanism's options only with the mechanism on:
`cpu_offloading_num_layers` without `cpu_offloading`, or the fine-grained
options without fine-grained offloading, change nothing.

## Optimizer offloading

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]optimizer-cpu-offload` | (off) | Keeps a share of the optimizer state, its float32 masters and their step on the host: Megatron-Core's hybrid device optimizer steps the host's share with PyTorch's AdamW and the rest on the device with Transformer Engine's fused Adam | AdamW only; Megatron-LM's command line wants the precision-aware optimizer with it, which Bridge does not check |
| `--optimizer-offload-fraction F` | 0 to 1 (0) | The share moved, taken parameter by parameter in their group order | Megatron's default, 0, moves nothing even with the offload on |
| `--[no-]overlap-cpu-optimizer-d2h-h2d` | (off) | Overlaps the host optimizer's transfers with its step | |
| `--[no-]pin-cpu-grads` | (on) | Pins the host copies of the gradients the host optimizer reads | |
| `--[no-]pin-cpu-params` | (on) | Pins the float32 copies the host optimizer steps when it receives bfloat16 shards, as on the precision-aware path; with float32 shards it steps their pinned host clones, and the option does nothing | |

Megatron offloads no parameters to the host: the weights and the gradient
buffer stay on the device, so a model larger than one device trains only
sharded across ranks.

## Mixture of experts

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--moe-token-dispatcher-type T` | `allgather`, `alltoall`, `flex` (`alltoall` for Qwen 3 MoE) | How tokens reach their experts: every rank gathers every token, each token goes to its experts' ranks, or a dedicated dispatcher | `flex` names its backend |
| `--moe-flex-dispatcher-backend B` | `deepep`, `deepepv2`, `hybridep`, `ncclep` (`deepep`, read only with `flex`) | The flex dispatcher: DeepEP (compute capability 8 and 9, B200 and B300), DeepEP's version-2 interface, HybridEP (8 to 10) or Transformer Engine's NCCL EP (9 and 10) | Bridge checks the GPU and refuses `deepepv2`; none takes 12.0. DeepEP, HybridEP and NCCL EP are built by the setup where 9.0 or 10.0 is a target; elsewhere Megatron refuses them as it builds the dispatcher |
| `--[no-]moe-grouped-gemm` | (on for Qwen 3 MoE) | Runs a rank's experts as one grouped multiply | |
| `--[no-]moe-use-grouped-tensor` | (off) | Transformer Engine's grouped-tensor path, which graphs can capture | `moe_grouped_gemm`; the operation fuser turns it on |
| `--[no-]moe-permute-fusion` | (on for Qwen 3 MoE) | Fuses the token permutation of dispatch and combine | |
| `--[no-]moe-router-fusion` | (off) | Fuses the router's top-k and its balancing loss | |
| `--[no-]moe-shared-expert-overlap` | (off) | Overlaps the shared experts with the dispatcher's communication | `alltoall` or `flex`; not with `shared_experts` recomputation |
| `--[no-]overlap-moe-expert-parallel-comm` | (off) | Overlaps the expert all-to-alls with another microbatch's compute | Expert parallelism, `alltoall` or `flex`, a bfloat16 model, no full or `moe` recomputation, no `moe_shared_expert_overlap`, virtual stages under pipeline parallelism |
| `--[no-]delay-wgrad-compute` | (off) | Delays the experts' weight-gradient multiplies to overlap more communication; set on the model and on DDP, whose copy moves those weights' gradient reduction to their delayed multiplies | `overlap_moe_expert_parallel_comm` |
| `--moe-router-dtype D` | `fp32`, `fp64` (none: the model dtype) | The router's and the expert outputs' weighted sum's dtype | Megatron warns without it for 32 or more experts |

## Attention, fusions and kernels

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--attention-backend B` | `auto`, `flash`, `fused`, `unfused`, `local` (`auto`) | Transformer Engine's attention (its own choice, FlashAttention, cuDNN's fused attention, unfused), or Megatron-Core's PyTorch attention | Megatron sets Transformer Engine's attention variables from it, and refuses a process whose variables disagree |
| `--flash-attention-version V` | `2`, `3`, `4` (none: Transformer Engine's choice) | The FlashAttention generation | |
| `--[no-]apply-rope-fusion` | (on for Llama 3, off for Qwen 3 MoE) | The fused rotary-embedding kernel | Bridge turns it off without rotary embeddings |
| `--[no-]bias-activation-fusion` | (on for Llama 3, off for Qwen 3 MoE) | Fuses the bias with the activation | Not with `use_te_activation_func`; GELU, SwiGLU or quick GEGLU |
| `--[no-]bias-dropout-fusion` | (on for Llama 3, off for Qwen 3 MoE) | Fuses the bias, dropout and residual add | |
| `--[no-]masked-softmax-fusion` | (on) | The fused masked softmax | |
| `--[no-]persist-layer-norm` | (on for Llama 3, off for Qwen 3 MoE) | The persistent fused layer-norm kernel | |
| `--[no-]fused-residual-rmsnorm` | (off) | Fuses the residual add into RMSNorm | RMSNorm models |
| `--[no-]cross-entropy-loss-fusion`, `--cross-entropy-fusion-impl I` | (on); `native`, `te` (`native`) | The fused vocabulary-parallel cross entropy over the logits, Megatron-Core's or Transformer Engine's | The implementation is read only with `cross_entropy_loss_fusion` |
| `--[no-]gradient-accumulation-fusion` | (on where Transformer Engine imports) | As in [Gradient accumulation](#gradient-accumulation) | Megatron-Core's own output layer needs APEX's `fused_weight_gradient_mlp_cuda` with it, which the environment installs; Megatron-FSDP version 2 turns it off |
| `--[no-]use-transformer-engine-op-fuser` | (off) | Transformer Engine's operation fuser in the parts that take it | |
| `--[no-]use-te-activation-func` | (off) | Transformer Engine's activation functions in the MLP | |
| `--gdn-kernel-backend B` | `fla`, `transformer_engine`, `torch` (`fla`) | Gated DeltaNet's kernels (Qwen 3.5) | |
| `--[no-]disable-parameter-transpose-cache` | (off) | Does not keep the weights' transposes between updates | |

**The output layer and the loss.** Megatron-Core computes the whole logits,
then the cross entropy over them, fused (`cross_entropy_loss_fusion`) or not;
at this revision it has no projection and loss computed in chunks, nor fused
into one kernel. Unfused, or fused by Megatron-Core's own implementation
(`native`), the loss keeps the softmax at float32 for its backward, a buffer
of a microbatch's tokens times the vocabulary, with the bfloat16 logits
alive beside it while the loss is computed. Transformer Engine's (`te`)
keeps the bfloat16 logits buffer itself, which its backward overwrites with
the gradient, and two float32 statistics per token. Either way, at long
sequences what the loss keeps is the largest thing a step keeps that no
option recomputes or offloads. Under tensor parallelism the output layer
keeps its logits sharded by vocabulary (Bridge's `parallel_output`) and the
loss runs on the shards across the tensor-parallel group, so each rank holds
its share; on one device only a smaller microbatch makes it smaller.

## Graph capture

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--cuda-graph-impl I` | `none`, `local`, `transformer_engine`, `full_iteration` (`none`) | Megatron-Core's per-layer graphs, Transformer Engine's per-layer graphs, or forward and backward as one graph (the optimizer outside) | Bridge's local graphs capture parts of a layer (`cuda_graph_modules`) |
| `--cuda-graph-modules LIST` | `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba` (none: the whole layer) | With per-layer graphs, the parts of a layer captured | None with `full_iteration`; `moe` and `moe_router` not together; `moe_preprocess` with `moe_router`; `mamba` for Mamba layers, of a family no catalog model is; `mlp` for dense layers, `moe` for drop-and-pad MoE layers only (an expert capacity factor with `moe_pad_expert_input_to_capacity`, which no harness option sets) |
| `--cuda-graph-warmup-steps N` | updates (3) | N, which places the capture, as below | Fewer than `--warmup-steps` with graphs or `optimizer_cuda_graph` |

Bridge requires Transformer Engine's random-number tracker under graph capture
and does not turn it on; the harness turns it on whenever graphs are on (as
Megatron-LM's command line does). Where the graphs are captured depends on
the implementation, N being `cuda_graph_warmup_steps` and the updates
counted from 0: Transformer Engine's per-layer graphs at the top of update
N, before the step's callbacks start its clock; full-iteration graphs inside
update N; Megatron-Core's local graphs at the end of update 0's forward and
backward, whatever N is, each after N eager warmup runs; and the optimizer's
graph (`optimizer_cuda_graph`) inside update N's optimizer step. The
harness refuses a case with graphs or the optimizer's graph whose N is not
one of its warmup updates.

## The process environment

Two switches are read only from the process environment, so the case
decides them: the launcher sets them for the trainer, and unsets them where
the case says none, whatever the shell had. `case.json` and each rank's
request record them, and a trainer started in another environment refuses
to run.

| Variable | Set from | Why |
|---|---|---|
| `NVTE_CPU_OFFLOAD_V1` | `1` with fine-grained offloading, unset otherwise | Transformer Engine chooses its offload implementation once, as it is imported. Megatron's fine-grained offloading tells Transformer Engine that offloading is on through the version-1 implementation's flag, which Transformer Engine reads only with this variable set; Transformer Engine then marks its weights (and its quantized copies of them) to stay on the device. Bridge refuses fine-grained offloading without it. Unset, layer-level offloading runs Transformer Engine's default implementation |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `--cuda-device-max-connections N` (1 to 32), unset otherwise (the runtime's 8) | The device runtime reads it as it starts. 1 keeps tensor-parallel communication ahead of the multiplies it overlaps, as Megatron advises with sequence parallelism; Megatron-FSDP version 1 refuses 1; Megatron-Core's local graphs warn when it is unset or below 16, advising 16 or 32 for GTP workloads |

The trainer sets one more variable itself: `CUDA_HOME`, to the environment's
own toolkit ([Running a case](#running-a-case)) where it exists, whatever
the shell had, so that a kernel compiled when first used (NCCL EP's
dispatch, with the `ncclep` backend) compiles with the toolkit the
environment's builds used.

## Data

The tokens are the ones every harness trains on: synthetic tokens drawn once
per replica from `--seed`, or each replica's window of the corpus for each
update. Megatron's sampler asks for an update's samples by index, each
microbatch `micro_batch_size x replicas` consecutive indices with each
data-parallel rank serving its slice; the dataset provider maps an index to
its replica, update and position and serves that sequence. Bridge's
tokenizer is a null tokenizer of the catalog vocabulary, and evaluation,
checkpoint saving and logging inside the measured run are off.

## Measurement

Bridge's loop runs the case, and its step callbacks drive the runner's
clock: each update starts once every rank's device is idle and the ranks
have met at a barrier, and ends when this rank's device is idle, with no
barrier at the end; the warmup updates are excluded, and the result is the
median over the measured ones of the slowest rank's time. The learning rate
is constant and gradients are not clipped. Megatron's own checks of the
gradients and of each microbatch's loss for NaN are off: the runner checks
every measured loss, the other harnesses add no such host synchronization,
and a full-iteration graph refuses the loss check. Peaks are PyTorch
allocator measurements of the binding rank. The row's useful TFLOP/s per GPU
is counted from the catalog model ([The model](#the-model)).

## What a run writes

```text
output/
  case.json                 # the case the launcher validated and resolved, and its environment
  rank-NNNNN/request.json   # the case as described, the resolved precision, the corpus, and the
                            # configuration under Megatron's names, as handed to Bridge and, once
                            # training starts, as Bridge finalized it: the provider, its
                            # architecture, the precision fields, every option, the environment
  rank-NNNNN/result.json    # cycles, losses, peaks, precision evidence; or failure.json
  results.json              # every rank's measurements (first rank)
  row.json                  # the shared result row (first rank), with the parameters and the
                            # trainable parameters of the model Megatron built
```

## Hardware limits

On compute capability 12.0 (the RTX 5090) there is no MXFP8 and no flex
dispatcher backend. FP8 needs compute capability 8.9 and NVFP4 10.0, so
neither runs on Ampere.
