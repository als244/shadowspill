# NeMo AutoModel over the catalog models

This engine trains NVIDIA's implementation of a catalog model's
architecture with NeMo AutoModel's fine-tuning recipe
(`TrainFinetuneRecipeForNextTokenPrediction`) and AutoModel's own training
loop, under the shared objective, geometry, precision contract and result
row of `baselines/common`. AutoModel builds its own class of the
architecture where its registry has one, and Hugging Face's otherwise; the
tokens are the ones every engine trains on, through AutoModel's own data
path.

AutoModel's options are settings under AutoModel's own names, each left to
AutoModel's default unless the case names it. The engine sets two defaults
of its own: the loss, and the linear layers beside it
([The loss](#the-loss)). It refuses only what it cannot say or what would
train another objective: a value of the wrong type or outside AutoModel's
choices, a precision the contract asks for that AutoModel's options cannot
express, the shared settings where they contradict AutoModel's own, a
mixture AutoModel builds from Hugging Face's classes, and a Qwen 3.5 or
Qwen 3.5 MoE checkpoint, whose class adds a multi-token-prediction loss
([The model](#the-model)). What goes together in AutoModel is AutoModel's to
decide as it sets up; a case it refuses fails there, before training, with
its message. This page gathers its rules.

**Contents.** [Running a case](#running-a-case) · [The model](#the-model) ·
[Shared settings](#shared-settings) · [Precision](#precision) ·
[The loss](#the-loss) · [Attention](#attention) ·
[The backend](#the-backend) · [Distribution](#distribution) ·
[Activation checkpointing and offloading](#activation-checkpointing-and-offloading) ·
[FP8 and compilation](#fp8-and-compilation) · [LoRA](#lora) ·
[AutoModel's benchmark recipes](#automodels-benchmark-recipes) ·
[Data](#data) · [Measurement](#measurement) ·
[What a run writes](#what-a-run-writes) · [Limits](#limits)

The [options catalog](options/) groups declarations by parallelism, model and
training. [settings.py](settings.py) validates and translates them; adding an
option does not require editing the training loop.

## Running a case

The trainer runs in the `nvidia-training` environment, built in one command
by `bash baselines/scripts/setup_nvidia_training.sh`
(`../megatron_core/README.md` says what it installs for Megatron). For
AutoModel it adds the kernels AutoModel's defaults and Transformer Engine
take where they import:

| Package | Source | What takes it |
|---|---|---|
| NeMo AutoModel | its checkout at the pinned revision | the recipe and its models |
| Cut Cross-Entropy, Liger | AutoModel's pinned revisions | `--loss fused_linear` (Cut Cross-Entropy); Liger's kernels in classes built from Hugging Face's |
| causal-conv1d (1.7.0) | a third-party prebuilt wheel of its v1.7.0 source for PyTorch 2.13 and CUDA 13.0, x86_64 and aarch64 (astral-sh-build/build-causal-conv1d) checked against its digest, compiled for the architectures its `setup.py` lists; Dao-AILab publishes none for PyTorch 2.13, and another platform builds it from that source | Qwen 3.5's gated delta net, whose convolution otherwise runs in PyTorch |
| FlashAttention 2 (2.8.3) | a third-party prebuilt wheel for x86_64 (mjun0812/flash-attention-prebuild-wheels v0.9.47) checked against its digest; neither Dao-AILab nor PyTorch publishes one for PyTorch 2.13 | AutoModel's default `attn_implementation`; Transformer Engine's attention below compute capability 9.0, and from 9.0 where cuDNN's fused attention cannot run (at 9.0 itself, where FlashAttention 3 cannot either) |
| FlashAttention 3 | PyTorch's `cu130` wheel index, as mlops installs it, on every x86_64 and aarch64 machine whatever its GPU | `--attn-implementation flash_attention_3`, and Transformer Engine's attention where cuDNN's cannot run, both at compute capability 9.0 only |

A platform without a pinned wheel keeps PyTorch's and cuDNN's attention;
the setup says so, and the lock file records what it installed.

One case, through the launcher every engine starts from:

```bash
python -m baselines.common.launch --engine automodel --gpus 1 \
  --output <dir> \
  --model llama3-tiny --sequence-length 64 --tokens-per-step 1024 \
  --tokens-per-microbatch 128 --warmup-steps 1 --steps 2
```

The tiny qualification matrix runs the one-rank precision mechanisms, the
chunked, masked and fused-linear losses, the three checkpointing modes,
FlashAttention 2 and Transformer Engine's attention, and the LoRA selections
on the tiny models, every case's first update checking its dtypes against
the contract (there is no reference comparison, since the model is
NVIDIA's); its checkpoint LoRA cases adapt the tiny checkpoint that
`python -m baselines.scripts.write_checkpoint --model llama3-tiny --output runs/checkpoints/llama3-tiny`
writes:

```bash
python -m baselines.sweep --engine automodel \
  --config baselines/engines/automodel/configs/qualification.json \
  --output <dir> --gpus 1
```

## The model

A case names a catalog model. Its Hugging Face configuration comes from the
named checkpoint's `config.json` (AutoModel's `from_pretrained`, importing
its weights), or from the family's `to_huggingface` for the catalog entry with
the case's `--model-config`, written to the rank's `huggingface/config.json`
(AutoModel's `from_config`, AutoModel's own initialization under `--seed`).
AutoModel looks the configuration's architecture up in its registry and
builds its own class where it has one; `--force-hf` takes Hugging Face's for
every family:

| Family | The class AutoModel builds | Its head | Its layers |
|---|---|---|---|
| Llama 3 | AutoModel's (`components/models/llama`) | `nn.Linear` | `nn.Linear` projections and MLP; norms by `--rms-norm`; Hugging Face's attention interface |
| Qwen 3 | AutoModel's (`components/models/qwen3`), a subclass of Hugging Face's attention and decoder layer adding a packed Transformer Engine path | `nn.Linear` | Hugging Face's linears and norms; Hugging Face's attention interface |
| Qwen 3 MoE | AutoModel's (`components/models/qwen3_moe`) | by `--linear` | projections and any dense MLP by `--linear`, attention by `--attn`, norms by `--rms-norm`, experts by `--experts`, tokens by `--dispatcher` |
| Qwen 3.5 | AutoModel's (`components/models/qwen3_5`), on Qwen3-Next's block, from a configuration; a checkpoint is refused | `nn.Linear` | full-attention projections and MLP by `--linear`, attention by `--attn`, its own zero-centered norms; the gated delta net keeps Hugging Face's `nn.Linear` projections and runs FLA's gated delta rule |
| Qwen 3.5 MoE | AutoModel's (`components/models/qwen3_5_moe`), from a configuration; a checkpoint is refused | by `--linear` | as Qwen 3.5, with experts, a dispatcher, and a shared expert and its gate by `--linear` |
| OLMoE | refused: AutoModel registers no class and would build Hugging Face's | | |

A mixture AutoModel builds from Hugging Face's classes (OLMoE, or any
mixture under `--force-hf`) is refused: AutoModel trains no balancing loss
there (the recipe takes the labels before the forward, and Hugging Face adds
the term only to a loss it computes from labels, with `output_router_logits`
off by default) and its LoRA does not adapt the fused experts, both of which
the shared objective has. A Qwen 3.5 or Qwen 3.5 MoE case from a checkpoint
is refused as training another objective too: the published checkpoints
carry vision-language configurations, from which AutoModel builds its
vision-language class with a multi-token-prediction layer, whose loss joins
the objective, while the shared objective is the text model's. From a
configuration both build AutoModel's text classes and run. Qwen 3.5 has a
Hugging Face counterpart only with zero-centered norms. The backend options
act only on AutoModel's own classes; a case that names one for a class
AutoModel builds from Hugging Face's is refused, since Hugging Face's classes
take no backend.

The row names the implementation `automodel`. Its useful work is counted from
the catalog's own model of the same architecture with the convention every
engine uses, so recomputation, the optimizer and communication are not
credited.

## Shared settings

| Flags | Values (default) | Meaning |
|---|---|---|
| `--model`, `--model-config JSON` | a catalog name; field overrides | The catalog model and overrides of its configuration's fields |
| `--implementation` | `automodel` | NVIDIA's implementation, the only one this engine trains |
| `--mode`, `--lora JSON` | `full` or `lora` (`full`); LoRA options | Full or LoRA training ([LoRA](#lora)) |
| `--checkpoint DIR` | a directory (none) | A Hugging Face checkpoint to import |
| `--sequence-length`, `--tokens-per-step`, `--tokens-per-microbatch` | tokens | One update's geometry in whole sequences; a replica is tensor x pipeline x context parallel ranks |
| `--model-dtype`, `--master-dtype`, `--grad-dtype`, `--opt-state-dtype`, `--parameter-rounding`, `--opt-state-rounding`, `--[no-]fuse-narrow-gradient-accumulation` | the contract's; fused accumulation off | The precision contract ([Precision](#precision)) |
| `--data DIR` | a corpus (synthetic tokens) | A corpus `python -m baselines.scripts.prepare_corpus` wrote |
| `--recompute F` | 0 or 1 (none) | AutoModel checkpoints every decoder layer or none: 1 is full activation checkpointing, 0 is none, and none leaves it to `--activation-checkpointing` |
| `--warmup-steps`, `--steps` | updates (3, 5) | Excluded warmups and measured updates |
| `--learning-rate`, `--weight-decay`, `--seed` | (3e-4, 0.01, 0) | Adam's rate (constant) and decay; betas 0.9 and 0.999, epsilon 1e-8, no clipping |

`--verify-reference` is refused: the reference check compares updates of the
catalog's own implementation, and this engine trains NVIDIA's.

Each AutoModel option below is `--<name-with-dashes> VALUE`, a boolean
`--<name>` or `--no-<name>`, a list comma-separated. Each goes into the
recipe section AutoModel reads it from under AutoModel's key; the default in
each table is AutoModel's own when the case names nothing, unless the table
says it is the engine's, and each rank's request records the whole recipe
configuration the run used.

## Precision

[`../../PRECISION.md`](../../PRECISION.md) sets this engine beside the others.

### What AutoModel offers

| | AutoModel's default | AutoModel's other options |
|---|---|---|
| Weights' storage | bfloat16 for AutoModel's own classes on both paths; for Hugging Face's, the checkpoint's dtype (`from_pretrained`) or bfloat16 (`from_config`) | `torch_dtype` |
| Compute and the gradient reduction | FSDP2's `mp_policy`: compute and gathers at bfloat16, gradients reduce-scattered at float32 and stored at the weights' dtype. Only where FSDP2 shards: on one rank AutoModel parallelizes nothing, so the model computes at its storage dtype, except that every frozen parameter outside the model's `_keep_in_fp32_modules`, and the trainable parameters of every module that also owns frozen ones (a LoRA module's adapters), are cast to `mp_policy.param_dtype` | `param_dtype`, `reduce_dtype`, `output_dtype`; float32 storage with bfloat16 compute is AutoModel's float32-master-weights pattern, on two or more ranks |
| One rank under DDP or Megatron-FSDP | The model is cast to bfloat16 | |
| Megatron-FSDP's main weights and gradients | The weights' own, with no separate main copies | `preserve_fp32_weights` (float32 main weights), `grad_reduce_in_fp32` (float32 main gradients) |
| The optimizer | A torch optimizer stepping the stored weights: no masters, moments at the weights' dtype; AutoModel's examples most often use torch's Adam (241 of their 551 optimizer sections), then AdamW (200), unfused | torch's `AdamW` (`fused`); Transformer Engine's `FusedAdam` (`master_weights` at `master_weight_dtype`, its moments at `exp_avg_dtype` and `exp_avg_sq_dtype`, `store_param_remainders`); `FlashAdamW` (24-bit masters) |
| Clipping | A norm of 1.0, which the configuration cannot turn off | |

### The contract in AutoModel's terms

The contract picks the AutoModel option that expresses it on the case's
ranks; every option stays nameable, a case's own option against the
contract is refused, and `case.json` records the mechanism
(`precision_mechanism`):

| The contract | AutoModel's mechanism |
|---|---|
| bfloat16 model, no master, moments at bfloat16 (`--master-dtype none --opt-state-dtype parameter`) | bfloat16 storage stepped in place by torch's AdamW |
| bfloat16 model, float32 or float16 masters (the default contract) | bfloat16 storage, Transformer Engine's `FusedAdam` holding the masters at the contract's dtype and the moments at the contract's dtype, on any number of ranks (Megatron-FSDP's sharding aside) |
| bfloat16 model, float32 masters and float32 gradients, FSDP2 | FSDP2's float32 storage computed at bfloat16 (`mp_policy`), gradients reduced and kept at float32, torch's AdamW: two or more ranks |
| bfloat16 model, float32 masters, Megatron-FSDP sharded over two or more data-parallel ranks | Megatron-FSDP's float32 main weights (`preserve_fp32_weights`), float32 main gradients when the contract's are (`grad_reduce_in_fp32`), torch's AdamW over them |
| bfloat16 model, no master, moments at float32 or float16 | `FusedAdam` without masters, moments at the contract's dtype |
| float32 model | float32 storage stepped in place by torch's AdamW, the weights their own masters; FSDP2 only on one rank, where DDP and Megatron-FSDP cast to bfloat16 |

FSDP2 reduces gradients at the contract's gradient dtype (`reduce_dtype`), as
the FSDP2 engine does. A parameter the model itself keeps at float32 inside a
narrower model (Qwen 3.5's gated delta net keeps `A_log` and `dt_bias` at
float32) stays there and is stepped beside the others; the first update's
evidence counts such parameters apart (`pinned_float32`), and the request
names them (`automodel_pinned_float32`).

The engine never substitutes a dtype, so a contract no mechanism expresses
is refused. Megatron-FSDP's options apply where it shards, over a
data-parallel shard group of two or more ranks: under tensor parallelism
over the whole world AutoModel does not wrap the model in Megatron-FSDP, and
a contract that needs its float32 options there is refused. Float32
gradients of bfloat16 weights are refused on any number of ranks (one rank,
DDP, or FSDP2 without float32 masters), since AutoModel accumulates
gradients at the weights' dtype; they take FSDP2's float32 storage with
float32 masters on two or more ranks, or Megatron-FSDP's float32 main
gradients where it shards. A float32 model with float16 masters or bfloat16
gradients is refused, since its weights are their own float32 masters and
accumulate at float32, and so is FSDP2's float32 storage with float16
masters, since that storage is the masters. Where Megatron-FSDP shards it
keeps float32 main weights or none, so float16 masters are refused there,
and torch's AdamW steps the main weights with its moments at their dtype, so
moments other than float32 are refused where masters are kept, and other
than the model's dtype without them. On one rank LoRA factors at a dtype
other than the model's are refused, since AutoModel casts the adapters of
every module that owns frozen weights to the compute dtype there. Gradients
accumulate as autograd writes them, so fused narrow-gradient accumulation is
refused and the engine defaults to adding after. Rounding is to nearest.

The first update checks the weights, the gradients, the masters (a remainder
kept by `store_param_remainders` counts as a float32 master) and the moments
against the mechanism (where Megatron-FSDP shards, its main weights, its
main gradients and the moments), and a step that holds another dtype fails
there.

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--optimizer O` | `adamw`, `adam`, `fused_adam`, `flash_adamw` (the contract's choice) | torch's AdamW or Adam, Transformer Engine's FusedAdam, or FlashAdamW | A contract with masters takes `fused_adam` (or Megatron-FSDP's own); torch's optimizers keep their moments at the stepped weights' dtype; FlashAdamW's 24-bit masters are not a dtype the contract names |
| `--[no-]adamw-fused` | (torch's default, unfused) | torch's fused implementation | `adamw`, `adam` |
| `--[no-]store-param-remainders` | (off) | A float32 master kept as its bfloat16 weight and a 16-bit remainder | `fused_adam` with float32 masters |
| `--master-weight-bits N` | integer (24) | FlashAdamW's master bits | `flash_adamw` |
| `--mp-param-dtype D`, `--mp-reduce-dtype D`, `--mp-output-dtype D` | `bfloat16`, `float16`, `float32` (the contract's; output none) | FSDP2's compute and gather dtype, gradient reduce-scatter dtype, and module output cast | FSDP2 on two or more ranks; refused under DDP and Megatron-FSDP |
| `--[no-]preserve-fp32-weights`, `--[no-]grad-reduce-in-fp32` | (the contract's) | Megatron-FSDP's float32 main weights and gradients | Megatron-FSDP sharded over two or more data-parallel ranks |

## The loss

The engine has one default loss. With a tensor-parallel size of 1 it is
`ChunkedCrossEntropy`, whose head and loss never hold the full logits, so a
single device keeps no `tokens x vocabulary` buffer. Under tensor
parallelism it is `TEParallelCrossEntropy`, AutoModel's one loss that
computes on vocabulary shards: the linear losses refuse a tensor-parallel
head at the first forward ("Tensor-parallel or hierarchical layouts need an
explicit compatible loss path"), and the masked loss gathers the whole
logits onto every rank (`full_tensor()`). AutoModel's plans for Llama 3 and
Qwen 3 shard the head over the vocabulary. AutoModel's MoE classes have no
registered plan: at an expert-parallel size of 1 they take AutoModel's base
plan, whose head gathers its output, so `TEParallelCrossEntropy` computes
the whole vocabulary on each rank, and under expert parallelism AutoModel
refuses tensor parallelism without a plan. AutoModel's Qwen 3.5 class
carries no Hugging Face plan (`_tp_plan`), which the plan registered for it
reads, so that plan raises. Qwen 3.5, and the MoE classes under expert
parallelism, take tensor parallelism with `--tp-plan`. A full node without
tensor parallelism (FSDP2 with expert parallelism, as AutoModel's MoE
benchmark recipe) has no vocabulary split: each rank computes the whole
vocabulary over its own tokens, under the chunked loss as on one device.

The chunked loss needs a plain bias-free `nn.Linear` head. AutoModel's MoE
models build theirs through the linear backend, which is Transformer
Engine's where it imports, so beside the chunked loss the linear layers
default to torch's (`--linear torch`), as AutoModel's own chunked recipes
set them. It changes nothing in Llama 3 and Qwen 3, whose layers are
`nn.Linear` or Hugging Face's whatever the backend (`quack` aside); in the
MoE families the attention projections, the dense and shared MLPs and the
head become `nn.Linear`, and in Qwen 3.5 its full-attention projections and
MLP. A case that names `--linear` gets its own, and with `--linear te` names
a loss that takes Transformer Engine's head.

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--loss L` | `chunked`, `masked`, `fused_linear`, `te_parallel` (the engine's: `chunked`; `te_parallel` under tensor parallelism) | `chunked`: `ChunkedCrossEntropy` projects the head and computes the loss `chunk_len` token rows at a time, recomputing each chunk in backward. `masked`: `MaskedCrossEntropy` over the full logits, AutoModel's example recipes' loss. `fused_linear`: `FusedLinearCrossEntropy` on the hidden states and the head's weight. `te_parallel`: Transformer Engine's cross entropy over vocabulary-sharded logits | `chunked` needs a plain bias-free `nn.Linear` head and no logit softcapping; the linear losses need a model whose forward takes `logits_to_keep`, no tensor-parallel head, and pipeline stages that return hidden states |
| `--linear L` | `torch`, `te`, `quack` (the engine's: `torch` with `chunked`; AutoModel's `te` where Transformer Engine imports, else `torch`) | The linear layers of AutoModel's own classes ([The backend](#the-backend)) | |
| `--chunk-len N` | token rows (512) | `chunked`: rows per chunk | |
| `--[no-]chunked-compile` | (on) | `chunked`: compiles each chunk's projection and loss with their backward | |
| `--fused-linear-impl I` | `cce`, `torch_compile` (`cce`) | `fused_linear`: Cut Cross-Entropy, which never holds the logits and is the slowest, or a compiled projection and loss, several times faster with memory growing as tokens times vocabulary | |
| `--[no-]masked-fp32-upcast` | (on) | `masked`: upcasts the logits to float32 | |

Where a model's forward takes no `logits_to_keep`, AutoModel refuses the
chunked loss and replaces `fused_linear` and `te_parallel` with its masked
loss, warning; under pipeline parallelism with stages that return no hidden
states it refuses the chunked loss and replaces `fused_linear`. The engine
reads the loss AutoModel built and refuses a case whose loss was replaced.
Every loss sums over a microbatch's trained tokens and AutoModel divides by
the update's trained tokens across the data-parallel ranks, so the gradient
is that of the update's mean over trained tokens, the shared objective.
AutoModel's own MoE classes add the balancing loss in backward, each
microbatch's term scaled by one over the microbatches; with every position
trained and every microbatch the same length, that is the shared weighting
by trained tokens.

## Attention

Llama 3 and Qwen 3 (AutoModel's classes) and every class AutoModel builds
from Hugging Face's attend through Hugging Face's attention interface, by
`attn_implementation`. Qwen 3 MoE and Qwen 3.5's attention layers attend
through the backend's `--attn`, Transformer Engine's `DotProductAttention`
where it imports. Transformer Engine chooses among FlashAttention (2, or 3 on
compute capability 9.0), cuDNN's fused attention and its unfused path by the
device, the dtype and the shapes, preferring FlashAttention below compute
capability 9.0 and cuDNN from 9.0 where both can run. PyTorch's SDPA picks
among its flash, memory-efficient, math and cuDNN kernels, within the
backends `--sdpa-method` allows; the pinned PyTorch's `libtorch_cuda.so`
carries `TORCH_CUDNN_SDPA_DEPRIORITIZED`, a switch on cuDNN's place in that
choice.

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--attn-implementation I` | `sdpa`, `eager`, `flash_attention_2`, `flash_attention_3`, `flash_attention_4`, `te` (AutoModel's default: `flash_attention_2` where flash-attn imports, else `sdpa`) | Hugging Face's attention, or Transformer Engine's injected into it (`te`) | FlashAttention takes bfloat16 and float16 only, so a float32 model names `sdpa`; FlashAttention 3 on compute capability 9.0 only. `te` is injected into classes built from Hugging Face's, and into AutoModel's Llama 3 and Qwen 3 only with another `--attn`: they hold Transformer Engine's attention for packed batches, so AutoModel skips the injection and they attend by SDPA. Context parallelism takes `sdpa` for classes built from Hugging Face's. AutoModel applies its default to a checkpoint's configuration but leaves its own classes' configuration unset under `from_config`, so the engine puts it on the configuration too |
| `--sdpa-method LIST` | `flash_attention`, `efficient_attention`, `math`, `cudnn_attention` (AutoModel's: flash and efficient under context parallelism, flash, efficient and math under activation checkpointing, none otherwise) | The SDPA backends a patched forward allows; the patch also sets them process-wide, so recomputation in backward takes the same | `--use-sdpa-patching`: without the patch the list does nothing |
| `--[no-]use-sdpa-patching` | (on) | Runs the forward of classes built from Hugging Face's inside `sdpa_kernel(sdpa_method)`, all four backends where no list applies, and of AutoModel's own whenever a list applies: a named `--sdpa-method`, or AutoModel's own under context parallelism or activation checkpointing | |
| `--[no-]use-liger-kernel` | (on) | Liger's kernels in classes built from Hugging Face's | Off under tensor or context parallelism |
| `--[no-]force-hf` | (off) | Hugging Face's implementation even where AutoModel registers its own | A mixture is refused |

The request records the attention each model's configuration names
(`automodel_built`); under `te` that is `sdpa`, whether or not Transformer
Engine's was injected.

## The backend

AutoModel's own classes take a backend (`model.backend`); every field of
AutoModel's `BackendConfig` is an option. Defaults that depend on what
imports are AutoModel's on the device the case trains on.

| Option | Values (default) | What it does | Acts on |
|---|---|---|---|
| `--attn A` | `torch`, `te`, `sdpa`, `flex`, `eager`, `tilelang`, `cudnn`, `fa4`, `magi` (`te` where Transformer Engine imports, else `sdpa`) | The attention module | Qwen 3 MoE and Qwen 3.5; Llama 3 and Qwen 3 take `te` only for packed batches |
| `--sparse-attn S` | `generic`, `msa` (`generic`) | Sparse attention: each model's own, or the SM100 MSA kernels | MiniMax M3; none in the catalog |
| `--linear L` | `torch`, `te`, `quack` | See [The loss](#the-loss): `torch` and `te` act on Qwen 3 MoE (projections, dense and shared MLPs, head), Qwen 3.5 (full-attention projections and MLP) and Qwen 3.5 MoE; `quack` replaces every plain `nn.Linear` of AutoModel's own classes | |
| `--rms-norm N` | `torch`, `torch_fp32`, `te`, `quack` (`torch_fp32`, a compiled float32 norm) | The RMSNorm | Llama 3 and Qwen 3 MoE; Qwen 3 keeps Hugging Face's, Qwen 3.5 its own |
| `--rope R`, `--[no-]rope-fusion` | `torch`, `quack` (`torch`); fusion forced off | The rotary embedding (QuACK's for the Llama family); AutoModel turns Transformer Engine's fused rotary embedding off whatever the option says | |
| `--experts E` | `torch`, `te`, `torch_mm`, `torch_mm_mxfp8` (`torch_mm`) | A loop over experts, Transformer Engine's grouped linear, `torch._grouped_mm`, or its MXFP8 form (compute capability 10.0 or later with torchao's MoE training importable, else `torch._grouped_mm` with a warning) | MoE models; `te` without a flex dispatcher is reset to `torch_mm` |
| `--dispatcher D` | `torch`, `deepep`, `hybridep`, `uccl_ep`, `mok` (`deepep` where DeepEP imports, `uccl_ep` where UCCL-EP does, else `torch`) | The token dispatcher | MoE models under expert parallelism; on one rank DeepEP, HybridEP and UCCL-EP fall back to the grouped experts with a warning (and `--experts te` with them loops over experts); `mok` needs a world size divisible by 4 |
| `--dispatcher-num-sms N`, `--[no-]dispatcher-share-token-dispatcher`, `--[no-]dispatcher-async-dispatch` | (20), (on), (off) | A DeepEP-family dispatcher's multiprocessors, one communication manager for every layer, asynchronous dispatch and combine | Flex dispatchers |
| `--[no-]dispatcher-hybridep-permute-fusion`, `--[no-]dispatcher-hybridep-compact-routing`, `--dispatcher-hybridep-num-sms-preprocessing N`, `--dispatcher-hybridep-num-blocks-permute N`, `--dispatcher-hybridep-num-blocks-unpermute N` | (off), (off), (HybridEP's) | HybridEP's permutation fusion, compact routing and kernel sizes | `--dispatcher hybridep` |
| `--mok-precision P`, `--mok-fwd-num-comm-sms N`, `--mok-bwd-num-comm-sms N`, `--mok-minibatch-size N`, `--mok-macrobatch-size N`, `--mok-schedule-capacity-multiplier F`, `--mok-all-gather-top-experts-chunk-bytes N` | `bf16`, `mxfp8` (`bf16`); (40, 28, 4096, 131072, 0.5, 2048) | Mixture-of-Kittens' tuning | `--dispatcher mok`, on a world size divisible by 4 |
| `--[no-]fake-balanced-gate`, `--fake-gate-noise F`, `--[no-]benchmark-static-routing` | (off); 0 to 1 (0); (off) | Uniform routing in place of the router, its noise, and the first microbatch's routing metadata reused | Performance studies, not training |
| `--gate-precision D` | a dtype (none: the router projects at the activations' dtype and scores at float32) | The router computation's dtype | MoE models |
| `--[no-]enable-fsdp-optimizations` | (off) | Defers gradient sync and resharding to the last microbatch | MoE models |
| `--[no-]enable-hf-state-dict-adapter` | (on) | Converts between Hugging Face's checkpoint layout and the model's | Checkpoint import |
| `--te-fp8 R` | `current`, `block`, `mxfp8` (none) | Transformer Engine's FP8 around every forward | `--linear te` or `--experts te` |
| `--[no-]compile-attn` | (off) | Compiles the attention forward whole | Qwen 3 MoE with `--attn sdpa --linear torch --rms-norm torch`; ignored with a warning under another `--attn` |
| `--[no-]compile-router-weight` | (off) | Compiles the float32 router-weight multiply of the grouped experts | MoE models |
| `--[no-]compile-situ`, `--[no-]compile-norm`, `--[no-]compile-hc`, `--[no-]shared-expert-overlap` | (off) | Compiled SiTU activations, float32 norm chains and hyper-connection cores; shared experts on a side stream | The models that opt in (Kimi K3, DeepSeek V4.1, MiniMax M3's norms); none in the catalog |
| `--cuda-graph-modules LIST` | `attn`, `te_dpa`, `moe_router`, `moe_preprocess` (none) | Graph capture of these parts, after the first update | `attn` and `te_dpa` need `--attn te`, `attn` also `--linear torch --rms-norm torch`; `moe_preprocess` needs `hybridep` and `moe_router`. Refused: under pipeline parallelism; `attn` with `te_dpa`; `attn` and the MoE graphs under activation checkpointing; `moe_router` with the fake gate; FP8 attention; a graphed part that does not run in the first update, as `te_dpa` in Llama 3 and Qwen 3 or wherever Transformer Engine picks FlashAttention |

## Distribution

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--strategy S` | `fsdp2`, `megatron_fsdp`, `ddp` (`fsdp2`) | PyTorch's FSDP2, Megatron-FSDP, or PyTorch's DDP | AutoModel refuses an option of another strategy ("Unknown options for strategy") |
| `--tp-size N`, `--pp-size N`, `--cp-size N`, `--ep-size N` | integers (1) | Tensor-parallel ranks, pipeline stages, context-parallel ranks, expert-parallel ranks | The world divides by tensor x context x pipeline, and the expert ranks divide data x context x tensor; DDP refuses any of the four above 1, Megatron-FSDP pipeline and expert ranks |
| `--dp-replicate-size N` | integer (none) | FSDP2's hybrid sharding: replicas of the sharded model | FSDP2 |

On one rank no strategy shards or wraps: FSDP2 skips `fully_shard` (and with
it `mp_policy` and `offload_policy`), and DDP and Megatron-FSDP cast the model
to bfloat16. Megatron-FSDP also wraps nothing where its data-parallel shard
group is one rank, as under tensor parallelism over the whole world:
AutoModel applies tensor parallelism alone, and Megatron-FSDP's options do
not apply. The engine writes `--cpu-offload`, `--offload-pin-memory` and
the `--mp-*` dtypes into FSDP2's policies, so under DDP and Megatron-FSDP it
refuses them, as AutoModel refuses an option of another strategy.

### FSDP2

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]sequence-parallel`, `--tp-plan PATH` | (off); an import path (AutoModel's plan) | The sequence-parallel variant of the tensor-parallel plan; a custom plan | Tensor parallelism; under expert parallelism sequence parallelism is refused and tensor parallelism takes `--tp-plan`, as Qwen 3.5's does ([The loss](#the-loss)) |
| `--[no-]defer-fsdp-grad-sync` | (on) | Skips the reduce-scatter on all but the last microbatch, accumulating unsharded gradients | |
| `--[no-]reshard-after-forward` | (every layer but the last; none under pipeline parallelism; under expert parallelism the MoE path's, off) | Frees each layer's gathered weights after its forward | Named, it overrides the MoE path's default |
| `--[no-]enable-compile` | (off) | `torch.compile` on each decoder layer after sharding | The FSDP2 way to compile |
| `--[no-]enable-fsdp2-prefetch`, `--fsdp2-backward-prefetch-depth N`, `--fsdp2-forward-prefetch-depth N` | (off); (2); (1) | Explicit prefetch chains between layers, and their depths | No effect under pipeline or expert parallelism, with `--no-reshard-after-forward`, or on Qwen 3.5 |
| `--[no-]enable-async-tensor-parallel` | (off) | Asynchronous tensor parallelism; it also compiles each layer | Refused without sequence parallelism at tensor parallelism above 1, and on the MoE path |
| `--[no-]patch-is-packed-sequence` | (off) | Skips transformers' packed-sequence check, one host synchronization per attention layer | Unpacked batches, as the engine's |
| `--[no-]cpu-offload`, `--[no-]offload-pin-memory` | (off), (on) | FSDP2's `CPUOffloadPolicy`: the weight shards, gradients and optimizer step on the host | Two or more ranks; refused under DDP and Megatron-FSDP, and with a contract that maps to Transformer Engine's `FusedAdam` (masters, or moments at a dtype other than the stepped weights'), which steps on the device while the shards stay on the host. On one rank nothing offloads: weights built without a checkpoint, or for LoRA, stay on the host unmoved; a full fine-tune from a checkpoint loads on the device |

### Megatron-FSDP

| Option | Values (default) | What it does |
|---|---|---|
| `--zero-dp-strategy N` | `1`, `2`, `3` (`3`) | The ZeRO level; below 3 the weights are not sharded |
| `--[no-]overlap-grad-reduce`, `--[no-]overlap-param-gather` | (on), (on) | Overlaps the gradient reduction with backward and the weight gather with forward |
| `--[no-]check-for-nan-in-grad` | (on, a no-op) | Megatron-FSDP 0.5.0 checks nothing here; AutoModel drops the flag with a warning |
| `--[no-]average-in-collective`, `--[no-]disable-bucketing`, `--[no-]nccl-ub`, `--[no-]fsdp-double-buffer`, `--[no-]init-fsdp-with-meta-device` | (off) | Averages in the reduction; one bucket per parameter; NCCL user buffers; persistent double buffers; meta initialization |

AutoModel skips activation checkpointing under Megatron-FSDP on more than one
rank, logging an error.

### DDP

| Option | Values (default) | What it does |
|---|---|---|
| `--bucket-cap-mb F` | megabytes (PyTorch's) | Gradient bucket size |
| `--[no-]gradient-as-bucket-view`, `--[no-]static-graph`, `--[no-]broadcast-buffers`, `--[no-]find-unused-parameters` | (off) | Gradients as views of the buckets; a graph that does not change; buffers broadcast every forward; parameters a forward did not use |

### Mixture of experts and pipeline parallelism

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]ignore-router-for-ac` | (on) | Saves the router's projection and top-k under checkpointing, so the recomputation routes the same tokens | `--ep-size` above 1 |
| `--[no-]moe-reshard-after-forward` | (off) | Reshards every decoder block (dense ones too), the embeddings, the head and the roots after forward | `--ep-size` above 1; FSDP2's `--reshard-after-forward` overrides it |
| `--lm-head-precision D` | a dtype (none) | float32 gives the untied head float32 compute, reduction and output; other dtypes do nothing, and a tied head ignores it with a warning | `--ep-size` above 1 |
| `--[no-]wrap-outer-model` | (on) | Wraps the whole model in an FSDP unit of its own | `--ep-size` above 1 |
| `--pp-schedule NAME`, `--pp-schedule-csv PATH` | PyTorch's schedule names (`1f1b`); a file | The pipeline schedule | `--pp-size` above 1 |
| `--pp-microbatch-size N` | sequences (1) | Sequences per pipeline microbatch | A microbatch's sequences divide into at least as many pipeline microbatches as stages |
| `--layers-per-stage N`, `--round-virtual-stages-to-pp-multiple up\|down` | (even), (none) | Layers per (virtual) stage; the rounding takes the virtual-stage count, layers over layers per stage rounded up, to a multiple of the stages, which without it must already divide by them | |
| `--[no-]scale-grads-in-schedule`, `--[no-]patch-inner-model`, `--[no-]patch-causal-lm-model`, `--[no-]pp-recv-buffer-pool` | (off), (on), (on), (off) | The schedule scales the gradients; the stage forwards patched; receive buffers reused | |

A `--pp-size` above 1 with no pipeline option takes AutoModel's pipeline
defaults: the engine writes the pipeline section, which AutoModel
otherwise requires.

## Activation checkpointing and offloading

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--activation-checkpointing C` | `none`, `full`, `selective` (`none`) | `full` under FSDP2: Hugging Face's classes recomputed whole layer by layer, AutoModel's own with each layer's attention, MLP and norms wrapped one by one, the residual adds outside; with `--enable-compile` on two or more ranks only attention and MLP are wrapped; under DDP on two or more ranks every class is wrapped by parts; under expert parallelism every block is wrapped whole, saving the router's projection and top-k; under Megatron-FSDP on one rank only classes with Hugging Face's `gradient_checkpointing_enable` (AutoModel's Qwen 3 MoE, Qwen 3.5 and Qwen 3.5 MoE have none, and AutoModel skips them with an error). `selective`: every decoder block recomputed under a policy that saves every other matrix multiply (`mm`, `linear`, grouped `mm`), always saves `addmm`, batched and scaled multiplies, PyTorch's compute-intensive operations, `topk`, `max` and copies to the host, saves SDPA and flex attention, saves all-to-all and reduce-scatter, and recomputes the rest, FlashAttention 2's kernel and Transformer Engine's fused attention included | `selective` with FSDP2 or DDP only; the shared `--recompute` is `full` or none, so a case names one or the other |
| `--activation-checkpointing-scope LIST` | `all`, `language`, `vision`, `audio`, `multimodal` (`all`) | The layer groups checkpointing wraps | FSDP2 or DDP; meaningful for multimodal models |

AutoModel offloads no activations: its one offload is FSDP2's
`CPUOffloadPolicy`, which needs two or more ranks to shard, so on one
device AutoModel offloads nothing.

### What each offload option moves

Where each option puts the weights, gradients, optimizer state and activations,
and where the optimizer steps; an empty cell is the default's.

| Option | Weights the compute uses | Weights the optimizer steps | Gradients | Optimizer state and step | Activations |
|---|---|---|---|---|---|
| none (the default) | device | device | device | device | device |
| `--cpu-offload` (FSDP2's `CPUOffloadPolicy`, two or more ranks) | device: each block's gathered from the host before it runs | host: the shards | host | host; the step on the CPU | device |

## FP8 and compilation

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]fp8`, `--fp8-recipe-name R` | (off); `tensorwise`, `rowwise`, `rowwise_with_gw_hp` (`tensorwise`) | torchao's float8 training of the linear layers whose sizes divide by 16, the head included; a conversion error is logged and the model kept unconverted | Compute capability 8.9 or later; Transformer Engine's linears are not converted; the converted head is one the chunked loss refuses, so `--fp8` names another loss |
| `--[no-]enable-fsdp-float8-all-gather`, `--[no-]precompute-float8-dynamic-scale-for-fsdp`, `--[no-]force-recompute-fp8-weight-in-bwd` | (off) | float8 weight gathers, scales precomputed after each update, float8 weights recomputed in backward | `--fp8`; the gather and the recompute with `tensorwise` only; the precompute with FSDP2 on two or more ranks and no pipeline parallelism |
| `--[no-]compile`, `--compile-mode M`, `--[no-]compile-fullgraph`, `--[no-]compile-dynamic`, `--compile-backend B`, `--dynamo-cache-size-limit N` | (off); PyTorch's modes (`default`); (off); (off); (inductor); (256) | `torch.compile` of the whole model | Ignored under FSDP2 (which takes `--enable-compile`) and with pipeline parallelism |
| `--gc-every-steps N` | updates (none: automatic collection) | Turns automatic garbage collection off and collects every N updates | |

## LoRA

LoRA is AutoModel's `PeftConfig` over the modules the shared LoRA
definition's rules select (`workloads/lora`), applied to AutoModel's own
module layout: every linear layer but the head, and the routed experts of
each mixture block; inside a mixture block nothing else, unless the case's
`shared_experts` adds its linear layers other than its router. The trainer
builds the class AutoModel will build on the meta device, which holds no
weights, selects the modules by those rules by module type, and passes their
names as `target_modules`; the request records them
(`automodel_lora_targets`). AutoModel adapts what a target matches and
passes over a target that matches nothing, so the engine refuses a target
that matches no module of the model AutoModel built. The adapters take the
case's rank (`dim`), its alpha as given (AutoModel scales the adapters'
output by alpha over `dim`) and its `factor_dtype`, the head frozen. A case
takes the `rank`, `alpha`, `factor_dtype`, `head` (frozen) and
`shared_experts` entries of `--lora`; LoRA adapts a pretrained model when it
names `--checkpoint`, and AutoModel's own initialization otherwise.

Each engine trains its framework's default LoRA layout over the shared
definition's targets. AutoModel adapts each Hugging Face projection
separately, so where the catalog fuses projections (Qwen 3.5's
linear-attention inputs `w_qkvz` and `w_ba`, and the shared expert's
`gate_up`), AutoModel's split ones (`in_proj_qkv`, `in_proj_z`, `in_proj_b`
and `in_proj_a`; `gate_proj` and `up_proj`) each carry their own A factor, a
few more trainable parameters than the catalog's; the routed experts'
factors are the catalog's shapes transposed. Engines are compared by total
trainable parameters, the elements of every trainable tensor, which each row
records from the model AutoModel built.

| Option | Values (default) | What it does | Goes with |
|---|---|---|---|
| `--[no-]use-memory-efficient-lora` | (on) | The custom autograd LoRA path, and fused LoRA MLPs | |
| `--[no-]use-triton` | (off) | Triton LoRA kernels | AutoModel turns them off where Transformer Engine imports, and under tensor or pipeline parallelism and DoRA |
| `--[no-]moe-rank-scaling` | (off) | Expert adapters of rank `dim // experts per token` | |
| `--[no-]use-dora` | (off) | DoRA | Plain `nn.Linear` layers only: it raises on Transformer Engine's linears and on grouped experts |
| `--lora-dropout F`, `--lora-dropout-position P` | (0); `pre`, `post` (`post`) | Dropout on the adapters' path, before or after the adapter | The shared objective has none: a case with dropout departs from the other engines' |

## AutoModel's benchmark recipes

AutoModel's own benchmark recipes map onto engine options; the shared
settings take the geometry, the optimizer's hyperparameters and the data.
Llama 3.1 8B LoRA
([`llama3_1_8b_peft_benchmark.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/llm_benchmark/llama3_1/llama3_1_8b_peft_benchmark.yaml),
written for one node; here on one device):

```bash
--model llama3-8b --mode lora --lora '{"rank": 16, "alpha": 32}' --checkpoint <Llama 3.1 8B> \
  --sequence-length 4096 --tokens-per-step 131072 --tokens-per-microbatch 8192 \
  --force-hf --loss masked --no-use-memory-efficient-lora --use-triton \
  --master-dtype none --opt-state-dtype parameter --learning-rate 1e-5 --weight-decay 0.01
```

Qwen 3 MoE 30B on one node with expert parallelism
([`qwen3_moe_30b_te_deepep.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml)):

```bash
--model qwen3moe-30b-a3b --sequence-length 4096 --tokens-per-step 2097152 \
  --tokens-per-microbatch 16384 --ep-size 8 --activation-checkpointing full \
  --no-moe-reshard-after-forward --no-ignore-router-for-ac \
  --attn te --linear te --rms-norm te --experts torch_mm \
  --dispatcher deepep --fake-balanced-gate --enable-hf-state-dict-adapter \
  --enable-fsdp-optimizations --loss masked --no-masked-fp32-upcast \
  --gc-every-steps 10 --optimizer adam --master-dtype none \
  --opt-state-dtype parameter --learning-rate 1e-4 --weight-decay 0
```

Where they differ from the recipes: the first recipe's LoRA dropout of 0.1
is `--lora-dropout 0.1`, outside the shared objective, its `use_triton` is a
no-op where Transformer Engine imports, as in the nvidia-training
environment, and it names no `clip_grad_norm`, so AutoModel clips at its
default norm of 1.0, which the engine does not; the second recipe reduces
gradients at float32 over bfloat16 weights, which the contract does not name
(its gradient dtype is the reduction's), sets `rope_fusion`, which AutoModel
turns off, and runs AutoModel's benchmarking recipe, which never clips and
sums its loss unnormalized; its DeepEP is the one the nvidia-training
environment builds where 9.0 or 10.0 is a target ([Limits](#limits)).

## Data

The tokens are the ones every engine trains on (`baselines/common/runner.py`):
synthetic tokens drawn once per data-parallel replica from `--seed`, or each
replica's window of the corpus for each update. They reach AutoModel through
its own data path: the recipe's dataset is the engine's
`baselines.engines.automodel.data.CaseTokens`, which yields each update's
microbatches as AutoModel's language-model datasets yield batches, each
`sequences per microbatch x sequence length` input tokens with their next
tokens as labels and their positions, every position trained. AutoModel's
dataloader routes each rank to its replica through the dataset's `shard`, as
it shards any iterable dataset, and the recipe moves each batch to the
device. The dataset section names no tokenizer, which the recipe would
otherwise load from the model's path.

## Measurement

AutoModel's loop runs the case. The runner's clock times each optimizer
update inside AutoModel's step (`_run_train_optim_step`), from a device
synchronization and a barrier across the ranks to the end of the step's
blocking loss reduction. The configuration cannot turn clipping off, so the
engine calls the step without a norm, as every engine steps; AutoModel
then computes no gradient norm (its log shows 0). AutoModel's logging,
garbage collection and checkpoint checks between updates fall outside.
Checkpointing, validation and the preemption handler are off, and the
learning rate is constant. AutoModel's first rank resets the allocator's peak
each time it logs an update, so the engine records each measured update's
peak itself and the row takes the largest.

## What a run writes

```text
output/
  case.json                    # the case the launcher resolved: loss, linear layers, precision mechanism
  rank-NNNNN/request.json      # the case as described, the whole recipe configuration, and what AutoModel built
  rank-NNNNN/result.json       # cycles, losses, peaks, precision evidence; or failure.json
  rank-NNNNN/huggingface/      # the configuration a from_config model is built from
  rank-NNNNN/automodel/        # AutoModel's training.jsonl (written by the first rank)
  results.json                 # every rank's measurements (first rank)
  row.json                     # the shared result row (first rank)
```

`automodel_built` in the request names the model's class (under DDP or
Megatron-FSDP, that of the model inside the wrapper), its head's class, the
attention its configuration names, its backend and the loss;
`automodel_replica` the data-parallel replica the dataset served,
`automodel_pinned_float32` the parameters the model keeps at float32, and
`automodel_lora_targets` the modules LoRA adapts. The row's trainable
parameter count is that of the model AutoModel built.

## Limits

On one device AutoModel shards nothing: it offloads nothing (its one
offload, FSDP2's `CPUOffloadPolicy`, needs sharding), FSDP2's mixed
precision does not apply, and float32 masters come only from Transformer
Engine's `FusedAdam`. A model whose weights, gradients and optimizer state
exceed one device does not train on one device. The nvidia-training
environment builds DeepEP, with HybridEP, only where 9.0 or 10.0 is a target
and installs no UCCL-EP or Mixture-of-Kittens, so elsewhere the dispatcher
is `torch`; FlashAttention 3, installed whatever the GPU, runs on compute
capability 9.0 only.
