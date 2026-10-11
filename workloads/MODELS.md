# Example model catalog

Start here to choose a supplied architecture, change its dimensions, and select
training precision. These are ordinary PyTorch modules passed into the generic
[Trainer](../docs/python/api/training.md), `Forward`, or planning APIs. Model
construction, trainable parameters, and data semantics belong to the caller.

## Contents

- [Architecture index](#architecture-index)
- [Preset dimensions and parameter counts](#preset-dimensions-and-parameter-counts)
- [Llama 3](#llama-3)
- [Dense Qwen 3.5](#dense-qwen-35)
- [OLMoE](#olmoe)
- [Qwen 3 MoE](#qwen-3-moe)
- [Qwen 3.5 MoE](#qwen-35-moe)
- [GLM-5.3-Flash checkpoint integration](#glm-53-flash-checkpoint-integration)
- [Dtypes and optimizer state](#dtypes-and-optimizer-state)
- [Trainable parameters and LoRA](#trainable-parameters-and-lora)
- [Expert parallelism](#expert-parallelism)
- [Inputs, outputs, and loss](#inputs-outputs-and-loss)
- [Construction and runnable recipes](#construction-and-runnable-recipes)
- [Source organization](#source-organization)

## Architecture index

`mlops_qwen3moe` means **Qwen 3 with MoE**, whose default shape is 30B-A3B.
`mlops_qwen35` means the **dense** Qwen 3.5 architecture.
`mlops_qwen35moe` means **Qwen 3.5 with MoE**, whose default shape is 35B-A3B.
Dimensions are configurable; the architecture name does not force a model size.

| Architecture | Python model and configuration | Implementations | Quickstart names |
|---|---|---|---|
| [Llama 3](#llama-3) | `Llama3`, `Llama3Config` | `workloads.pytorch`, `workloads.mlops` | `pytorch_llama3`, `mlops_llama3` |
| [Dense Qwen 3.5](#dense-qwen-35) | `Qwen35`, `Qwen35Config` | `workloads.pytorch`, `workloads.mlops` | `pytorch_qwen35`, `mlops_qwen35` |
| [OLMoE](#olmoe) | `OLMoE`, `OLMoEConfig` | `workloads.pytorch`, `workloads.mlops`; optional EP in MLOps | `mlops_olmoe` |
| [Qwen 3 MoE](#qwen-3-moe) | `Qwen3MoE`, `Qwen3MoEConfig` | `workloads.mlops`; optional EP constructor | `mlops_qwen3moe` |
| [Qwen 3.5 MoE](#qwen-35-moe) | `Qwen35MoE`, `Qwen35MoEConfig` | `workloads.mlops`; optional EP constructor | `mlops_qwen35moe` |

Pure PyTorch implementations provide readable numerical references. MLOps
implementations use the separately installed operation library. PyTorch OLMoE
is available as a model, but has no supplied full-model quickstart preset.
All three MLOps MoE architectures can enable EP through the same Python
constructor arguments. An experiment factory supplies groups and resources;
a text quickstart name alone does not enable EP.

For a non-text example, see the [regression recipe](recipes/regression.py) and
[generic training walkthrough](../docs/examples/generic-training.md).

## Preset dimensions and parameter counts

`numerical()` supplies a smaller validation shape; `throughput()` supplies a
larger performance shape. Both are ordinary configuration dataclasses. Qwen
MoE configurations also have published dimensions as constructor defaults.

Counts below include embeddings and the output head, count shared Parameter
objects once, and exclude buffers and optimizer state. They describe the whole
unsharded model, and equal its trainable count under default full training.
They are **total parameters**, not the subset used by one routed token.

| Architecture / preset | Layers | Model width | Q / KV heads | Head width | MLP or expert width | Experts / top-k | Vocabulary | Total parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3 numerical | 12 | 2,048 | 16 / 4 | 128 | 7,168 | — | 128,256 | 1,179,699,200 |
| Llama 3 throughput | 32 | 4,096 | 32 / 8 | 128 | 14,336 | — | 128,256 | 8,030,261,248 |
| Dense Qwen 3.5 numerical | 8 | 1,536 | 12 / 4 | 128 | 4,608 | — | 248,320 | 1,006,955,408 |
| Dense Qwen 3.5 throughput | 32 | 4,096 | 16 / 4 | 256 | 12,288 | — | 248,320 | 8,953,803,264 |
| OLMoE numerical | 16 | 1,024 | 8 / 8 | 128 | 1,024 | 16 / 4 | 50,304 | 975,766,528 |
| OLMoE throughput | 16 | 2,048 | 16 / 16 | 128 | 1,024 | 64 / 8 | 50,304 | 6,919,161,856 |
| Qwen3-30B-A3B default | 48 | 2,048 | 32 / 4 | 128 | 768 | 128 / 8 | 151,936 | 30,532,122,624 |
| Qwen 3.5 MoE default | 40 | 2,048 | 16 / 2 | 256 | 512 routed + 512 shared | 256 / 8 | 248,320 | 34,660,610,688 |

The dense Qwen presets use a full-attention block every fourth layer. Their
DeltaNet key/value head counts are 6/12 (numerical) or 16/32 (throughput), with
128-wide heads and a width-4 convolution. The Qwen 3.5 MoE default uses 16/32
DeltaNet heads of width 128 and the same convolution width.

Qualification may select smaller shapes and FP16 precision on devices below
SM80; that policy lives in [qualification](../qualification/device_defaults.py),
not in the models. Parameter counts are independent of dtype. Physical storage
also depends on dtype, tensor representations, sharding, and runtime resources.

## Llama 3

GQA attention and a dense SwiGLU MLP in every decoder block, with RMSNorm and
RoPE. Embedding and output-head weights are separate.

Configuration: [Llama3Config](pytorch/llama3.py). Implementations:
[PyTorch](pytorch/llama3.py), [MLOps](mlops/llama3.py).

| Option | Meaning / default |
|---|---|
| `n_layers`, `d_model` | Number of decoder blocks and residual width |
| `n_heads`, `n_kv_heads` | Query heads and grouped key/value heads; query heads must divide into KV groups |
| `d_ff` | Intermediate width of each dense SwiGLU projection |
| `vocab_size` | Embedding/output-head row count; may include padded vocabulary rows |
| `rope_base` | RoPE base; 500,000 |
| `max_seq_len` | Maximum supported sequence length / rotary-table capacity; 131,072 |

`head_dim` is derived as `d_model // n_heads`; the division must be exact.
All shape fields above `rope_base` are required for a custom configuration.
Use `Llama3Config.numerical()` or `.throughput()` for ready-made shapes.

## Dense Qwen 3.5

A repeating hybrid of Gated DeltaNet and gated GQA, followed by a dense SwiGLU
MLP in every block. `full_attention_interval=4` means three DeltaNet blocks,
then one full-attention block. This example has no routed experts.

Configuration: [Qwen35Config](pytorch/qwen35.py). Implementations:
[PyTorch](pytorch/qwen35.py), [MLOps](mlops/qwen35.py).

| Options | Meaning / defaults |
|---|---|
| `n_layers`, `d_model`, `d_ff`, `vocab_size` | Decoder depth, residual width, dense MLP width, vocabulary rows |
| `full_attention_interval` | Full attention every Nth block, counting from 1 |
| `n_heads`, `n_kv_heads`, `head_dim` | Full-attention query/KV heads and per-head width |
| `partial_rotary_factor` | Fraction of each attention head using RoPE; 0.25 in both presets |
| `lin_k_heads`, `lin_v_heads` | DeltaNet key/value head counts; value heads must divide into key-head groups |
| `lin_k_head_dim`, `lin_v_head_dim` | DeltaNet key/value head widths |
| `lin_conv_kernel` | Depthwise causal-convolution width |
| `rope_base` | 10,000,000 |
| `tied_embeddings` | Share embedding and output-head weights; false |
| `max_seq_len` | Maximum sequence length; 131,072 |

Full-attention projection width is `n_heads * head_dim` and may differ from
`d_model`. The rotary width must be positive and even. With tied embeddings,
freezing or training either name affects the same Parameter.

## OLMoE

GQA attention followed by a routed SwiGLU expert block. The router takes a
softmax over all experts and selects top-k without renormalizing the selected
weights. Each expert has a packed gate/up projection and a down projection.
The configuration contains no shared expert.

Configuration: [OLMoEConfig](pytorch/olmoe.py). Implementations:
[PyTorch](pytorch/olmoe.py), [MLOps with optional EP](mlops/olmoe.py).

| Options | Meaning / defaults |
|---|---|
| `n_layers`, `d_model`, `vocab_size` | Decoder depth, residual width, vocabulary rows |
| `n_heads`, `n_kv_heads`, `head_dim` | Attention head geometry |
| `n_experts` | Total routed experts per layer, across the entire EP group when used |
| `top_k` | Experts selected per token; between 1 and `n_experts` |
| `d_ff_expert` | Intermediate width of each expert |
| `rope_base`, `max_seq_len` | 10,000 and 131,072 |

Q/K normalization is applied before rotary attention. The auxiliary load
balancing loss is returned separately by `hidden`; `loss(..., aux_coef=...)`
chooses its contribution. The model loss defaults to coefficient 0.0; supplied
training/qualification objectives can choose another value.

## Qwen 3 MoE

GQA attention followed by MoE in **every** block. The selected routing
probabilities sum to one. The default shape follows the published
[Qwen3-30B-A3B configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json).

Use `workloads.mlops.Qwen3MoE(Qwen3MoEConfig(...))`. Configuration:
[Qwen3MoEConfig and Qwen3MoE](mlops/qwen3_moe.py).

## Qwen 3.5 MoE

Three Gated DeltaNet blocks followed by gated GQA, repeating through the
network, with MoE after every mixer. The default shape follows the text decoder
of [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json).
It has a shared SwiGLU expert with a learned sigmoid gate, zero-centered RMSNorm,
and normalized top-k routed probabilities.

Use `workloads.mlops.Qwen35MoE(Qwen35MoEConfig(...))`. Configuration:
[Qwen35MoEConfig and Qwen35MoE](mlops/qwen35_moe.py). The vision encoder and multi-token prediction
module are outside this text-model example. These constructors initialize model
state; they do not download or import pretrained Hugging Face checkpoints.

Both Qwen MoE configurations expose the following fields:

| Options | Qwen3MoE default | Qwen35MoE default |
|---|---|---|
| `n_layers`, `d_model` | 48, 2,048 | 40, 2,048 |
| `n_heads`, `n_kv_heads`, `head_dim` | 32, 4, 128 | 16, 2, 256 |
| `n_experts`, `top_k`, `d_ff_expert` | 128, 8, 768 | 256, 8, 512 |
| `d_ff_shared` | 0 (disabled) | 512 |
| `vocab_size`, `max_seq_len` | 151,936; 40,960 | 248,320; 262,144 |
| `rope_base`, `partial_rotary_factor` | 1e6, 1.0 | 1e7, 0.25 |
| `norm_epsilon`, `zero_centered_norm` | 1e-6, false | 1e-6, true |
| `attention_gate`, `full_attention_interval` | false, 1 | true, 4 |
| `lin_k_heads`, `lin_v_heads` | 16, 32 | 16, 32 |
| `lin_k_head_dim`, `lin_v_head_dim`, `lin_conv_kernel` | 128, 128, 4 | 128, 128, 4 |
| `initializer_range` | 0.02 | 0.02 |
| `router_aux_loss_coef` | 0.001 | 0.001 |

DeltaNet settings apply to layers selected as linear attention. Both defaults
use separate embedding and head weights. Balancing follows the across-layer
Qwen convention: balanced top-8 routing gives an unweighted auxiliary near 8.
`loss(..., aux_coef=...)` overrides the configured coefficient.

## GLM-5.3-Flash checkpoint integration

The [GLM-5.3-Flash workload](mlops/glm53_flash/README.md) imports local
Hugging Face safetensors directly into an initialized runtime pool. It preserves
the original FP8 or NVFP4 expert storage and initially uses BF16 GEMMs.
No permanent converted checkpoint is required. An SSD pool still contains a
temporary runtime copy.

Its architecture has 45 layers: 34 KDA, 11 NoPE sparse-MLA, mHC streams, three
initial dense MLPs, then 288 routed experts/top-8 plus a shared expert.
This checkpoint integration has its own text/image forward-validation CLI; it
is not yet a supplied quickstart/trainer preset. Complete text forwards for both
checkpoints and the checkpoint vision encoders execute through SSD-backed plans.
Reduced-model LoRA updates and selected-state checkpoint replay are also tested.
See the workload README for numerical limits and the current image/text status;
full-model LoRA training and checkpoint-specific low-precision GEMMs remain
unqualified.

## Dtypes and optimizer state

Shape dataclasses describe architecture. Precision is chosen during model
construction and when building the trainer/optimizer. Internal reductions may
use FP32 even when stored weights and activations use BF16 or FP16.

| Setting | Python / recipe location | Text quickstart flag | Default / choices |
|---|---|---|---|
| Model weights | Construct with the desired PyTorch dtype; recipe `model.dtype` | `--model-dtype` | Quickstart: `bfloat16`; also `float16`, `float32` |
| Optimizer masters | Trainer `master_dtype` | `--master-dtype` | No masters; or a floating dtype |
| Accumulated gradients | Trainer `grad_dtype` | `--grad-dtype` | Parameter dtype; or a floating dtype |
| Optimizer moments | `mlops.optim.AdamW(opt_state_dtype=...)` / `optimizer_args.opt_state_dtype` | `--opt-state-dtype` | MLOps: `bfloat16`; also `float16`, `float32`, or `parameter` |
| Parameter rounding | Optimizer `parameter_rounding` | `--parameter-rounding` | `nearest` or `stochastic` |
| Moment rounding | Optimizer `opt_state_rounding` | `--opt-state-rounding` | `nearest` or `stochastic`; ShadowSpill defaults to stochastic for BF16 AdamW moments, nearest otherwise |

Ordinary constructors follow PyTorch's default dtype unless a dtype is supplied
or the module is converted. The text recipe's `build_on_meta(..., dtype=...)`
selects storage dtype without allocating weights. Quickstart does not choose
FP16 automatically on older hardware. For example:

```bash
python -m training.train training/configs/llama3_1b.json \
  model.dtype=float16 grad_dtype=@torch:float32 \
  optimizer_args.opt_state_dtype=@torch:float32 master_dtype=null
```

For stock `torch.optim.AdamW`, FP32 moments with FP16 compute weights require
FP32 masters. MLOps AdamW supports independently selected moment storage. See
[training precision](../training/README.md#precision-and-initialization).

FP8 is an [expert-compute setting](#expert-parallelism), not a value for
`--model-dtype`. Selecting FP8 experts does not convert attention, router,
normalization, shared experts, or the output head to FP8.

## Trainable parameters and LoRA

All supplied full-model constructors default to full training. Select frozen
or trainable parameters **before** creating an optimizer or preparing a trainer.
The generic contract is `Parameter.requires_grad`; the planner does not need a
model-specific training-mode flag.

```python
# An example of training only the complete output head.
model.requires_grad_(False)
model.lm_head.weight.requires_grad_(True)

trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
print("trainable parameters:", sum(p.numel() for p in trainable.values()))
```

Common names include `embed.weight`, `lm_head.weight`, `blocks.N.attn` (Llama
and OLMoE), `blocks.N.mixer` (Qwen), `blocks.N.mlp` (dense feedforward), and
`blocks.N.moe` (experts). Inspect `named_parameters()` for exact paths; packed
expert banks are Parameters rather than collections of `nn.Linear` children.
A tied embedding/head remains one Parameter and one trainability decision.

The local models support whole-model LoRA through a one-time recipe:

```python
from workloads.lora import configure_lora, parameter_report
from workloads.mlops import Llama3, Llama3Config

model = Llama3(Llama3Config.numerical())
configure_lora(model, rank=32, alpha=32, head="frozen")
print(parameter_report(model))
# Pass model to the same generic Trainer or planning API used for full training.
```

| Option | Default and behavior |
|---|---|
| `rank`, `alpha` | 32, 32; scale is alpha/rank |
| `factor_dtype` | `"float32"` storage; compute uses the activation dtype |
| `targets` | Attention/mixer linear projections, dense MLPs and routed experts; explicit module globs replace the preset |
| `head` | `"frozen"`, `"lora"`, or `"full"` |
| `shared_experts` | False; opt in to LoRA on shared-expert linear projections |
| `trainable_base` | Empty; parameter globs can select full training alongside LoRA |

The preset supports PyTorch and MLOps Llama3, dense Qwen3.5 and OLMoE, plus
MLOps Qwen3 MoE and Qwen3.5 MoE. Routers, normalization weights, embeddings,
and shared experts remain frozen by default. Expert modules use independent
factors per expert, joint gate/up factors, and separate down factors.
Pure PyTorch expert LoRA retains independent reference math.

Model forward methods contain no LoRA branches. Dense/head replacement and
expert-module selection happen in `workloads.lora`; reusable projections and
conversion live in `mlops.lora`. MLOps heads call the same head module for
logits and bounded loss, so both paths include its factors. State dictionaries
retain original weight names and add `lora_*` parameters. Apply the same LoRA
configuration before strict checkpoint loading.

The text recipe `workloads.recipes.text.models.build_on_meta` also accepts a
`lora` dictionary with these options, so a training config can construct LoRA
before initialization and optimizer creation. There is no dedicated quickstart
`--lora` flag yet. EP full-model conversion is not currently supported by this
recipe: the separately available `QuackMoELoRA` and `TEMoELoRA` require their
own layer construction and distributed validation.

Frozen state remains a forward/backward input when needed. Memory savings come
primarily from omitted gradients and optimizer states; LoRA can add activation
workspace. Compare actual host usage and individual graph-pair tables, rather
than inferring task memory from trainable counts alone.

## Expert parallelism

EP is a constructor choice for **every MLOps MoE architecture**: `OLMoE`,
`Qwen3MoE`, and `Qwen35MoE`. Local experts are the default; the pure-PyTorch
implementations remain independent references. Model dimensions and attention
structure do not change when enabling EP.

```python
from workloads.mlops import OLMoE, OLMoEConfig

config = OLMoEConfig.throughput()
model = OLMoE(config)  # local experts

# After creating the process group, on the rank's selected compute device:
model = OLMoE(
    config,
    ep_group=group,
    token_capacity=32768,
    device=device,
    parameter_device="cpu",
)
```

The same keyword arguments work with `Qwen3MoE(config, ...)` and
`Qwen35MoE(config, ...)`:

| Option | Meaning / default |
|---|---|
| `ep_group` | Existing process group; omitted for local experts |
| `token_capacity` | Create one model-owned buffer for this local microbatch size |
| `buffer` | Borrow one existing buffer instead of supplying `token_capacity` |
| `device` | Rank's compute device; select it explicitly for EP |
| `parameter_device` | Initial logical parameter storage; compute device by default, or `"cpu"` |
| `dtype` | Dense weights and hidden activations; EP requires BF16 and defaults to it |
| `router_dtype` | EP router precision; OLMoE defaults to FP32, Qwen to BF16; either accepts BF16 or FP32 |
| `compute_precision` | EP routed expert compute: `"bf16"` default or `"fp8_current"` |
| `weight_grad_dtype` | EP expert gradients: BF16 default, or FP32 |
| `activation_transport` | EP activation communication: `"bf16"` default, or `"fp8"` with FP8 compute |

With EP, supply exactly one of `token_capacity` and `buffer`. A single buffer
and its expert publication/reduction banks are reused across every model layer;
model parameters remain distinct. Capacity fixes the supported local microbatch
token count. Construct each planning candidate with its own capacity.

Construct after entering the runtime and creating the process group.
Borrowed buffers remain caller-owned. Close compiled callables before
`model.close()`; this releases layer runtimes and any model-owned buffer.
Failed construction releases the resources it created. Ordinary CPU/meta model
construction does not load MoonEP or Quack kernels.

The optional EP setup is shared in
[mlops/_expert_parallel.py](mlops/_expert_parallel.py); it is temporary
construction state, not part of the model or the generic training API. Routing
normalization, shared experts, and auxiliary-loss definitions remain properties
of each architecture. In local mode, ordinary model dtype controls the router
and experts; the EP-specific precision options in this table apply to EP only.

For FP8 training, select `weight_grad_dtype=torch.float32` and choose trainer
masters/gradient precision explicitly. ShadowSpill handles payloads and scales
through its generic [tensor representation contract](../docs/architecture/state-import.md#tensor-representations).
Quack's grouped expert kernels require H100/SM90. Install its optional
backend from the MLOps checkout with
`./scripts/setup_expert_parallel.sh --backend quack --python /path/to/python`.
The standalone Transformer Engine modules also expose `fp8_block`; these model
constructors select QuackMoE and do not expose a TE-backend switch.

`model.expert_parameters()` identifies unique home expert shards. Attention,
routers, shared experts, normalization, embeddings, and output heads are
replicated across the EP group. For the generic distributed trainer, declare:

```python
from shadowspill.pytorch import Distributed

distributed = Distributed(
    group,
    replica_overrides=[(model.expert_parameters(), None)],
    groups={"ep": group},
)
```

The caller defines global loss normalization and any replication across EP
groups. See the [distributed API](../docs/python/api/distributed.md). Model
implementations contain no ShadowSpill imports.

## Inputs, outputs, and loss

| Interface | Meaning |
|---|---|
| `model(tokens, sequence_lengths=None)` | Logits with final dimension `vocab_size` |
| `model.hidden(tokens, sequence_lengths=None)` | Hidden activations; MoE models also return auxiliary loss |
| `model.loss(tokens, targets, seq_lens=None, reduction=...)` | Language-model objective; `sum` is useful for explicit whole-step normalization |
| Packed sequence metadata | Describes document boundaries within the token tensor; see the text recipe |
| `return_metrics=True` | Additional summaries on supported MLOps model loss/hidden methods; the returned structure is model-specific |

Tokens/targets are integer tensors; target `-100` is ignored. The recipe controls
sequence length, tokens per update, microbatch candidates, schedules, evaluation,
and checkpointing. Those are not architecture dimensions. Run data must use a
compatible tokenizer and IDs below the configured vocabulary size.

## Construction and runnable recipes

Customize a preset without editing the model definition:

```python
from dataclasses import replace
from workloads.mlops import Llama3, Llama3Config

config = replace(Llama3Config.numerical(), n_layers=4, max_seq_len=2048)
model = Llama3(config)  # ordinary initialized PyTorch module
```

Inspect a large architecture without allocating its weights:

```python
import torch
from workloads.mlops import Qwen35MoE, Qwen35MoEConfig

with torch.device("meta"):
    model = Qwen35MoE(Qwen35MoEConfig())
print(sum(p.numel() for p in model.parameters()))  # 34,660,610,688
```

A saved text recipe specifies the constructor, shape and dtype explicitly:

```json
{
  "model": {
    "@call": "workloads.recipes.text.models:build_on_meta",
    "model": "@workloads.mlops:Llama3",
    "dtype": "bfloat16",
    "config": {
      "@call": "workloads.recipes.text.models:config_preset",
      "preset": "@workloads.mlops:Llama3Config.numerical",
      "max_seq_len": 2048
    }
  }
}
```

This is the model portion of a request. Complete runnable examples are
[Llama](../training/configs/llama3_1b.json),
[dense Qwen](../training/configs/qwen35_1b.json), and
[OLMoE](../training/configs/olmoe_1b.json). The
[training guide](../training/README.md) explains data preparation, overrides,
optimizer schedules, evaluation, W&B, and checkpoints.

For a benchmark using a supplied throughput preset and saved resolution plans:

```bash
python -m benchmarking.quickstart mlops_llama3 \
  --sequence-length 1024 --sequences-per-step 64 \
  --min-tokens-per-microbatch 8192 --max-tokens-per-microbatch 16384 \
  --search-budget-gib 20,30 --run-budget-gib 20,30 \
  --spill-gib 112 --plots --resolution-plans
```

The [quickstart guide](../benchmarking/quickstart.md) describes every flag and
custom experiment factories. A factory supplies custom model shapes or EP
resources. Text `--distributed` presets use replicated data parallelism.

## Source organization

| Path | Responsibility |
|---|---|
| [pytorch/](pytorch/) | Readable model definitions and shared dense/OLMoE shape dataclasses |
| [mlops/](mlops/) | Corresponding definitions using MLOps operations |
| [mlops/olmoe.py](mlops/olmoe.py) | OLMoE architecture, with local or EP experts |
| [mlops/qwen3_moe.py](mlops/qwen3_moe.py) | Qwen 3 MoE public model and configuration |
| [mlops/qwen35_moe.py](mlops/qwen35_moe.py) | Qwen 3.5 MoE public model and configuration |
| [mlops/_qwen_moe/](mlops/_qwen_moe/) | Private initialization, mixers, experts, and decoder code shared by the two Qwen families |
| [mlops/_expert_parallel.py](mlops/_expert_parallel.py) | Shared EP construction, ownership and routing helpers for all MLOps MoE models |
| [common/](common/) | Shared reference math, rotary tables, packing metadata and loss utilities |
| [recipes/](recipes/) | Caller-side model construction, objectives and data composition |
| [precision.py](precision.py) | Supplied workload precision configuration |
| [full_model.py](full_model.py) | Reproducible full-model specifications and construction |
| [training/configs/](../training/configs/) | Concrete example training requests |

The extra private Qwen files factor code shared by two architectures, including
DeltaNet and full attention. Public family modules remain small; adding EP does
not require copying the decoder or changing the trainer/planner.

Reusable LoRA modules belong in MLOps. Model-family target selection belongs
with the examples. The generic trainer and planner operate on the resulting
PyTorch model and never import this catalog or the workload packages.
