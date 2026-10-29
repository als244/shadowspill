# Example model catalog

Start here to choose a supplied architecture, change its dimensions, and select
training precision. These are ordinary PyTorch modules passed into the generic
[Trainer](../docs/python/api/training.md), `Forward`, or planning APIs. Model
construction, trainable parameters, and data semantics belong to the caller.

## Contents

- [Architecture index](#architecture-index)
- [Preset dimensions and parameter counts](#preset-dimensions-and-parameter-counts)
- [Families](#families)
- [Dtypes and optimizer state](#dtypes-and-optimizer-state)
- [Trainable parameters and LoRA](#trainable-parameters-and-lora)
- [Expert parallelism](#expert-parallelism)
- [Inputs, outputs, and loss](#inputs-outputs-and-loss)
- [Construction and runnable recipes](#construction-and-runnable-recipes)
- [Source organization](#source-organization)

## Architecture index

A catalog name is `<family>-<size>`: `qwen35-9b` is the **dense** Qwen 3.5
architecture at Qwen3.5-9B's shape and `qwen35-27b` the same architecture at
Qwen3.5-27B's; `qwen3-8b` is **dense Qwen 3** at Qwen3-8B's shape,
`qwen3moe-30b-a3b` **Qwen 3 with MoE** and `qwen35moe-35b-a3b` **Qwen 3.5
with MoE**. The implementation is chosen beside the name (`--implementation
pytorch|mlops`, `mlops` by default), and dimensions stay configurable: a name
is a starting shape, not a constraint.

| Architecture | Family package | Implementations | Catalog names |
|---|---|---|---|
| [Llama 3](models/llama3/README.md) | `workloads.models.llama3` | `pytorch`, `mlops` | `llama3-numerical`, `llama3-tiny`, `llama3-1b`, `llama3-8b` |
| [Dense Qwen 3](models/qwen3/README.md) | `workloads.models.qwen3` | `pytorch`, `mlops` | `qwen3-tiny`, `qwen3-1.7b`, `qwen3-8b`, `qwen3-32b` |
| [Dense Qwen 3.5](models/qwen35/README.md) | `workloads.models.qwen35` | `pytorch`, `mlops` | `qwen35-numerical`, `qwen35-tiny`, `qwen35-2b`, `qwen35-9b`, `qwen35-27b` |
| [OLMoE](models/olmoe/README.md) | `workloads.models.olmoe` | `pytorch`, `mlops`; optional EP in MLOps | `olmoe-numerical`, `olmoe-tiny`, `olmoe-7b-a1b` |
| [Qwen 3 MoE](models/qwen3moe/README.md) | `workloads.models.qwen3moe` | `mlops`; optional EP constructor | `qwen3moe-tiny`, `qwen3moe-30b-a3b` |
| [Qwen 3.5 MoE](models/qwen35moe/README.md) | `workloads.models.qwen35moe` | `mlops`; optional EP constructor | `qwen35moe-tiny`, `qwen35moe-35b-a3b` |
| [GLM-5.3](models/glm53/README.md) and [GLM-5.3-Flash](models/glm53_flash/README.md) | `workloads.models.glm53`, `workloads.models.glm53_flash` | `mlops`, through each family's quickstart factory | `glm53`, `glm53-tiny`, `glm53-flash`, `glm53-flash-tiny` |

A family package holds `Config` in its `config.py` and one module per
implementation under `implementations/`, each exposing `Model`. Pure PyTorch
implementations provide readable numerical references. MLOps
implementations use the separately installed operation library. All three
MLOps MoE architectures can enable EP through the same Python constructor
arguments. An experiment factory supplies groups and resources; a catalog
name alone does not enable EP.

For a non-text example, see the [regression recipe](recipes/regression/benchmark.py) and
[generic training walkthrough](../docs/examples/generic-training.md).

## Preset dimensions and parameter counts

Each family's `config.py` holds `PRESETS`, its named shapes by size, and the
[catalog](models/catalog/) names every one `<family>-<size>`: a `numerical`
shape is the small exact-scale one the numerical gate runs, `tiny` a smoke-test
shape, and a published size is that checkpoint's text decoder (`qwen35-27b`
and `qwen35-2b` without their multi-token prediction layer). The GLM
families' shapes come from their checkpoints; see their READMEs under
[Families](#families).

Counts below include embeddings and the output head, count shared Parameter
objects once, and exclude buffers and optimizer state. They describe the whole
unsharded model, and equal its trainable count under default full training.
They are **total parameters**, not the subset used by one routed token.

| Catalog name | Layers | Model width | Q / KV heads | Head width | MLP or expert width | Experts / top-k | Vocabulary | Total parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3-numerical` | 12 | 2,048 | 16 / 4 | 128 | 7,168 | — | 128,256 | 1,179,699,200 |
| `llama3-tiny` | 2 | 128 | 4 / 2 | 32 | 256 | — | 512 | 426,624 |
| `llama3-1b` | 16 | 2,048 | 32 / 8 | 64 | 8,192 | — | 128,256 | 1,235,814,400 |
| `llama3-8b` | 32 | 4,096 | 32 / 8 | 128 | 14,336 | — | 128,256 | 8,030,261,248 |
| `qwen3-tiny` | 2 | 128 | 4 / 2 | 64 | 256 | — | 512 | 525,184 |
| `qwen3-1.7b` | 28 | 2,048 | 16 / 8 | 128 | 6,144 | — | 151,936 | 1,720,574,976 |
| `qwen3-8b` | 36 | 4,096 | 32 / 8 | 128 | 12,288 | — | 151,936 | 8,190,735,360 |
| `qwen3-32b` | 64 | 5,120 | 64 / 8 | 128 | 25,600 | — | 151,936 | 32,762,123,264 |
| `qwen35-numerical` | 8 | 1,536 | 12 / 4 | 128 | 4,608 | — | 248,320 | 1,006,955,408 |
| `qwen35-tiny` | 2 | 128 | 4 / 2 | 32 | 256 | — | 512 | 461,544 |
| `qwen35-2b` | 24 | 2,048 | 8 / 2 | 256 | 6,144 | — | 248,320 | 1,881,825,088 |
| `qwen35-9b` | 32 | 4,096 | 16 / 4 | 256 | 12,288 | — | 248,320 | 8,953,803,264 |
| `qwen35-27b` | 64 | 5,120 | 24 / 4 | 256 | 17,408 | — | 248,320 | 26,895,998,464 |
| `olmoe-numerical` | 16 | 1,024 | 8 / 8 | 128 | 1,024 | 16 / 4 | 50,304 | 975,766,528 |
| `olmoe-tiny` | 2 | 128 | 4 / 2 | 32 | 128 | 8 / 2 | 512 | 1,018,880 |
| `olmoe-7b-a1b` | 16 | 2,048 | 16 / 16 | 128 | 1,024 | 64 / 8 | 50,304 | 6,919,161,856 |
| `qwen3moe-tiny` | 2 | 128 | 4 / 2 | 32 | 128 | 8 / 2 | 1,024 | 1,149,696 |
| `qwen3moe-30b-a3b` | 48 | 2,048 | 32 / 4 | 128 | 768 | 128 / 8 | 151,936 | 30,532,122,624 |
| `qwen35moe-tiny` | 2 | 128 | 4 / 2 | 32 | 128 routed + 128 shared | 8 / 2 | 1,024 | 1,283,048 |
| `qwen35moe-35b-a3b` | 40 | 2,048 | 16 / 2 | 256 | 512 routed + 512 shared | 256 / 8 | 248,320 | 34,660,610,688 |

Dense Qwen 3 has full attention in every block; in `qwen3-tiny` and
`qwen3-32b` its query heads together are wider than the model. The dense
Qwen 3.5 shapes use a full-attention block every fourth layer (every second
in `qwen35-tiny`). Their DeltaNet key/value head counts are 6/12
(`qwen35-numerical`), 2/4 (`qwen35-tiny`), 16/16 (`qwen35-2b`), 16/32
(`qwen35-9b`) and 16/48 (`qwen35-27b`), with 128-wide heads (32 in `tiny`)
and a width-4 convolution. `qwen35moe-35b-a3b` uses 16/32 DeltaNet heads of
width 128 and the same convolution width; `qwen35moe-tiny` 2/4 of width 32.

Qualification may select smaller shapes and FP16 precision on devices below
SM80; that policy lives in [qualification](../qualification/device_defaults.py),
not in the models. Parameter counts are independent of dtype. Physical storage
also depends on dtype, tensor representations, sharding, and runtime resources.

## Families

Each family package under [models/](models/) has a README with its
architecture, shapes, configuration fields, implementations and construction,
and a checkpoint reader (`checkpoint.py`, a local Hugging Face checkpoint into
the model, parameter by parameter, through `config.from_huggingface` and
`load_into`, or chunk by chunk through `copy_chunk`). The catalog names each
reader: `catalog.build_from_checkpoint(family, directory)` declares a family's
model at a checkpoint's configuration, and
`workloads.recipes.text.initialization.from_checkpoint` streams the checkpoint
into a run, so a fine-tune is a config naming a family and a directory:

| Family | What it is |
|---|---|
| [Llama 3](models/llama3/README.md) | GQA attention and a dense SwiGLU MLP in every block; `pytorch` and `mlops` |
| [Dense Qwen 3](models/qwen3/README.md) | GQA attention with per-head query and key norms and a dense SwiGLU MLP in every block; `pytorch` and `mlops` |
| [Dense Qwen 3.5](models/qwen35/README.md) | A repeating hybrid of Gated DeltaNet and gated GQA with dense MLPs; `pytorch` and `mlops` |
| [OLMoE](models/olmoe/README.md) | GQA attention and routed SwiGLU experts under a softmax router; `pytorch` and `mlops`, optional EP |
| [Qwen 3 MoE](models/qwen3moe/README.md) | GQA attention and routed experts in every block; `mlops`, optional EP |
| [Qwen 3.5 MoE](models/qwen35moe/README.md) | The Qwen 3.5 hybrid with routed and gated shared experts; `mlops`, optional EP |
| [GLM-5.3](models/glm53/README.md) | 78 layers of positional sparse MLA with shared indices and 256-expert MoE, LoRA over a local checkpoint streamed into the pool; `mlops` |
| [GLM-5.3-Flash](models/glm53_flash/README.md) | 45 layers of KDA and NoPE sparse MLA on mHC streams with 288-expert MoE and an optional vision encoder, LoRA over a local checkpoint streamed into the pool; `mlops` |

The two GLM families share their building blocks, checkpoint importer, stage
partition and LoRA recipe through [models/shared/glm/](models/shared/glm/);
their catalog entries resolve to the quickstart factories, so
`--model glm53-flash` with `--factory-args` runs one.

## Dtypes and optimizer state

Shape dataclasses describe architecture. Precision is chosen during model
construction and when building the trainer/optimizer. Internal reductions may
use FP32 even when stored weights and activations use BF16 or FP16.

| Setting | Python / recipe location | Catalog factory `precision` field | Default / choices |
|---|---|---|---|
| Model weights | Construct with the desired PyTorch dtype; recipe `model.dtype` | `model_dtype` | Quickstart: `bfloat16`; also `float16`, `float32` |
| Optimizer masters | Builder `master_dtype` | `master_dtype` | No masters; or a floating dtype |
| Accumulated gradients | Builder `grad_dtype` | `grad_dtype` | Parameter dtype; or a floating dtype |
| Optimizer moments | `mlops.optim.AdamW(opt_state_dtype=...)` / `optimizer_args.opt_state_dtype` | `opt_state_dtype` | MLOps: `bfloat16`; also `float16`, `float32`, or `parameter` |
| Parameter rounding | Optimizer `parameter_rounding` | `parameter_rounding` | `nearest` or `stochastic` |
| Moment rounding | Optimizer `opt_state_rounding` | `opt_state_rounding` | `nearest` or `stochastic`; ShadowSpill defaults to stochastic for BF16 AdamW moments, nearest otherwise |

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
`precision.model_dtype`. Selecting FP8 experts does not convert attention, router,
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

Common names include `embed.weight`, `lm_head.weight`, `blocks.N.attn` (Llama,
dense Qwen 3 and OLMoE), `blocks.N.mixer` (Qwen 3.5 and the Qwen mixtures),
`blocks.N.mlp` (dense feedforward), and `blocks.N.moe` (experts). Inspect
`named_parameters()` for exact paths; packed expert banks are Parameters
rather than collections of `nn.Linear` children. A tied embedding/head
remains one Parameter and one trainability decision.

The local models support whole-model LoRA through a one-time recipe:

```python
from workloads.lora import configure_lora, parameter_report
from workloads.models.llama3.config import PRESETS
from workloads.models.llama3.implementations.mlops import Model

model = Model(PRESETS["numerical"])
configure_lora(model, rank=32, alpha=32, head="frozen")
print(parameter_report(model))
# Pass model to the same generic Trainer or planning API used for full training.
```

| Option | Default and behavior |
|---|---|
| `rank`, `alpha` | 16, 16; scale is alpha/rank |
| `factor_dtype` | `"bfloat16"` storage; compute uses the activation dtype |
| `targets` | Attention/mixer linear projections, dense MLPs and routed experts; explicit module globs replace the preset |
| `head` | `"frozen"`, `"lora"`, or `"full"` |
| `shared_experts` | False; opt in to LoRA on shared-expert linear projections |
| `trainable_base` | Empty; parameter globs can select full training alongside LoRA |

The preset supports PyTorch and MLOps Llama3, dense Qwen3, dense Qwen3.5 and
OLMoE, plus MLOps Qwen3 MoE and Qwen3.5 MoE. Routers, normalization weights,
embeddings, and shared experts remain frozen by default. Expert modules use
independent factors per expert, joint gate/up factors, and separate down
factors. Pure PyTorch expert LoRA retains independent reference math.

Dense projections and heads take LoRA by replacement: `workloads.lora` selects
them and `mlops.lora` converts them, so their forwards have no LoRA branch.
Routed experts take it in place, one path for every MoE family
(`workloads.models.shared.routed_lora`): a MoE module declares its expert
banks' shapes as `expert_shapes`, `add_factors` registers the factors beside
the banks, sized from that declaration rather than from the weights, and the
module's forward runs its experts with them when present (`expert_lora`,
mlops's in the MLOps implementations and the module's ordinary-PyTorch
reference in the pytorch one). Because nothing reads an expert bank's
shape, a caller that partitions parameters as their modules are
constructed builds LoRA models too. MLOps heads call the same head module for
logits and bounded loss, so both paths include its factors. State dictionaries
retain original weight names and add `lora_*` parameters. Apply the same LoRA
configuration before strict checkpoint loading.

`workloads.models.catalog.build_on_meta` and the generic
`workloads.models.construction.build_on_meta` also accept a `lora` dictionary
with these options, so a training config can construct LoRA before
initialization and optimizer creation; the quickstart's `--lora` flags do the
same for a catalog model. EP full-model conversion is not currently supported
by this recipe: the separately available `QuackMoELoRA` and `TEMoELoRA` require their
own layer construction and distributed validation.

Frozen state remains a forward/backward input when needed. Memory savings come
primarily from omitted gradients and optimizer states; LoRA can add activation
workspace. Compare actual host usage and individual graph-pair tables, rather
than inferring task memory from trainable counts alone.

## Expert parallelism

EP is a constructor choice for **every MLOps MoE architecture**: OLMoE,
Qwen 3 MoE and Qwen 3.5 MoE. Local experts are the default; the pure-PyTorch
implementations remain independent references. Model dimensions and attention
structure do not change when enabling EP.

```python
from workloads.models.olmoe.config import PRESETS
from workloads.models.olmoe.implementations.mlops import Model

config = PRESETS["7b-a1b"]
model = Model(config)  # local experts

# After creating the process group, on the rank's selected compute device:
model = Model(
    config,
    ep_group=group,
    token_capacity=32768,
    device=device,
    parameter_device="cpu",
)
```

The same keyword arguments work with the two Qwen MoE families' `Model`:

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
| `expert_backend` | Qwen EP experts: `"quack"` default, or `"te"` for Transformer Engine's TEMoE with BF16 compute and activation transport; OLMoE always uses QuackMoE |
| `expert_factory` | Alternative callable constructing expert leaves; portable layers need no live group or buffer during model construction |

With EP, supply exactly one of `token_capacity` and `buffer`. A single buffer
and its expert publication/reduction banks are reused across every model layer;
model parameters remain distinct. Capacity fixes the supported local microbatch
token count. Construct each planning candidate with its own capacity.

The live-group form requires an existing group and device. Borrowed buffers
remain caller-owned. Close compiled callables before
`model.close()`; this releases layer runtimes and any model-owned buffer.
Failed construction releases the resources it created. Ordinary CPU/meta model
construction does not load MoonEP or Quack kernels.

The optional EP setup is shared in
[models/shared/expert_parallel](models/shared/expert_parallel/); it is temporary
construction state, not part of the model or the generic training API. Routing
normalization, shared experts, and auxiliary-loss definitions remain properties
of each architecture. In local mode, ordinary model dtype controls the router
and experts; the EP-specific precision options in this table apply to EP only.

For FP8 training, select `weight_grad_dtype=torch.float32` and choose trainer
masters/gradient precision explicitly. ShadowSpill handles payloads and scales
through its generic [tensor representation contract](../docs/architecture/state-import.md#tensor-representations).
Quack's grouped expert kernels require H100/SM90. Install its optional
backend with this repository's `./scripts/setup.sh --expert-parallel quack`, or
add it to an existing environment from the MLOps checkout with
`./scripts/setup_expert_parallel.sh --backend quack --python /path/to/python`.
The standalone Transformer Engine modules also expose `fp8_block`, which these
constructors do not use.

`model.expert_parameters()` identifies unique home expert shards. Attention,
routers, shared experts, normalization, embeddings, and output heads are
replicated across the EP group. Saved-phase callers use `expert_factory` to
construct portable leaves and pass their provider descriptions through the
builders' generic `resources` argument. The central `Parallelism` specifies
expert-bank names as `Sharded` and dense parameters as replicated, with separate
storage settings. The [EP recipe](recipes/text/expert_parallel/README.md) composes
these declarations without creating a communicator in the model.

The caller defines global loss normalization and any replication across EP
groups. See the [distributed API](../docs/python/api/distributed.md). Model
implementations contain no ShadowSpill imports. The portable resource path still
needs its own distributed qualification; existing live-module checks do not
qualify a saved Plan.

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
from workloads.models.llama3.config import PRESETS
from workloads.models.llama3.implementations.mlops import Model

config = replace(PRESETS["numerical"], n_layers=4, max_seq_len=2048)
model = Model(config)  # ordinary initialized PyTorch module
```

Inspect a large architecture without allocating its weights:

```python
import torch
from workloads.models.qwen35moe.config import Config
from workloads.models.qwen35moe.implementations.mlops import Model

with torch.device("meta"):
    model = Model(Config())
print(sum(p.numel() for p in model.parameters()))  # 34,660,610,688
```

A saved text recipe names a catalog model, its implementation and dtype, and
any configuration field it changes:

```json
{
  "model": {
    "@call": "workloads.models.catalog:build_on_meta",
    "model": "llama3-numerical",
    "implementation": "mlops",
    "dtype": "bfloat16",
    "max_seq_len": 2048
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
python -m benchmarking.quickstart \
  --factory workloads.recipes.text.benchmark.factory:experiment \
  --factory-args '{"model":"llama3-8b","sequence_length":1024,"sequences_per_step":64,"min_tokens_per_example":8192,"max_tokens_per_example":16384}' \
  --search-budget-gib 20 30 --run-budget-gib 20 30 \
  --spill-gib 112 --output-dir runs/llama3
```

The [quickstart guide](../benchmarking/quickstart.md) describes every flag and
custom experiment factories. The [text factory](recipes/text/benchmark/README.md)
supplies catalog models and optional LoRA; `ranks` selects sharded DP. The GLM
families use the same phase runner with family-specific construction and loading.
Model-owned EP resources use the same phase runner; their build and profiling
workers recreate the explicitly described resources.

## Source organization

| Path | Responsibility |
|---|---|
| [models/](models/) | One package per family: `config.py`, its `checkpoint.py` reading a local Hugging Face checkpoint, a README and `implementations/` with one module per implementation |
| [models/shared/](models/shared/) | Code families share: reference decoder math, rotary tables, packing metadata and loss helpers, the mlops norm containers, the Gated DeltaNet mixer, EP construction, the Qwen mixture blocks and the GLM building blocks |
| [models/shared/qwen_moe/](models/shared/qwen_moe/) | Initialization, mixers, experts and the decoder shared by the two Qwen MoE families |
| [models/shared/expert_parallel](models/shared/expert_parallel/) | Shared EP construction and routing helpers for all MLOps MoE models |
| [lora/](lora/) | Model-family LoRA targeting over the reusable projections in MLOps |
| [recipes/](recipes/) | Caller-side model construction, objectives and data composition |
| [precision.py](precision.py) | Supplied workload precision configuration |
| [full_model.py](full_model.py) | Reproducible full-model specifications and construction |
| [training/configs/](../training/configs/) | Concrete example training requests |

The shared Qwen package factors code two architectures use, including
DeltaNet and full attention. Family packages remain small; adding EP does
not require copying the decoder or changing the trainer/planner.

Reusable LoRA modules belong in MLOps. Model-family target selection belongs
with the examples. The generic trainer and planner operate on the resulting
PyTorch model and never import this catalog or the workload packages.


## Family LoRA targets

Each family declares its default module globs in `models/<family>/lora.py`.
Both implementations expose the same `LORA_TARGETS`; `configure_lora(targets=...)`
can override them explicitly. New linear modules are not automatically adapted
by a family preset. The head remains frozen unless its existing option selects it.

| Family | Default adapted projections | Frozen controls |
| --- | --- | --- |
| Llama3, Qwen3 | Q/K/V/O and dense SwiGLU projections | Norms and other non-projection state |
| Qwen3.5 dense | Full-attention Q/K/V/O, packed DeltaNet QKV+Z/out, dense SwiGLU | DeltaNet `w_ba`, decay state, convolution |
| OLMoE, Qwen3 MoE | Q/K/V/O and routed expert factors | Router; shared experts by default |
| Qwen3.5 MoE | Full-attention Q/K/V/O, DeltaNet QKV+Z/out, routed expert factors | `w_ba`, router, shared mixing gate |
| GLM-5.3 | MLA Q/KV/O projections, dense MLP, routed expert factors | Indexer, router, shared experts |
| GLM-5.3-Flash | KDA Q/K/V/O, MLA Q/KV/O, dense MLP, routed expert factors | Five KDA gate projections, indexer, router, mHC, convolution, shared experts |

For GLM Flash the excluded KDA projections are `forget_gate.f_a_proj`,
`forget_gate.f_b_proj`, `b_proj`, `g_a_proj`, and `g_b_proj`. The large MLP
`gate_proj` remains a main feed-forward target. Existing optional vision
projection targets are explicitly listed in the Flash preset.

Qwen3.5 retains its packed QKV+Z projection: adapting `in_proj_z` alongside
`in_proj_qkv` and `out_proj` follows the
[Axolotl Qwen3.5 recipe](https://docs.axolotl.ai/docs/models/qwen3.5.html#gated-deltanet-linear-attention).
The tiny beta/decay projections (`w_ba`) are excluded. An explicit
`shared_experts=True` selects only the family's `SHARED_TARGETS` (large
shared-expert projections), not its scalar mixing gate or router.

These preset changes alter adapter parameter sets. An adapter checkpoint created
with the previous all-linear defaults may contain gate factors that are no
longer present; select the old targets explicitly when continuing that run.
