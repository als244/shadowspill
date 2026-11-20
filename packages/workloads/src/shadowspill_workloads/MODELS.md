# Model architectures

For training, start with the [built-in catalog](../../../../docs/python/catalog.md)
and [client guide](../../../../docs/python/client.md). This page describes the
model implementations and their shapes. All families use the same public
registration and training APIs.

## Architecture index

| Family | Architecture | Implementations |
| --- | --- | --- |
| [Llama 3](models/llama3/README.md) | GQA and dense SwiGLU | `pytorch`, `mlops` |
| [Qwen 3](models/qwen3/README.md) | GQA with Q/K norms and dense SwiGLU | `pytorch`, `mlops` |
| [Qwen 3.5](models/qwen35/README.md) | Gated DeltaNet and gated GQA, dense MLPs | `pytorch`, `mlops` |
| [OLMoE](models/olmoe/README.md) | GQA and routed SwiGLU experts | `pytorch`, `mlops` |
| [Qwen 3 MoE](models/qwen3moe/README.md) | GQA and routed experts | `mlops` |
| [Qwen 3.5 MoE](models/qwen35moe/README.md) | Hybrid attention, routed and gated shared experts | `mlops` |
| [GLM-5.3](models/glm53/README.md) | Positional sparse MLA and routed/shared experts | `mlops` |
| [GLM-5.3-Flash](models/glm53_flash/README.md) | KDA and NoPE sparse MLA, mHC streams and routed/shared experts | `mlops` |

A family owns its configuration, computation, checkpoint mapping, LoRA targets
and registration. Shared layers live in `models/shared/`; reusable setup,
initialization and policies live outside the model implementations. The
[contributor guide](../../../../docs/development/workloads.md) describes this
structure and the steps for adding a family. The
[regression model](models/regression) is a small non-text example.

## Preset dimensions and parameter counts

Each family's `config.py` holds `PRESETS`, its named shapes by size, and the
[catalog](models/catalog) names every one `<family>-<size>`: a `numerical`
shape is the small exact-scale one the numerical gate runs, `tiny` a smoke-test
shape, and a published size is that checkpoint's text decoder (every Qwen 3.5
size without its multi-token prediction layer). For checkpoint-derived shapes, the reader uses that checkpoint's configuration.

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

Parameter counts are independent of dtype. Physical storage also depends on
dtype, tensor representations, sharding, and runtime resources.

## Construction

Direct users of the [phase APIs](../../../../docs/python/phases.md) can construct
an ordinary module from a family's `Config`, or declare one on `meta`:

```python
from shadowspill_workloads.models.catalog import build_on_meta

model = build_on_meta("qwen35-9b", dtype="bfloat16")
```

This declares model structure. The public client's `Model.from_pretrained(...)`
resolves a checkpoint and uses the registered bounded initializer during worker
startup. Training checkpoints preserve mutable training state separately from
the frozen base; see [saved Plans and checkpoints](../../../../docs/python/advanced/saved-plans.md).

## Precision and LoRA

Choose trainable parameters, LoRA rank/targets and optimizer in the
[client configuration](../../../../docs/python/catalog.md). Family `lora.py`
files supply the default adapter targets. Explicit targets replace those defaults;
base weights stay frozen when LoRA is enabled.

Parameter storage overrides, gradient/master/state dtypes and stochastic rounding
are described in [precision](../../../../docs/python/advanced/precision.md).
Family pages document architecture-specific defaults such as Qwen3.5's FP32 decay
parameters and GLM's compressed expert weights.

## Expert parallelism

Select DP or EP during client preparation; runtime commands still submit one
global batch. The server supplies worker resources and communication bindings.
See the [parallelism reference](../../../../docs/reference/engine/parallelism.rst) for public
configuration and direct-Executor contracts.

The MoE implementations accept an expert factory for sharded execution. Shared
expert construction lives in `models/shared/expert_parallel/`; routing and
architecture-specific expert math stay with the model. MLOps owns the distributed
expert kernels and provider dispatch.

## Inputs, outputs and objectives

Public batches use `Datum` records and the model's declared input schema. Shared
packing validates boundaries and scheduled padding before submission. Use the
[objective guide](../../../../docs/python/advanced/objectives.md) for cross entropy,
masked fine-tuning, balancing, policy losses and distillation.

Direct model calls accept tensors and packed sequence metadata. Their family
pages describe any additional inputs. Training uses the shared chunked head
objectives; chunking limits temporary logits without changing token weighting.
