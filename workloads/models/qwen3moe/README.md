# Qwen 3 MoE

GQA attention followed by MoE in **every** block. The selected routing
probabilities sum to one. The default shape follows the published
[Qwen3-30B-A3B configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json).

## Shapes

[config.py](config.py) holds `Config` and `PRESETS`, the named shapes the
[catalog](../catalog/) serves; the
[catalog table](../../MODELS.md#preset-dimensions-and-parameter-counts) lists
their widths and parameter counts.

| Catalog name | Shape |
|---|---|
| `qwen3moe-tiny` | a two-layer smoke-test shape with eight experts |
| `qwen3moe-30b-a3b` | Qwen3-30B-A3B |

## Configuration

`Config` shares its fields with Qwen 3.5 MoE through
[shared/qwen_moe/config.py](../shared/qwen_moe/config.py); each family's
`Config` supplies its published defaults:

| Options | Qwen 3 MoE default | Qwen 3.5 MoE default |
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

## Implementations

[implementations/mlops.py](implementations/mlops.py) exposes `Model`, built
from the decoder, mixers and experts in
[shared/qwen_moe/](../shared/qwen_moe/) that both Qwen mixture families use,
with local experts by default or experts sharded over an expert-parallel group
through the constructor arguments the
[model catalog](../../MODELS.md#expert-parallelism) describes. There is no
pure-PyTorch implementation.

## Checkpoint

`checkpoint.py` reads a local Hugging Face Qwen 3 MoE checkpoint:
`Qwen3MoECheckpoint(directory)` turns its `config.json` into this family's
`Config` through `config.from_huggingface` and `load_into(model)` writes
every parameter once, through a plain tensor or this rank's rows of an FSDP2
shard. The expert banks are stacked from the per-expert projections as
OLMoE's are, `w13` from each expert's gate then up projection and `w2` from
its down projection; the router is `mlp.gate`.

## Construction

```python
from workloads.models.qwen3moe.config import PRESETS
from workloads.models.qwen3moe.implementations.mlops import Model

model = Model(PRESETS["30b-a3b"])
```

`workloads.models.catalog.build_on_meta("qwen3moe-30b-a3b", dtype="bfloat16")`
declares the same model on `meta`, and the quickstart runs it with
`python -m benchmarking.quickstart --model qwen3moe-30b-a3b`. The
[model catalog](../../MODELS.md) covers dtypes, LoRA, inputs and the loss.
