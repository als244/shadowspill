# OLMoE

GQA attention followed by a routed SwiGLU expert block. The router takes a
softmax over all experts and selects top-k without renormalizing the selected
weights. Each expert has a packed gate/up projection and a down projection.
The configuration contains no shared expert.

## Shapes

[config.py](config.py) holds `Config` and `PRESETS`, the named shapes the
[catalog](../catalog/) serves; the
[catalog table](../../MODELS.md#preset-dimensions-and-parameter-counts) lists
their widths and parameter counts.

| Catalog name | Shape |
|---|---|
| `olmoe-numerical` | the exact-scale shape the numerical gate runs |
| `olmoe-tiny` | a two-layer smoke-test shape |
| `olmoe-7b-a1b` | OLMoE-1B-7B |

## Configuration

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

## Implementations

[implementations/pytorch.py](implementations/pytorch.py) is the readable
numerical authority, with its own expert math;
[implementations/mlops.py](implementations/mlops.py) is its
state-dictionary-compatible twin on the operation library, with local experts
by default or experts sharded over an expert-parallel group through the
constructor arguments the
[model catalog](../../MODELS.md#expert-parallelism) describes. Both expose
`Model`.

## Checkpoint

`checkpoint.py` reads a local Hugging Face OLMoE checkpoint:
`OLMoECheckpoint(directory)` turns its `config.json` into this family's
`Config` through `config.from_huggingface` and `load_into(model)` writes
every parameter once, through a plain tensor or this rank's rows of an FSDP2
shard. The router is the checkpoint's `mlp.gate`; the expert banks are
stacked from the per-expert projections, `w13_experts` from each expert's
gate projection followed by its up projection and `w2_experts` from its down
projection.

## Construction

```python
from workloads.models.olmoe.config import PRESETS
from workloads.models.olmoe.implementations.mlops import Model

model = Model(PRESETS["7b-a1b"])
```

`workloads.models.catalog.build_on_meta("olmoe-7b-a1b", "mlops", dtype="bfloat16")`
declares the same model on `meta`, and the quickstart runs it with
`python -m benchmarking.quickstart.cli --model olmoe-7b-a1b`. The
[model catalog](../../MODELS.md) covers dtypes, LoRA, inputs and the loss.
