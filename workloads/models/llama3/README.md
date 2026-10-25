# Llama 3

GQA attention and a dense SwiGLU MLP in every decoder block, with RMSNorm and
RoPE. Embedding and output-head weights are separate unless `tied_embeddings`
shares them, as Llama-3.2-1B does.

## Shapes

[config.py](config.py) holds `Config` and `PRESETS`, the named shapes the
[catalog](../catalog/) serves; the
[catalog table](../../MODELS.md#preset-dimensions-and-parameter-counts) lists
their widths and parameter counts.

| Catalog name | Shape |
|---|---|
| `llama3-numerical` | the exact-scale shape the numerical gate runs |
| `llama3-tiny` | a two-layer smoke-test shape |
| `llama3-1b` | Llama-3.2-1B's decoder, with tied embeddings |
| `llama3-8b` | Llama-3.1-8B's decoder |

`llama3-1b` and `llama3-8b` scale their rotary frequencies as Llama 3.2 and
Llama 3.1 do past the 8,192-token context their base models were trained at
(`rope_scaling`): a wavelength beyond that context divided by
`low_freq_factor` is slowed by `factor`, one within the context divided by
`high_freq_factor` is kept, and the band between is blended.
`llama3-numerical` and `llama3-tiny` apply no scaling.

## Configuration

| Option | Meaning / default |
|---|---|
| `n_layers`, `d_model` | Number of decoder blocks and residual width |
| `n_heads`, `n_kv_heads` | Query heads and grouped key/value heads; query heads must divide into KV groups |
| `d_ff` | Intermediate width of each dense SwiGLU projection |
| `vocab_size` | Embedding/output-head row count; may include padded vocabulary rows |
| `rope_base` | RoPE base; 500,000 |
| `max_seq_len` | Maximum supported sequence length / rotary-table capacity; 131,072 |
| `tied_embeddings` | Share embedding and output-head weights; false |
| `rope_scaling` | Llama 3's rotary frequency scaling (`factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings`), or none; none |
| `norm_epsilon` | Stabilizer of every RMSNorm; 1e-5 |

`head_dim` is derived as `d_model // n_heads`; the division must be exact.
All shape fields above `rope_base` are required for a custom configuration.
With tied embeddings, freezing or training either name affects the same
Parameter.

## Implementations

[implementations/pytorch.py](implementations/pytorch.py) is the readable
numerical authority; [implementations/mlops.py](implementations/mlops.py) is
its state-dictionary-compatible twin on the operation library. Both expose
`Model`.

## Checkpoint

`checkpoint.py` reads a local Hugging Face Llama 3 checkpoint:
`Llama3Checkpoint(directory)` turns its `config.json` into this family's
`Config` through `config.from_huggingface` (the rotary scaling of Llama 3.1
and 3.2 included; `max_seq_len` stays the preset's rotary capacity unless
overridden) and `load_into(model)` writes every parameter once from the
memory-mapped safetensors shards, through a plain tensor or this rank's rows
of an FSDP2 shard. The names map one to one, a tied head is written once
from the embedding, and no projection is permuted: the rotary tables are
computed, and the rotate-half convention matches the checkpoint's.

## Construction

```python
from workloads.models.llama3.config import PRESETS
from workloads.models.llama3.implementations.mlops import Model

model = Model(PRESETS["8b"])
```

`workloads.models.catalog.build_on_meta("llama3-8b", "mlops", dtype="bfloat16")`
declares the same model on `meta`, and the quickstart runs it with
`python -m benchmarking.quickstart --model llama3-8b`. The
[model catalog](../../MODELS.md) covers dtypes, LoRA, inputs and the loss.
