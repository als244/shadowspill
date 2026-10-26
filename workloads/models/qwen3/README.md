# Dense Qwen 3

GQA attention and a dense SwiGLU MLP in every decoder block, with RMSNorm and
RoPE, as in Llama 3. Each head's query and key are RMS-normalized over the head
width before the rotation, and the attention has a width of its own,
`n_heads * head_dim`, which need not be the model's. The embedding and the
output head share one weight in the smaller published sizes, as Qwen3-1.7B's
`tied_embeddings` does. The shapes follow the published configurations, such
as [Qwen3-8B's](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json).

## Shapes

[config.py](config.py) holds `Config` and `PRESETS`, the named shapes the
[catalog](../catalog/) serves; the
[catalog table](../../MODELS.md#preset-dimensions-and-parameter-counts) lists
their widths and parameter counts.

| Catalog name | Shape |
|---|---|
| `qwen3-tiny` | a two-layer smoke-test shape whose attention is twice its model width |
| `qwen3-1.7b` | Qwen3-1.7B, with tied embeddings |
| `qwen3-8b` | Qwen3-8B |
| `qwen3-32b` | Qwen3-32B, whose attention (8,192) is wider than its model (5,120) |

## Configuration

| Option | Meaning / default |
|---|---|
| `n_layers`, `d_model` | Number of decoder blocks and residual width |
| `n_heads`, `n_kv_heads` | Query heads and grouped key/value heads; query heads must divide into KV groups |
| `head_dim` | Width of every query, key and value head and of the query and key norms; even, since RoPE rotates channel pairs |
| `d_ff` | Intermediate width of each dense SwiGLU projection |
| `vocab_size` | Embedding/output-head row count; may include padded vocabulary rows |
| `rope_base` | RoPE base; 1,000,000 |
| `max_seq_len` | Maximum supported sequence length / rotary-table capacity; 40,960 |
| `tied_embeddings` | Share embedding and output-head weights; false |
| `norm_epsilon` | Stabilizer of every RMSNorm: the block, final, query and key norms; 1e-6 |

All shape fields above `rope_base` are required for a custom configuration.
`attention_width` (`n_heads * head_dim`) and `key_value_width`
(`n_kv_heads * head_dim`) are derived. With tied embeddings, freezing or
training either name affects the same Parameter.

## Implementations

[implementations/pytorch.py](implementations/pytorch.py) is the readable
numerical authority; [implementations/mlops.py](implementations/mlops.py) is
its state-dictionary-compatible twin on the operation library. Both expose
`Model`. The norms are Llama's: the normalized value is rounded to the input
dtype before the gain, where Qwen 3.5 keeps its weights as offsets from one.

## Checkpoint

`checkpoint.py` reads a local Hugging Face Qwen 3 checkpoint:
`Qwen3Checkpoint(directory)` turns its `config.json` into this family's
`Config` through `config.from_huggingface` (the rotary base from
`rope_theta` or `rope_parameters`; `max_seq_len` stays the preset's rotary
capacity unless overridden) and `load_into(model)` writes every parameter once
from the memory-mapped safetensors shards, through a plain tensor or this
rank's rows of an FSDP2 shard. The names map one to one, the query and key
norms included, and a tied head is written once from the embedding; a tied
checkpoint that also stores the head, as Qwen3-1.7B's does, is read only if
the stored head equals the embedding. No projection is permuted: the rotary
tables are computed, and the rotate-half convention matches the
checkpoint's. A checkpoint that enables what the
definition does not reproduce is refused: attention biases, sliding-window
attention, a rotary scaling such as YaRN, or an activation other than SiLU.

## Construction

```python
from workloads.models.qwen3.config import PRESETS
from workloads.models.qwen3.implementations.mlops import Model

model = Model(PRESETS["8b"])
```

`workloads.models.catalog.build_on_meta("qwen3-8b", "mlops", dtype="bfloat16")`
declares the same model on `meta`, and the quickstart runs it with
the [phase benchmark factory](../../recipes/text/benchmark/README.md)
with `model="qwen3-8b"`. The
[model catalog](../../MODELS.md) covers dtypes, LoRA, inputs and the loss.
