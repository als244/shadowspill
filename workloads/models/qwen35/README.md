# Dense Qwen 3.5

A repeating hybrid of Gated DeltaNet and gated GQA, followed by a dense SwiGLU
MLP in every block. `full_attention_interval=4` means three DeltaNet blocks,
then one full-attention block. This family has no routed experts.

## Shapes

[config.py](config.py) holds `Config` and `PRESETS`, the named shapes the
[catalog](../catalog.py) serves; the
[catalog table](../../MODELS.md#preset-dimensions-and-parameter-counts) lists
their widths and parameter counts. Each published size is the text decoder
alone: the checkpoints' vision encoder and multi-token prediction layer are
outside this model.

| Catalog name | Shape |
|---|---|
| `qwen35-numerical` | the exact-scale shape the numerical gate runs |
| `qwen35-tiny` | a two-layer smoke-test shape, one DeltaNet block then one full-attention block |
| `qwen35-2b` | Qwen3.5-2B's text decoder, with tied embeddings and 262,144 positions |
| `qwen35-9b` | Qwen3.5-9B's text decoder |
| `qwen35-27b` | Qwen3.5-27B's text decoder, 262,144 positions |

## Configuration

| Options | Meaning / defaults |
|---|---|
| `n_layers`, `d_model`, `d_ff`, `vocab_size` | Decoder depth, residual width, dense MLP width, vocabulary rows |
| `full_attention_interval` | Full attention every Nth block, counting from 1 |
| `n_heads`, `n_kv_heads`, `head_dim` | Full-attention query/KV heads and per-head width |
| `partial_rotary_factor` | Fraction of each attention head using RoPE; 0.25 in every preset |
| `lin_k_heads`, `lin_v_heads` | DeltaNet key/value head counts; value heads must divide into key-head groups |
| `lin_k_head_dim`, `lin_v_head_dim` | DeltaNet key/value head widths |
| `lin_conv_kernel` | Depthwise causal-convolution width |
| `rope_base` | 10,000,000 |
| `tied_embeddings` | Share embedding and output-head weights; false |
| `max_seq_len` | Maximum sequence length; 131,072, or 262,144 in the `2b` and `27b` presets |

Full-attention projection width is `n_heads * head_dim` and may differ from
`d_model`. The rotary width must be positive and even. With tied embeddings,
freezing or training either name affects the same Parameter.

## Implementations

[implementations/pytorch.py](implementations/pytorch.py) is the readable
numerical authority; [implementations/mlops.py](implementations/mlops.py) is
its state-dictionary-compatible twin on the operation library, whose DeltaNet
mixer lives in [shared/delta_net.py](../shared/delta_net.py) because the Qwen
mixture families use it too. Both expose `Model`.

## Checkpoint

`checkpoint.py` reads a local Hugging Face Qwen 3.5 checkpoint, the
vision-language model whose text decoder's keys begin
`model.language_model.` (the vision tower is ignored):
`Qwen35Checkpoint(directory)` turns the `text_config` into this family's
`Config` through `config.from_huggingface` and `load_into(model)` writes
every parameter once, through a plain tensor or this rank's rows of an FSDP2
shard. A full-attention block's query projection carries the attention gate,
interleaved per head in the checkpoint and split into all queries then all
gates here, so its rows are reordered on load; a linear-attention block's
`w_qkvz` is `in_proj_qkv` followed by `in_proj_z` and its `w_ba` is
`in_proj_b` followed by `in_proj_a`.

## Construction

```python
from workloads.models.qwen35.config import PRESETS
from workloads.models.qwen35.implementations.mlops import Model

model = Model(PRESETS["9b"])
```

`workloads.models.catalog.build_on_meta("qwen35-9b", "mlops", dtype="bfloat16")`
declares the same model on `meta`, and the quickstart runs it with
`python -m benchmarking.quickstart --model qwen35-9b`. The
[model catalog](../../MODELS.md) covers dtypes, LoRA, inputs and the loss.
