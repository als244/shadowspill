# Qwen 3.5 MoE

Three Gated DeltaNet blocks followed by gated GQA, repeating through the
network, with MoE after every mixer. The default shape follows the text decoder
of [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json).
It has a shared SwiGLU expert with a learned sigmoid gate, zero-centered RMSNorm,
and normalized top-k routed probabilities. The checkpoint's vision encoder and
multi-token prediction module are outside this text model.

## Shapes

[config.py](config.py) holds `Config` and `PRESETS`, the named shapes the
[catalog](../catalog.py) serves; the
[catalog table](../../MODELS.md#preset-dimensions-and-parameter-counts) lists
their widths and parameter counts.

| Catalog name | Shape |
|---|---|
| `qwen35moe-tiny` | a two-layer smoke-test shape, one DeltaNet block then one full-attention block, eight routed experts and a shared one |
| `qwen35moe-35b-a3b` | Qwen3.5-35B-A3B's text decoder |

## Configuration

`Config` shares its fields with Qwen 3 MoE through
[shared/qwen_moe/config.py](../shared/qwen_moe/config.py); the
[Qwen 3 MoE README](../qwen3moe/README.md#configuration) tabulates every
field with both families' defaults. This family's defaults are 40 layers of
width 2,048, 16/2 heads of width 256, 256 experts of width 512 and top-8, a
512-wide shared expert, a 248,320-row vocabulary, 262,144 positions, RoPE base
1e7 on a quarter of each head, zero-centered norms, gated attention and a
full-attention block every fourth layer.

## Implementations

[implementations/mlops.py](implementations/mlops.py) exposes `Model`, built
from the decoder, mixers and experts in
[shared/qwen_moe/](../shared/qwen_moe/) that both Qwen mixture families use,
with local experts by default or experts sharded over an expert-parallel group
through the constructor arguments the
[model catalog](../../MODELS.md#expert-parallelism) describes. There is no
pure-PyTorch implementation.

## Construction

```python
from workloads.models.qwen35moe.config import PRESETS
from workloads.models.qwen35moe.implementations.mlops import Model

model = Model(PRESETS["35b-a3b"])
```

`workloads.models.catalog.build_on_meta("qwen35moe-35b-a3b", dtype="bfloat16")`
declares the same model on `meta`, and the quickstart runs it with
`python -m benchmarking.quickstart --model qwen35moe-35b-a3b`. The
[model catalog](../../MODELS.md) covers dtypes, LoRA, inputs and the loss.
