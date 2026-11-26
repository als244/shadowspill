# Inkling-2-Mini

The `inkling2_mini` family trains the published language decoder on packed text. It predicts the next token over 200,058 classes. Text input skips the vision encoder and image-token injection. The separate speculative-decoding checkpoint is outside this workload.

```python
from shadowspill_client import LoRA, Model

model = Model.from_pretrained("thinkingmachines/Inkling-2-Mini", lora=LoRA(rank=32, alpha=32))
```

An absolute checkpoint path on the training server also works. The fine-tuning demo uses eight 8,192-token sequences per update and offers microbatch capacities of 8,192, 16,384 and 32,768 tokens. [Prepare the corpus and run the demo](../../../../../../docs/examples/client-training.md#fine-tune-inkling-2-mini-on-text)

## Shared layers and residual attention

The decoder makes 48 visits to 24 physical layers. It runs layers 0–5 once, layers 6–17 three times, and layers 18–23 once. Each later core pass restarts at the prefix state. Four learned gates per recurrent layer read messages from the preceding pass. Every visit uses the same physical layer weights and adapters and contributes to their gradients.

AttnRes mixes block histories independently at each token. The kernel reads the individual BF16 histories, normalizes and mixes them in FP32, and writes a BF16 result. Backward returns a gradient to each contributing history that requires one. Self-attention and the short convolutions use document-relative positions to respect packed sequence boundaries. [Decoder](implementations/mlops/model.py) · [Packed attention](implementations/mlops/attention.py)

## LoRA and the output head

| Rank-32 targets on the published checkpoint | Trainable tensors | Trainable parameters |
| --- | ---: | ---: |
| Default Q, K, V and output projections plus routed expert gate/up and down projections | 288 | 806,092,800 |
| Only the four attention projections | 192 | 13,369,344 |

Explicit `LoRA.targets` replace the defaults. The attention-only targets are `layers.*.attn.wq_du`, `layers.*.attn.wk_dv`, `layers.*.attn.wv_dv` and `layers.*.attn.wo_ud`. Include `lm_head` in the target list for head adapters. To train the full output matrix alongside the default adapters, use `trainable=["*.lora_*", "lm_head.weight"]`. [Default targets](lora.py)

The shared chunked head applies the logit softcap after the base and adapter projections and before cross entropy. It consumes vocabulary values one chunk of token rows at a time. A frozen head saves the hidden-state gradient when needed. A trainable head also accumulates its weight or adapter gradients. The base projection writes FP32 logits by default. Set `head_fp32_projection=False` under `config["architecture"]` to round the projection to the activation dtype before the FP32 softcap. Both prediction and training honor this setting.

## Checkpoint and expert precision

The loader checks the stored payloads and scales against `hf_quant_config.json` for both expert banks in each physical layer. Layers 0–19 use NVFP4 with one E4M3 scale per 16 columns and an FP32 outer factor per expert. Layers 20–23 use MXFP8. Its recipe declares 32 × 32 weight blocks, and the file stores a UE8M0 exponent per 32 columns of each serialized row. [Checkpoint mapping](checkpoint.py)

| Architecture settings for pretrained LoRA | Storage and arithmetic |
| --- | --- |
| Defaults, `expert_storage="auto"`, `expert_compute="auto"` | Keep the packed checkpoint. The RTX 5090 uses checkpoint activation quantization. The RTX 3090 decodes expert tiles for BF16 grouped GEMMs. |
| `expert_storage="packed", expert_compute="bf16"` | Keep the packed weights and decode on demand. Activations and cotangents use BF16. |
| `expert_storage="bf16"` | Decode once at initialization. The planner sees BF16 expert banks shared across visits. |
| `expert_storage="packed", expert_compute="checkpoint"` | Require the checkpoint recipe. An unsupported CUDA device reports an error. |

Pass overrides as `Model.from_pretrained(..., config={"architecture": {"expert_storage": "bf16"}})`. Full-parameter training defaults to dense floating-point expert banks. The planner accounts for the selected storage, optimizer state and task workspace.

In checkpoint mode, NVFP4 activations use a per-token outer scale and `fouroversix_mae` for each 16-value block. MXFP8 activations use groups of 32. The router weight multiplies the capped SwiGLU before quantization for the down projection. LoRA uses floating-point adapters. The frozen-base input gradient uses MXFP8 cotangents and a stochastically requantized transpose, adapting the published **Small** training recipe to Mini. RTX 5090 and RTX 3090 have end-to-end training coverage. SM90 has a checkpoint provider and awaits a physical H100 training run. [Published training recipe](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)

## Attention memory at long sequence lengths

The default partition follows decoder visits. The optional split places a boundary before the expert MLP at each of the six global-attention visits in the repeated core. Offer both as ordinary build cases so the planner can compare them under the same budget.

```python
from shadowspill_client import BuildCandidates, BuildCase, BuildOptions, Policy, Preparation

split = Policy("shadowspill/inkling2_mini_split_global_experts@1", {})
prepare = Preparation(candidates=BuildCandidates(cases=[
    BuildCase(32768, max_datums=1),
    BuildCase(32768, BuildOptions(partition=split), max_datums=1),
]))
```

- **Attention backend.** `attention_backend="auto"` selects tiled relative attention from 128 tokens at supported head widths. Its CUDA path streams score tiles. A positive `attention_kv_heads_per_chunk` also bounds the relative-bias profiles materialized together. The unit is K/V heads, and each head still covers its full sequence. Zero uses all heads.
- **Reference paths.** Set `attention_backend="eager"` for dense attention, or use a positive `attention_query_chunk_size` with the eager or automatic backend for the query-blocked reference. K/V-head chunking requires the fused operator and a zero query chunk size. [Attention settings](config.py)
- **Retention.** Pass `retention=["cost-input", "cost-attention"]` to `build_candidates` to compare the default policy with saving relative-attention outputs inside recomputed stages. The latter covers local and global self-attention. A custom policy can scope it to the global layers 5, 11, 17 and 23. [Client build options](../../../../../../docs/python/advanced/service-client.md) · [Retention rules](../../../../../../docs/architecture/graph-pair-construction.md)
