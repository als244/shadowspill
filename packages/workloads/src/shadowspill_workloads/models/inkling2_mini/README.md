# Inkling 2 Mini

The `inkling2_mini` family accepts packed text tokens. Its ordinary causal language head predicts the next token over the 200,058 unpadded classes. Text prompts run the full language decoder. Multimodal prompts would also require the image encoder and image-token injection. The separate MTP checkpoint serves speculative decoding.

Training uses the MLOps chunked head objective. It applies the checkpoint's logit softcap before cross entropy and consumes each token chunk's vocabulary values and gradients together. With a frozen head, backward saves the hidden-state gradient. The default base projection writes FP32 logits from the model's activation-precision operands. Set `config={"architecture": {"head_fp32_projection": False}}` to round the projection output to the activation dtype before applying the same FP32 softcap. The option also applies to ordinary logits returned by `forward`.

Include `"lm_head"` in the explicit `LoRA.targets` list to train output-head adapters. Explicit targets replace the family defaults. The head adds the scaled adapter projection before the softcap and accumulates gradients for its two factors across chunks. To train the full output matrix alongside the family's default adapters, pass `lora=LoRA(32, alpha=32), trainable=["*.lora_*", "lm_head.weight"]` to `Model.from_pretrained`. In that mode the objective accumulates one vocabulary-by-hidden weight gradient. The hidden-state gradient is saved when earlier layers need it.

## Shared layers and residual attention

The published shape has 24 physical layers and 48 visits. Layers 0–5 form the prefix, layers 6–17 run three times, and layers 18–23 form the suffix. A later pass restarts from the prefix state and retrieves four learned messages from the preceding pass. Attention residuals mix normalized block states at each token. Full-attention layers have independent queries for their Q, K and V inputs. The model declares one set of parameters and adapters per physical layer. ShadowSpill captures the separate visits and plans their activations, shared state and optimizer work. The `tiny` preset exercises the same three-pass structure. [Configuration](config.py) · [Decoder](implementations/mlops/model.py)

On recurrent visits, the Q, K, V and MLP cross-pass gates retain their narrow projected inputs for backward. Each gate rebuilds its FP32 forget and gain projections and the mixed state when the normalized gate receives its gradient. At 32,768 tokens the four rank-64 bottlenecks occupy 32 MiB combined. CUDA computes the RMS and tanh derivatives together per row in FP32. The portable implementation uses PyTorch. The learned gate weights and per-layer RMS gains support full-parameter gradients. [Decoder gates](implementations/mlops/model.py)

Inkling stores the running residual, completed block histories, cross-pass messages and the final mixture in BF16. The fused AttnRes kernel reads each existing history tensor directly. It computes RMS normalization, query scores, softmax and mixing in FP32, then writes its BF16 result. Backward reads the same histories and writes one BF16 gradient for each stream that needs it. Residual additions also accumulate in FP32 before storing the next state. Learned residual queries and their gradients use FP32. At 32,768 tokens one residual is 144 MiB. Reading five histories directly avoids a 720 MiB stacked input.

The default `attention_backend="auto"` selects the MLOps fused relative attention operator for at least 128 tokens and supported head widths. Its Triton kernels apply the learned per-query distance bias, sink and causal or sliding-window mask, including position resets at packed-document boundaries. CPU and unsupported device or dtype combinations use the bounded PyTorch implementation of that operator. Short sequences use the dense reference.

Set `config={"architecture": {"attention_backend": "eager"}}` for the dense reference, or set `attention_query_chunk_size` to a positive value such as 256 to use the query-blocked reference. Each query sees all the allowed keys in its document. These settings are independent of token capacity and the projection and MoE chunk settings. [Attention implementation](implementations/mlops/attention.py)

Use `retention=["cost-input", "cost-attention"]` in `build_candidates` to let the planner compare the default with retaining self-attention outputs inside recomputed stages. `cost-attention` retains the pre-gate output of each global or local MLOps relative-attention call, including the query-blocked and K/V-head-chunked forms. At 32,768 tokens that BF16 tensor is 192 MiB per visit. Backward uses it for the gate and projection and still computes the attention gradients. The dense eager reference has no operator-level retention rule. [Client build options](../../../../../../docs/python/advanced/service-client.md)

A custom policy can retain only global attention by scoping its `RetentionRule` entries to the attention modules of physical layers 5, 11, 17 and 23. Each rule covers every visit to its module. [Graph-pair retention](../../../../../../docs/architecture/graph-pair-construction.md)

## Checkpoint and expert precision

Residual attention, FP32 RMS normalization, L2 clipping scales and normalized router weights
use the same arithmetic in eager, saved and recomputed stages. A change at
a BF16 rounding boundary can change the block scale chosen by the next
expert quantizer.

On Chicago's RTX 5090, each recurrent message gate writes its two independently
normalized inputs directly into the BF16 controller projection input. Query and
key clipping also runs in one kernel and exposes one FP32 length per head and
token for backward. The relative projection keeps its 16-wide clipping path.

Use the regular client API on a server with the checkpoint installed.

```python
from shadowspill_client import LoRA, Model

model = Model.from_pretrained("/models/Inkling-2-Mini", lora=LoRA(2, alpha=2))
```

By default, LoRA keeps the released expert weights and scales in their original dtypes. The loader pins the checkpoint's `hf_quant_config.json` along with the weights. It checks the text tensor assignments, activation recipes, block shapes and physical scale tensors. Both the interleaved gate/up bank and the down bank use NVFP4 at physical layers 0–19 and MXFP8 at layers 20–23. NVFP4 weights have one E4M3 scale per 16 columns and an FP32 outer factor per expert. MXFP8 weights store E8M0 exponents for groups of 32 columns at each serialized row. [Checkpoint reader](checkpoint.py)

With packed storage, `expert_compute="auto"` uses the checkpoint arithmetic where MLOps supports it and BF16 grouped GEMMs elsewhere. Both released formats use the checkpoint path on SM90 and SM120 and BF16 on SM86. On SM89 only MXFP8 has checkpoint support. An explicit `"checkpoint"` requests the checkpoint recipe for each packed bank and reports unsupported devices. Its NVFP4 activations use a per-token outer scale and choose an E4M3 scale for each 16-value block by the four-over-six error rule. MXFP8 activations use 32-column groups. The router coefficient multiplies the capped SwiGLU before rounding and quantizing the input to the down projection. The frozen-base input gradient uses MXFP8 cotangents and stochastically requantizes the transposed decoded weights to MXFP8. This backward convention adapts the published Small training method to Mini LoRA. The low-rank branches use floating-point projections. [Thinking Machines training recipe](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/) · [vLLM Inkling experts](https://docs.vllm.ai/en/latest/api/vllm/models/inkling2/nvidia/moe/)

Two configurations use BF16 expert weights, activations and cotangents during the matrix products. Both call the ordinary MLOps grouped GEMM and accumulate each dot product in FP32.

| Architecture settings | Weight storage available to the planner | When conversion occurs |
| --- | --- | --- |
| `expert_storage="packed", expert_compute="bf16"` | Original checkpoint payloads and scales | Each forward or input-gradient projection decodes active experts in output-column tiles. The same scratch buffer serves every tile. |
| `expert_storage="bf16"` | BF16 expert banks shared across loop visits | The training initializer decodes the checkpoint directly into the requested parameter ranges. An already-BF16 checkpoint copies directly. Training reads the stored BF16 values. |

For example, this configuration converts the published checkpoint at load time. Both modes keep the base experts frozen under LoRA. ShadowSpill plans the weight objects, transfers, retention and task workspace using the selected representation.

```python
model = Model.from_pretrained(
    "/models/Inkling-2-Mini",
    lora=LoRA(2, alpha=2),
    config={"architecture": {"expert_storage": "bf16"}},
)
```

Full-parameter floating-point training uses dense expert banks. The bounded loader decodes directly into the requested destination ranges. On either path the ordinary public memory budgets and plan admission govern feasibility. The default LoRA targets train 50,380,800 parameters at rank two and 806,092,800 at rank 32, in 288 adapter tensors. They cover Q, K, V and output projections plus the routed expert gate/up and down projections of each of the 24 physical layers. Repeated visits accumulate into those same adapters. The router, language head, embeddings, AttnRes queries and other base parameters stay frozen under this target selection. [Adapter targets](lora.py) · [Model catalog](../../../../../../docs/python/catalog.md)

To train just the four attention projections, pass these targets to the same client API.

```python
lora = LoRA(2, alpha=2, targets=[
    "layers.*.attn.wq_du", "layers.*.attn.wk_dv",
    "layers.*.attn.wv_dv", "layers.*.attn.wo_ud",
])
```

This trains 835,584 elements in 192 adapter tensors. Frozen experts still propagate input gradients. With checkpoint computation on SM120 or SM90, their gate/up projection can gather and quantize in one kernel. An enabled expert adapter reads the original BF16 token rows by routing index for both its rank-wide projection and its weight gradient. With either target selection, the gate/up operation avoids the full-width BF16 assignment buffer.

## Attention memory at long sequence lengths

In global attention each K and V convolution consumes the packed positions, its static taps and a compact per-token tap projection. MLOps forms the four neighboring values and their weights inside the CUDA kernel, including during backward.

The default partition follows the decoder layers. For long documents the
workload also provides `shadowspill/inkling2_mini_split_global_experts@1`.
It adds a boundary before the routed MLP at each global-attention visit in
the repeated core. The published model has six such visits. Saved inputs,
recomputation and placement are chosen by ShadowSpill for each partition.
Both use whole AttnRes histories and the same model weights.

To offer both partitions at 32,768 tokens per microbatch, use the existing
client build cases in the preparation configuration.

```python
from shadowspill_client import BuildCandidates, BuildCase, BuildOptions, Policy, Preparation

split = Policy("shadowspill/inkling2_mini_split_global_experts@1", {})
prepare = Preparation(candidates=BuildCandidates(cases=[
    BuildCase(32768, max_datums=1),
    BuildCase(32768, BuildOptions(partition=split), max_datums=1),
]))
```

The planner can select either admitted candidate for the submitted batch.
The [fine-tuning example](../../../../../../examples/demos/inkling2_mini_lora.py)
uses the default partition and compares microbatch sizes.

Set `attention_kv_heads_per_chunk` in the architecture configuration to a positive number up to `n_kv_heads` to limit the number of relative-bias profiles materialized together. Each chunk contains complete grouped-query heads and spans every sequence position. Packed document boundaries, the global receptive field and the local window are preserved. Zero selects ordinary attention. This option combines the relative projection and the fused attention and can run alongside the independently chunked language head.
