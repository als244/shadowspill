# Workloads

This directory contains model and data definitions used by benchmarking and
qualification. They are clients of the public ShadowSpill API and are not
installed as part of the `shadowspill` package.

Core planning, runtime, and lowering code must never import `workloads`.

## Qwen MoE text models

`workloads.mlops.Qwen30B` and `Qwen35B` implement the published
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
and [Qwen3.5-35B-A3B text decoder](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json).
Their quickstart names are `mlops_qwen30b` and `mlops_qwen35b`.
The existing `mlops_qwen35` still names the **dense** Qwen3.5 workload.

| Setting | Qwen30B | Qwen35B |
|---|---:|---:|
| Text parameters | 30,532,122,624 | 34,660,610,688 |
| Layers / hidden width | 48 / 2,048 | 40 / 2,048 |
| Mixer pattern | GQA in every layer | 3 Gated DeltaNet, then 1 gated GQA |
| GQA heads / KV heads / head width | 32 / 4 / 128 | 16 / 2 / 256 |
| Routed experts / top-k / intermediate width | 128 / 8 / 768 | 256 / 8 / 512 |
| Shared expert | none | width 512, learned sigmoid gate |
| Vocabulary | 151,936 | 248,320 |
| RoPE base / fraction | 1e6 / 1 | 1e7 / 0.25 |

Every layer has MoE after its mixer. Both routers normalize selected top-k
probabilities. Qwen3.5 uses zero-centered RMSNorm, per-head query/output gates,
and a depthwise causal convolution of width 4 in DeltaNet. Norm epsilon is
1e-6. Embeddings and the output head are untied. Initialization uses standard
deviation 0.02, with the published norm/decay initialization. The optional
router balancing loss has coefficient 0.001 and follows the upstream
across-layer convention (balanced top-8 routing yields a value near 8).

These are fresh-initialized **text-training architectures**, not pretrained
checkpoint loaders. Qwen3.5's vision encoder and auxiliary multi-token prediction
module are not included. Text positions give equal phases on all multimodal
RoPE axes, so ordinary partial RoPE applies here. The parameter counts above
match the corresponding Hugging Face causal language models exactly.

```python
from dataclasses import replace
from workloads.mlops import Qwen30B, Qwen30BConfig

config = replace(Qwen30BConfig(), max_seq_len=1024)
# Construct after entering the runtime and creating the NCCL EP group.
model = Qwen30B(
    config, ep_group=ep_group, device=device,
    parameter_device="cpu", token_capacity=32768,
)
```

`Qwen35B` / `Qwen35BConfig` have the same interface. Without `ep_group`, the
model uses local MLOps experts and supports CPU/meta construction. With EP, it
uses the installed `mlops.expert_parallel.QuackMoE`; optional GPU dependencies
load only during EP construction. One MoonEP buffer and two projection banks
are shared across all layers. Pass `buffer=...` instead of `token_capacity` to
borrow an existing buffer. Buffer capacity fixes the exact microbatch token
count. Call `model.close()` after its compiled callables are closed.

Identify rank-unique weights using
`Distributed(group, replica_overrides=[(model.expert_parameters(), None)],
groups={"ep": group})`. Attention, shared experts, routers, norms, embeddings,
and output heads are replicated; routed expert parameters are sharded. The
caller supplies global loss normalization and optimizer policy. The model
contains no ShadowSpill imports. The EP8 quickstart/training experiment, exact
upstream revisions, CPU reference tests, and GPU validation status are recorded
under `docs/internal/plans/qwen_moe_ep8_1007/`.

For a local, full-model quickstart (no EP), provide enough host spill space:

```bash
python -m benchmarking.quickstart mlops_qwen30b \
  --sequence-length 1024 --sequences-per-step 64 \
  --min-tokens-per-microbatch 8192 --max-tokens-per-microbatch 32768 \
  --search-budget-gib 40,60,70 --run-budget-gib 40,60,70 \
  --spill-gib 320 --plots --resolution-plans
```

## QuackMoE OLMoE

`workloads.quack.OLMoE` is a regular decoder model: it has the same configuration,
forward, packed-input and loss interfaces as `workloads.mlops.OLMoE`, with
QuackMoE routed experts in each block. It uses MLOps attention and head/loss
operations. Expert computation defaults to BF16; `compute_precision="fp8_current"`
selects FP8. The router defaults to FP32;
`router_dtype=torch.bfloat16` selects BF16 router weights, computation and
gradients. The layer is packaged
as `mlops.expert_parallel.QuackMoE`. MoonEP and
their GPU dependencies are optional; importing the other workloads does not
import them. The current QuackMoE kernels require H100/SM90.

Install the optional backend from the MLOps checkout with
`./scripts/setup_expert_parallel.sh --backend quack --python /path/to/python`.
The workload imports the installed MLOps package and its external dependencies.

The caller selects the compute device, initialized NCCL EP group and local
microbatch capacity. The workload creates one MoonEP token buffer and one set of
expert communication banks, shared across all blocks. Parameters remain distinct.

```python
from workloads.quack import OLMoE, OLMoEConfig

config = OLMoEConfig(
    n_layers=2, d_model=512, n_heads=4, n_kv_heads=4, head_dim=128,
    n_experts=8, top_k=2, d_ff_expert=1024, vocab_size=1024, max_seq_len=256,
)
model = OLMoE(config, ep_group=ep_group, token_capacity=1024, device=device)
try:
    loss = model.loss(tokens, targets, reduction="sum", aux_coef=0.01)
    loss.backward()
finally:
    model.close()
```

Here `tokens` and `targets` contain 1,024 tokens on this rank's selected compute
device, in sequences of at most 256 tokens. Resource capacity fixes the local
microbatch size. Construct with the intended compute device; QuackMoE's fixed
communication mappings cannot be moved with `model.to()` afterward.
`parameter_device="cpu"` optionally initializes compact host model state for a
caller that supplies compute values from host memory during execution.

For FP8 training, construct with `compute_precision="fp8_current"` and
`weight_grad_dtype=torch.float32`. `activation_transport="fp8"` also sends
quantized activations through MoonEP; its default is `"bf16"`. The workload
constructs a compatible buffer. Pass `master_dtype=torch.float32` and
`grad_dtype=torch.float32` to the ordinary Trainer for dense optimizer masters
and gradients. ShadowSpill manages the weight payloads and scales through the
generic [tensor representation contract](../docs/architecture/state-import.md#tensor-representations).

The model does not own a trainer, planner, optimizer or data source. Its
`expert_parameters()` identifies home expert shards; all other parameters are
replicated within the EP group. A training caller sums replicated gradients
once and supplies the global loss normalization. QuackMoE already assembles the
home expert gradients. `model.close()` releases model-owned communication resources. Advanced callers
can instead pass `buffers=[buffer] * config.n_layers`; those buffers remain
caller-owned. Do not pass `token_capacity` along with `buffers`. When searching
microbatch capacities, construct the candidate model with its corresponding
capacity before capture/profiling and close it before constructing another.
The resource footprint then follows the candidate, rather than the largest case.

When using ShadowSpill, enter its runtime before importing this workload or
creating device communication resources: Quack's dependencies can inspect the
device during import. Pass the model into the ordinary `Trainer`. Declare unique expert
parameters with `Distributed(group, replica_overrides=[(model.expert_parameters(),
None)], groups={"ep": ep_group})`. The model itself contains no ShadowSpill code.
The Della integration probe and its validation evidence are maintained under
`docs/internal/plans/generic_training_0930/della_1001/`. Two-rank save/recompute
passed the independent full-model reference. The 50-step real-data EP1/EP2
comparison had a maximum objective difference of 0.000459.
