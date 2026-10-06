# Workloads

This directory contains model and data definitions used by benchmarking and
qualification. They are clients of the public ShadowSpill API and are not
installed as part of the `shadowspill` package.

Core planning, runtime, and lowering code must never import `workloads`.

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
