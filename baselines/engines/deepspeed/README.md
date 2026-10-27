# DeepSpeed ZeRO over the catalog models

DeepSpeed (ZeRO stages 1 to 3 with its optimizer and parameter offload, its
activation checkpointing and CPU checkpointing, and its DeepCompile backend
with its offloads) over any catalog model, full or LoRA, with the shared
objective, geometry, precision contract and result row of
`baselines/common`. No ShadowSpill runtime is used. Every case starts
through the launcher (`baselines.common.launch`), run from the repository
root in the `baselines` environment, the environment the trainer runs in.

One case:

```bash
python -m baselines.common.launch --engine deepspeed --gpus 1 \
  --output <dir> \
  --model llama3-8b --tokens-per-step 65536 --tokens-per-microbatch 8192 \
  --zero-stage 3 --offload-optimizer cpu --recompute 1
```

The tiny qualification matrix runs through the shared sweep, `python -m
baselines.sweep --engine deepspeed --config
baselines/engines/deepspeed/configs/qualification.json --output <dir> --gpus 1`; a
throughput matrix names only DeepSpeed's own knobs and runs inside an
experiment, which gives every engine the same model, geometry and
precision contract (`../../README.md`).

## What a case names

The common flags are the FSDP2 engine's (`../fsdp2/README.md`): the model,
the geometry, the precision contract, `--data`, `--checkpoint`,
`--recompute`, the allocator cap and the run. This engine adds:

| Flags | Meaning |
|---|---|
| `--zero-stage {1,2,3}` | ZeRO shards the optimizer state, then the gradients too, then the parameters too |
| `--offload-optimizer {none,cpu,nvme}` | Optimizer state and step on the host (DeepSpeed's CPU AdamW) or NVMe; stages 2 and 3 |
| `--offload-param {none,cpu,nvme}`, `--nvme-path DIR` | Parameters on the host or NVMe between uses; stage 3 |
| `--[no-]pin-memory` | Pin host-offloaded state |
| `--[no-]overlap-comm`, `--reduce-bucket-size N` | Gradient reduction overlapped with backward, in buckets of N elements |
| `--prefetch-bucket-size N`, `--max-live-parameters N`, `--param-persistence-threshold N` | Stage 3's parameter prefetch, the gathered parameters kept at once, and the size below which a parameter stays gathered |
| `--activation-offload {none,cpu}` | DeepSpeed's CPU checkpointing: the inputs of recomputed blocks on the host, moved by its asynchronous offload engine (pinned, pooled host buffers on a side stream; the newest input stays on the device) |
| `--contiguous-checkpointing` | One contiguous buffer for the checkpointed inputs |
| `--deepcompile`, `--deepcompile-backend {inductor,eager}` | DeepCompile: DeepSpeed's `torch.compile` backend, whose graph passes take over ZeRO's hooks (the gradient reduction in stages 1 and 2; gather, release, prefetch and selective gather in stage 3), handing the graphs to the named compiler |
| `--deepcompile-offload {none,activations,optimizer-states,parameters}` | Under DeepCompile and ZeRO-3: the saved activations, the optimizer state or the parameters moved to pinned host memory as the memory budget requires, one of them per run |
| `--deepcompile-free-activation` | Free each activation in DeepCompile's backward as soon as its last use passes (eager backend) |

Recomputation uses DeepSpeed's `checkpoint` on each selected block, and the
optimizer is always DeepSpeed's AdamW (fused on the device, its CPU
implementation when offloaded). DeepSpeed builds its extensions on first use
with the system's compiler toolkit; the trainer sets `DS_SKIP_CUDA_CHECK` so a
toolkit whose minor version differs from PyTorch's build is accepted. Each
microbatch's loss is handed to DeepSpeed as the microbatch's mean, since
DeepSpeed divides the gradients by the accumulation count itself; the
objective recorded is still the microbatch's share of the update.

DeepCompile compiles the whole model and applies its planning passes at
its sixth update, compiling the graphs again, so a DeepCompile case warms
up eight steps or more. Recomputation is refused beside it: DeepSpeed's
checkpointing inside the compiled graphs loses the recomputed blocks'
gradients, and DeepCompile offloads or frees activations instead. Its CPU
checkpointing is refused too (DeepCompile's activation offload takes its
place), as is its optimizer-state offload beside ZeRO's optimizer offload;
with parameters and optimizer state both offloaded to the host,
DeepCompile offloads parameters itself and the case says so. The parameter
offload takes the `pytorch` implementation: DeepSpeed's pass hands a
kernel's tuple-valued last use to its offload operator. The eager backend
needs scipy (DeepSpeed's communication profiler), which the environment
installs. ZeRO-1 under DeepCompile takes one microbatch per update: with
gradients accumulated across microbatches its reduce aborts at the second
update (DeepSpeed releases its reduce buckets after an update and ZeRO-1's
first non-boundary backward flushes them before it makes them again;
ZeRO-2 and 3 make them at every backward). The engine falls back to plain
`torch.compile` when it declines a configuration; the trainer refuses to
measure that. Under DeepCompile the tiny qualification judges the moments
and masters: DeepSpeed's gradient accessors refuse after a backward its
graphs ran.

## What each offload option moves

Where each option puts the weights, gradients, optimizer state and activations,
and where the optimizer steps; an empty cell is the default's.

| Option | Weights the compute uses | Weights the optimizer steps | Gradients | Optimizer state and step | Activations |
|---|---|---|---|---|---|
| none (the default) | device | device | device | device | device |
| `--offload-param cpu\|nvme` (ZeRO-3) | host or NVMe between uses, fetched to the device as each module runs | as `--offload-optimizer` says | as `--offload-optimizer` says | as `--offload-optimizer` says | device |
| `--offload-optimizer cpu\|nvme` (ZeRO-2 and 3) | device, unless the parameters are offloaded | host or NVMe: DeepSpeed's master copy | host: the reduced partitions | host or NVMe; the step on the CPU, DeepSpeed's CPU AdamW | device |
| `--activation-offload cpu` (DeepSpeed's CPU checkpointing) | | | | | host: the inputs of recomputed blocks |
| `--deepcompile-offload parameters\|optimizer-states\|activations` (DeepCompile, ZeRO-3) | as DeepCompile's passes place them, within its memory budget (above) | | | | |

## Precision

[`../../PRECISION.md`](../../PRECISION.md) sets this engine beside the others.
DeepSpeed's bf16 and fp16 modes cast every parameter to the model dtype,
the ones a model pins at float32 included (the row records which), and keep
float32 masters in the ZeRO optimizer with float32 AdamW moments; fp16 runs
under DeepSpeed's dynamic loss scaling. A bfloat16 model with
`--master-dtype none` steps bfloat16 values with bfloat16 moments and
gradients instead, through DeepSpeed's `bf16_master_weights_and_grads` and
`bf16_optimizer_states`: the ZeRO optimizer's partition of what it steps is
bfloat16, and its CPU AdamW stores bfloat16 moments when the optimizer is
offloaded. `--grad-dtype float32` selects FP32 gradient accumulation and
communication in ZeRO stages 1, 2 and 3 through `data_types.grad_accum_dtype`
and `communication_data_type`; without it, the model dtype is used. In
ZeRO-3, precision evidence checks the actual gradient partition buffer. A float32 model steps its weights themselves. Rounding choices, float32
moments beside bfloat16 values, and a float16 model without float32
masters are refused.

Under DeepCompile the combinations qualified are a bfloat16 model with
float32 masters and bfloat16 gradients, and a float32 model stepping its
weights, in ZeRO-2 and 3 (ZeRO-1 with one microbatch per update); the
float32-accumulating optimizer of a bfloat16 model is DeepSpeed's BF16
optimizer, which DeepCompile does not take, a float16 model under its loss
scaling is not qualified, and LoRA runs under ZeRO-3 only (the ZeRO-1 and
2 passes register the optimizer's parameters alone, so a frozen base
parameter has no identity in the graphs). Each of these is refused before
launch.

## What a run writes

As the FSDP2 engine: `case.json`, `rank-NNNNN/request.json` (with the
DeepSpeed configuration the case ran under), `result.json` or `failure.json`,
`results.json` and `row.json` from the first rank. The tiny qualification
compares every update's loss, gradients (read through DeepSpeed's accessors
before its step clears them), masters and moments to an unsharded copy,
compiled whole, that casts the pinned parameters as DeepSpeed does.
