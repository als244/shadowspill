# DeepSpeed ZeRO over the catalog models

DeepSpeed (ZeRO stages 1 to 3 with its optimizer and parameter offload, its
activation checkpointing and CPU checkpointing) over any catalog model, full
or LoRA, with the shared objective, geometry, precision contract and result
row of `baselines/common`. No ShadowSpill runtime is used. Run from the
repository root in the `baselines` environment.

One case:

```bash
torchrun --standalone --nproc-per-node=1 -m baselines.deepspeed.train \
  --output /mnt/storage/research/shadowspill/baselines/round1/deepspeed/llama3-8b-z3 \
  --model llama3-8b --tier rtx5090 --tokens-per-microbatch 8192 \
  --zero-stage 3 --offload-optimizer cpu --recompute 1
```

The tiny qualification matrix and a throughput matrix run through the
shared sweep, `python -m baselines.sweep --harness deepspeed --config
baselines/deepspeed/configs/qualification.json --output <dir> --gpus 1`.

## What a case names

The common flags are the FSDP2 harness's (`../fsdp2/README.md`): the model,
the geometry, the precision contract, `--data`, `--checkpoint`,
`--recompute`, the allocator cap and the run. This harness adds:

| Flags | Meaning |
|---|---|
| `--zero-stage {1,2,3}` | ZeRO shards the optimizer state, then the gradients too, then the parameters too |
| `--offload-optimizer {none,cpu,nvme}` | Optimizer state and step on the host (DeepSpeed's CPU AdamW) or NVMe; stages 2 and 3 |
| `--offload-param {none,cpu,nvme}`, `--nvme-path DIR` | Parameters on the host or NVMe between uses; stage 3 |
| `--[no-]pin-memory` | Pin host-offloaded state |
| `--[no-]overlap-comm`, `--reduce-bucket-size N` | Gradient reduction overlapped with backward, in buckets of N elements |
| `--prefetch-bucket-size N`, `--max-live-parameters N`, `--param-persistence-threshold N` | Stage 3's parameter prefetch, the gathered parameters kept at once, and the size below which a parameter stays gathered |
| `--activation-offload {none,cpu}` | DeepSpeed's CPU checkpointing: the inputs of recomputed blocks on the host |
| `--contiguous-checkpointing` | One contiguous buffer for the checkpointed inputs |

Recomputation uses DeepSpeed's `checkpoint` on each selected block, and the
optimizer is always DeepSpeed's AdamW (fused on the device, its CPU
implementation when offloaded). DeepSpeed builds its extensions on first use
with the system's compiler toolkit; the trainer sets `DS_SKIP_CUDA_CHECK` so a
toolkit whose minor version differs from PyTorch's build is accepted. Each
microbatch's loss is handed to DeepSpeed as the microbatch's mean, since
DeepSpeed divides the gradients by the accumulation count itself; the
objective recorded is still the microbatch's share of the update.

## Precision

DeepSpeed's bf16 and fp16 modes cast every parameter to the model dtype,
the ones a model pins at float32 included (the row records which), and keep
float32 masters in the ZeRO optimizer with float32 AdamW moments; fp16 runs
under DeepSpeed's dynamic loss scaling. Gradients reduce and accumulate in
the model dtype in stages 2 and 3, so those stages take `--grad-dtype` equal
to the model dtype; stage 1's optimizer accumulates at float32 when
`--grad-dtype float32` asks for it. A float32 model steps its weights
themselves. Rounding choices, bfloat16 moments and masters other than
float32 are refused.

## What a run writes

As the FSDP2 harness: `rank-NNNNN/request.json` (with the DeepSpeed
configuration the case ran under), `result.json` or `failure.json`,
`results.json` and `row.json` from the first rank. The tiny qualification
compares every update's loss, gradients (read through DeepSpeed's accessors
before its step clears them), masters and moments to an unsharded eager
copy that casts the pinned parameters as DeepSpeed does.
