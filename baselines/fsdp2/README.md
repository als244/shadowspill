# FSDP2 over the catalog models

Native PyTorch FSDP2 (`fully_shard`) over any catalog model, full or LoRA,
with the shared objective, geometry, precision contract and result row of
`baselines/common`. No ShadowSpill runtime is used: the trainer refuses to
finish if `shadowspill` was imported. Run from the repository root in the
`baselines` environment (`baselines/scripts/setup.sh`).

One case:

```bash
torchrun --standalone --nproc-per-node=1 -m baselines.fsdp2.train \
  --output /mnt/storage/research/shadowspill/baselines/round1/fsdp2/llama3-8b-r100 \
  --model llama3-8b --tier rtx5090 --tokens-per-microbatch 8192 \
  --recompute 1 --activation-offload none --offload none
```

The tiny qualification matrix, every family against an eager reference:

```bash
python -m baselines.sweep --harness fsdp2 \
  --config baselines/fsdp2/configs/qualification.json \
  --output /mnt/storage/research/shadowspill/baselines/dev/fsdp2-qualification --gpus 1
```

A throughput matrix, resumable, with the same manifest:

```bash
python -m baselines.sweep --harness fsdp2 \
  --config baselines/fsdp2/configs/rtx5090_llama3-8b.json \
  --output /mnt/storage/research/shadowspill/baselines/round1/fsdp2/llama3-8b --gpus 1
```

## What a case names

| Flags | Meaning |
|---|---|
| `--model`, `--implementation`, `--mode`, `--lora JSON`, `--model-config JSON` | The catalog model (`workloads/models/catalog.py`), its `pytorch` or `mlops` implementation, `full` or `lora` training with `configure_lora`'s options, and field overrides on the entry's configuration (`baselines.common.models`) |
| `--sequence-length`, `--tokens-per-microbatch`, `--tokens-per-step` or `--tier` | One update's geometry: `tokens_per_step = world_size * tokens_per_microbatch * accumulation`, in whole sequences; a geometry that does not divide is refused (`baselines.common.geometry`) |
| `--model-dtype`, `--master-dtype`, `--grad-dtype`, `--opt-state-dtype`, `--parameter-rounding`, `--opt-state-rounding`, `--round-accumulation-once` | The precision contract the quickstart names (`baselines.common.precision`); what FSDP2 cannot honor is refused, never substituted: see below |
| `--data DIR` | A corpus `baselines/scripts/prepare_corpus.py` wrote; each rank reads one contiguous window per step, disjoint across ranks and steps until the corpus wraps. Without it, synthetic tokens distinct per rank from `--seed` |
| `--checkpoint DIR` | A local Hugging Face checkpoint of the model, read through the family's `checkpoint.py` after the shards are materialized: every rank reads the whole tensors and keeps its rows, a checkpoint whose configuration is not the catalog entry's is refused, and keys the model does not take fail the run. Without it, the initialized weights |
| `--recompute F` | Fraction of whole transformer blocks checkpointed, spread evenly, from 0 to 1 |
| `--activation-offload {none,cpu,overlapped}` | `cpu` wraps every block in PyTorch's `offload_wrapper` (`save_on_cpu`, pinned, synchronous copies); `overlapped` is torchtune's stream-overlapped `OffloadActivations`, vendored as `offload_overlapped.py` with its license, entered at the first block and left after the last so the embedding and head stay on the device. Both compose with `--recompute`: a checkpointed block offloads its input, and recomputation stays on the device |
| `--offload {none,cpu}`, `--[no-]pin-memory` | FSDP2's `CPUOffloadPolicy`: parameter shards, gradients and optimizer state on the host, the optimizer step on CPU |
| `--forward-prefetch N`, `--backward-prefetch auto\|N` | Explicit lookahead in blocks; `0` leaves native forward behavior, backward depth zero is not an off switch and is refused |
| `--gradient-sync {per-microbatch,last-microbatch}` | Reduce each microbatch into the owned shards, or keep full gradients and reduce on the last |
| `--[no-]reshard-after-forward`, `--[no-]reshard-after-backward` | Release each block's gathered weights after forward; reshard between accumulation microbatches (the final backward always reshards) |
| `--compile-blocks` | `torch.compile` on every block |
| `--allocator-cap-gib G` | A PyTorch caching-allocator limit; NCCL and other external allocations are outside it |
| `--float8` | torchao Float8 training, tensorwise scaling with the Float8 all-gather, on every block linear the kernels take; the head stays as it is. Full mode on bfloat16 weights |
| `--optimizer {adamw,adamw8bit,adamw4bit,cpu-offload}` | Fused torch AdamW, torchao's 8-bit or 4-bit AdamW (their state is in low-bit blocks, so `--opt-state-dtype none`), or torchao's `CPUOffloadOptimizer` stepping on the host with the shards on the device |
| `--qlora-nf4` | LoRA with the frozen weight of every adapted linear quantized to NF4 (torchao); the model is materialized on the host, quantized, then sharded |
| `--warmup-steps`, `--steps` | Excluded warmups and measured complete optimizer updates |
| `--learning-rate`, `--weight-decay`, `--seed`, `--cpu-threads` | Optimizer settings, the data and initialization seed, intra-op threads per rank |
| `--verify-reference` | Tiny models: compare every update's loss, gradients, masters and both AdamW moments to an unsharded eager copy; never a throughput result |
| `--nsys-capture` | Bracket the measured steps with the profiler; not a throughput result |

## Precision

FSDP2 keeps a trained weight's shard at the dtype it was materialized at,
the master when one is kept, and sums gradients into that shard; AdamW keeps
its moments at it. So `--grad-dtype` and `--opt-state-dtype` must be the
stepped dtype (`float32` with masters, the model dtype without), or
`parameter`, or `none`. The default is bfloat16 compute with float32 masters,
gradients and moments. `--master-dtype none` with a bfloat16 model steps the
weights themselves; `--parameter-rounding stochastic` then uses torchao's
AdamW, which rounds the bfloat16 update stochastically. A float16 model would
need loss scaling the harness does not do, and `--round-accumulation-once`
is not available: both are refused. `mlops.dispatch.set_weight_gradient_dtype`
makes the MLOps kernels produce weight gradients at the stepped dtype.

## What a run writes

```text
output/
  rank-NNNNN/request.json   # the case as described, the hardware, the resolved precision,
                            # checkpointed layers, the model summary, the corpus
  rank-NNNNN/result.json    # cycles, losses, peaks, evidence; or failure.json
  results.json              # every rank's measurements (first rank)
  row.json                  # the shared result row (first rank)
```

A sweep adds `cases/<case>/attempt-NNN/` around that, `manifest.json`,
`results.json` and a `README.md` index (`baselines/sweep.py`), and the report
reads the rows (`baselines/report.py`).

## Measurement

Every rank's cycle times are kept; a step's time is the slowest rank's and
the result is the median over measured steps. Useful TFLOP/s per GPU uses
the convention of `baselines.common.models.useful_flops_per_token`, counted
from the model as built: recomputation, the optimizer and communication are
not credited. Peaks are PyTorch allocator measurements of the binding rank,
and the host peak is the sum of the ranks' high-water marks. The tiny
qualification and profiled runs are rows whose measurement is not a
throughput result. Every measured loss must be finite, and the first rank's
summary requires every rank and every measured step.

Parameters are created on meta, sharded before materialization, and
initialized in place on the device or the host as `--offload` asks; compute
buffers stay on the device. By default gradients reduce-scatter after every
microbatch and accumulate in owned shards at the stepped dtype. The first
update checks the shards, their gradients and the moments against the
contract with plain AdamW.
