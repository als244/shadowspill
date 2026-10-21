# FSDP2 over the catalog models

Native PyTorch FSDP2 (`fully_shard`) over any catalog model, full or LoRA,
with the shared objective, geometry, precision contract and result row of
`baselines/common`. No ShadowSpill runtime is used: the trainer refuses to
finish if `shadowspill` was imported. Every case starts through the launcher
(`baselines.common.launch`), run from the repository root in the `baselines`
environment (`baselines/scripts/setup.sh`); the trainer runs there too, the
environment this engine names.

One case:

```bash
python -m baselines.common.launch --engine fsdp2 --gpus 1 --output <dir> \
  --model llama3-8b --tokens-per-step 65536 --tokens-per-microbatch 8192 \
  --recompute 1 --activation-offload none --offload-policy cpu
```

The tiny qualification matrix, every family but Qwen 3.5 MoE
(`../PRECISION.md` says why) against a reference copy compiled whole:

```bash
python -m baselines.sweep --engine fsdp2 \
  --config baselines/fsdp2/configs/qualification.json --output <dir> --gpus 1
```

A throughput matrix names only FSDP2's own knobs and runs inside an
experiment, which gives every engine the same model, geometry and
precision contract (`../experiments/README.md`):

```bash
python -m baselines.experiment \
  --spec baselines/experiments/full_rtx5090/llama3-8b_4K_16seqs/experiment.json --output <dir>
```

## What a case names

| Flags | Meaning |
|---|---|
| `--model`, `--implementation`, `--mode`, `--lora JSON`, `--model-config JSON` | The catalog model (`workloads/models/catalog.py`), its `pytorch` or `mlops` implementation, `full` or `lora` training with `configure_lora`'s options, and field overrides on the entry's configuration (`baselines.common.models`) |
| `--sequence-length`, `--tokens-per-step`, `--tokens-per-microbatch` | One update's geometry: `tokens_per_step = world_size * tokens_per_microbatch * accumulation`, in whole sequences; a geometry that does not divide is refused (`baselines.common.geometry`) |
| `--model-dtype`, `--master-dtype`, `--grad-dtype`, `--opt-state-dtype`, `--parameter-rounding`, `--opt-state-rounding`, `--fuse-narrow-gradient-accumulation` | The precision contract the quickstart names (`baselines.common.precision`); what FSDP2 cannot honor is refused, never substituted: see below |
| `--data DIR` | A corpus `baselines/scripts/prepare_corpus.py` wrote; each rank reads one contiguous window per step, disjoint across ranks and steps until the corpus wraps. Without it, synthetic tokens distinct per rank from `--seed` |
| `--checkpoint DIR` | A local Hugging Face checkpoint of the model, read through the family's `checkpoint.py` after the shards are materialized: every rank reads the whole tensors and keeps its rows, a checkpoint whose configuration is not the catalog entry's is refused, and keys the model does not take fail the run. Without it, the initialized weights |
| `--recompute F` | Fraction of whole transformer blocks checkpointed, spread evenly, from 0 to 1 |
| `--activation-offload {none,cpu,overlapped}` | `cpu` wraps every block in PyTorch's `offload_wrapper` (`save_on_cpu`, pinned, synchronous copies); `overlapped` is torchtune's stream-overlapped `OffloadActivations`, vendored as `offload_overlapped.py` with its license, entered at the first block and left after the last so the embedding and head stay on the device. Both compose with `--recompute`: a checkpointed block offloads its input, and recomputation stays on the device |
| `--offload-policy {none,cpu}`, `--[no-]pin-memory` | FSDP2's `offload_policy`: `cpu` is its `CPUOffloadPolicy`, whose one option is `pin_memory`; what it moves is in the table below |
| `--forward-prefetch N`, `--backward-prefetch auto\|N` | Explicit lookahead in blocks; `0` leaves native forward behavior, backward depth zero is not an off switch and is refused |
| `--gradient-sync {per-microbatch,last-microbatch}` | Reduce each microbatch into the owned shards, or keep full gradients and reduce on the last |
| `--[no-]reshard-after-forward`, `--[no-]reshard-after-backward` | Release each block's gathered weights after forward; reshard between accumulation microbatches (the final backward always reshards) |
| `--compile-blocks` | `torch.compile` on every block |
| `--allocator-cap-gib G` | A PyTorch caching-allocator limit; NCCL and other external allocations are outside it |
| `--float8` | torchao Float8 training, tensorwise scaling with the Float8 all-gather, on every block linear the kernels take; the head stays as it is. Full mode on bfloat16 weights |
| `--optimizer {adamw,adamw8bit,adamw4bit,cpu-offload}` | Fused torch AdamW, torchao's 8-bit or 4-bit AdamW (their state is in low-bit blocks, so `--opt-state-dtype none`), or torchao's `CPUOffloadOptimizer` stepping on the host with the shards on the device |
| `--[no-]offload-gradients` | With `--optimizer cpu-offload`, torchao's `offload_gradients`: each device gradient is freed once it is copied to the host; one microbatch per update only |
| `--warmup-steps`, `--steps` | Excluded warmups and measured complete optimizer updates |
| `--learning-rate`, `--weight-decay`, `--seed`, `--cpu-threads` | Optimizer settings, the data and initialization seed, intra-op threads per rank |
| `--verify-reference` | Tiny models: compare every update's loss, gradients, masters and both AdamW moments to an unsharded copy compiled whole; never a throughput result |
| `--nsys-capture` | Bracket the measured steps with the profiler; not a throughput result |

## Where each mechanism comes from

Everything the engine does to place weights, gradients, optimizer state and
activations is a library's own mechanism; the engine only chooses its
arguments. Most of it is FSDP2's and PyTorch's. The rest comes from two
other libraries, torchtune and torchao, and the sections below say which.
A combination a section marks as refused fails when the case is validated,
before anything launches.

### FSDP2

| Option | FSDP2's call | What the engine enforces |
|---|---|---|
| `--offload-policy cpu`, `--[no-]pin-memory` | `fully_shard(offload_policy=CPUOffloadPolicy(pin_memory))`. Parameters, gradients and optimizer state move together; FSDP2 has no switch for one of them alone | The optimizer steps on the host. Refused with torchao's `--optimizer cpu-offload` |
| `--forward-prefetch N`, `--backward-prefetch auto\|N` | `set_modules_to_forward_prefetch` and `set_modules_to_backward_prefetch` on the root and every block, naming the next N blocks; `0` and `auto` make no call, leaving FSDP2's own behavior | Backward depth zero is refused: it is not an off switch |
| `--[no-]reshard-after-forward` | `fully_shard(reshard_after_forward=)` on every block; the root keeps its weights gathered | |
| `--[no-]reshard-after-backward` | `set_reshard_after_backward` between accumulated microbatches | The last microbatch always reshards, so with one microbatch per update the switch has no effect |
| `--gradient-sync` | `set_requires_gradient_sync` per microbatch | The last microbatch always synchronizes, so with one microbatch per update the setting has no effect |
| The precision contract | `MixedPrecisionPolicy(param_dtype, reduce_dtype)` | See Precision below |

### PyTorch

| Option | PyTorch's call | What the engine enforces |
|---|---|---|
| `--recompute F` | `checkpoint_wrapper` on each recomputed block | |
| `--activation-offload cpu` | `offload_wrapper` around every block: `save_on_cpu` into pinned memory, the copies synchronous | |
| `--optimizer adamw` | `torch.optim.AdamW(fused=True)`, its fused implementation on the device, and on the host under `--offload-policy cpu` | |
| `--compile-blocks` | `torch.compile` on every block | |

### torchtune

| Option | torchtune's call | What the engine enforces |
|---|---|---|
| `--activation-offload overlapped` | `OffloadActivations`, vendored as `offload_overlapped.py` with its license: the saved values copied to pinned host memory on side streams, overlapping compute, entered at the first block and left after the last so the embedding and head stay on the device | Composes with `--recompute`: a recomputed block offloads its input |

### torchao

None of these is an offload FSDP2 makes, and one of them cannot run beside
FSDP2's: the CPU offload optimizer takes the weights on the device, which
`--offload-policy cpu` keeps on the host.

| Option | torchao's call | What the engine enforces |
|---|---|---|
| `--float8` | `convert_to_float8_training` with `Float8LinearConfig.from_recipe_name("tensorwise")` and `enable_fsdp_float8_all_gather=True` on every block linear whose dimensions are multiples of 16, and `precompute_float8_dynamic_scale_for_fsdp` after each update; the head and the embedding stay as they are | bfloat16 weights only; full training only, since LoRA adapts the linears Float8 converts; a device of compute capability 8.9 or above, checked before the model is built; refused with `--verify-reference` |
| `--parameter-rounding stochastic` with `--optimizer adamw` | `_AdamW(bf16_stochastic_round=True)` in place of PyTorch's AdamW | bfloat16 weights only |
| `--optimizer adamw8bit`, `adamw4bit` | `AdamW8bit`, `AdamW4bit`, their state in low-bit blocks; they step host tensors too, so they run beside `--offload-policy cpu` | `--opt-state-dtype none`; refused with `--verify-reference` |
| `--optimizer cpu-offload` | `CPUOffloadOptimizer(torch.optim.AdamW, fused=True)`. The weights and gradients stay on the device; the optimizer state and its step go to the host | The weights must be on the device: refused with `--offload-policy cpu`. It steps with PyTorch's AdamW, so it refuses `--parameter-rounding stochastic`; refused with `--verify-reference` |
| `--offload-gradients` | `CPUOffloadOptimizer(offload_gradients=True)`. Each device gradient is freed once it is copied to the host, which torchao does not support across accumulated microbatches | Only with `--optimizer cpu-offload` and one microbatch per update (`tokens_per_step` equal to the world size times `tokens_per_microbatch`) |

## What each offload option moves

Where each option puts the weights, gradients, optimizer state and activations,
and where the optimizer steps; an empty cell is the default's.

| Option | Weights the compute uses | Weights the optimizer steps | Gradients | Optimizer state and step | Activations |
|---|---|---|---|---|---|
| none (the default) | device | device: FSDP2's shards | device | device | device |
| `--offload-policy cpu` (FSDP2's `CPUOffloadPolicy`) | device: each block's gathered from the host before it runs, freed as resharding says | host: the shards, the float32 masters when the contract keeps them and the weights themselves otherwise | host: reduced into the shards | host; the step on the CPU | device |
| `--optimizer cpu-offload` (torchao's `CPUOffloadOptimizer`) | device | device, and a pinned host copy the optimizer steps and copies back; weights under 4096 elements stay on the device with a device optimizer | device, copied to the host after each backward | host; the step on the CPU | device |
| and `--offload-gradients` (torchao's `offload_gradients`, one microbatch per update) | device | as above | host: each device gradient freed once it is copied | host; the step on the CPU | device |
| `--activation-offload cpu` (PyTorch's `offload_wrapper`) or `overlapped` (torchtune's `OffloadActivations`, vendored) | | | | | host: each block's saved values, copied synchronously or on side streams; beside `--recompute`, a recomputed block's input |

## Precision

[`../PRECISION.md`](../PRECISION.md) sets this engine beside the others.
FSDP2 keeps a trained weight's shard at the dtype it was materialized at,
the master when one is kept, and sums gradients into that shard; AdamW keeps
its moments at it. So `--grad-dtype` and `--opt-state-dtype` must be the
stepped dtype (`float32` with masters, the model dtype without), or
`parameter`, or `none`. The default is bfloat16 compute with float32 masters,
gradients and moments: gradients default to float32 here, where the shared
default is the model dtype, since the shard they are summed into is the
float32 master. `--master-dtype none` with a bfloat16 model steps the
weights themselves; `--parameter-rounding stochastic` then uses torchao's
AdamW, which rounds the bfloat16 update stochastically. A float16 model would
need loss scaling the engine does not do, and `--fuse-narrow-gradient-accumulation`
is not available: both are refused. Each microbatch's gradients are
created at the compute dtype, since autograd gives a weight's gradient the
weight's own dtype; FSDP2 casts them to `--grad-dtype` for the
reduce-scatter (`MixedPrecisionPolicy`'s `reduce_dtype`) and sums them into
the shard.

## What a run writes

```text
output/
  case.json                 # the case the launcher validated and resolved
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
the result is the median over measured steps (`baselines.common.runner`).
Useful TFLOP/s per GPU uses the convention of
`baselines.common.models.useful_flops_per_token`, counted by the launcher
from the catalog model the case builds: recomputation, the optimizer and
communication are not credited. Peaks are PyTorch allocator measurements of the binding rank,
and the host peak is the sum of the ranks' high-water marks. The tiny
qualification and profiled runs are rows whose measurement is not a
throughput result. Every measured loss must be finite, and the first rank's
summary requires every rank and every measured step.

Parameters are created on meta, sharded before materialization, and
initialized in place on the device or the host as `--offload-policy` asks; compute
buffers stay on the device. By default gradients reduce-scatter after every
microbatch and accumulate in owned shards at the stepped dtype. The first
update checks the shards, their gradients and the moments against the
contract with plain AdamW.
