# FSDP2 with the existing MLOps Qwen model

Run from the repository root using an environment with the same PyTorch/MLOps
versions as the comparison. No ShadowSpill runtime is used. The model, actual
top-k routing, auxiliary loss, and chunked language-model head come from
`workloads/mlops/qwen3_moe.py` and its existing implementation.

The trainer and sweep driver have passed a single-GPU numerical matrix covering
no/partial/full recomputation, CPU offload with/without pinned memory, prefetch,
both synchronization modes, and resharding controls. Multi-GPU qualification is
still required before accepting full-model baseline measurements.

Run the small numerical check first:

```bash
torchrun --standalone --nproc-per-node=8 -m baselines.fsdp2.train \
  --tiny --verify-reference --sequence-length 64 --global-tokens 2048 \
  --microbatch-tokens 64 --recompute 1 --warmup-steps 1 --steps 2 \
  --output runs/baselines/fsdp2/qualification-recompute
```

This compares loss, FP32 gradients, masters, and both AdamW moments against an
unsharded eager reference across multiple accumulation microbatches and updates.
The reference shares MLOps kernels; it checks FSDP reduction/accumulation and
optimizer ownership, not independent kernel mathematics. Each update starts
from identical master values and rotary buffers; independent optimizer moments
carry across updates. Re-aligning the already-checked master values prevents
tiny CPU/fused-Adam rounding differences from crossing BF16 rounding boundaries
and changing routing in subsequent steps. Qualification runs are marked as
unsuitable for throughput reporting.

After qualification, one full-model case is:

```bash
torchrun --standalone --nproc-per-node=8 -m baselines.fsdp2.train \
  --sequence-length 4096 --global-tokens 2097152 \
  --microbatch-tokens 16384 --recompute 1 \
  --warmup-steps 3 --steps 5 \
  --output runs/baselines/fsdp2/qwen30b/16K-recompute100
```

| Argument | Meaning |
|---|---|
| `--microbatch-tokens` | Tokens per microbatch per rank; must divide the global update across ranks |
| `--gradient-accumulation` | Optional explicit microbatches per update; otherwise derived from global tokens. Inconsistent values are rejected |
| `--global-tokens` | Tokens across all ranks per optimizer update, default 2,097,152 |
| `--sequence-length` | Independent sequence length, default 4,096 |
| `--recompute` | Fraction of evenly distributed layers checkpointed, from 0 to 1 |
| `--offload {none,cpu}` | Native FSDP2 offload of parameter shards, gradients, and optimizer state. CPU mode runs standard single-tensor AdamW on CPU; GPU mode uses fused AdamW |
| `--[no-]pin-memory` | Pin offloaded CPU shards; default on, applicable only to CPU offload |
| `--forward-prefetch N` | Explicit lookahead in transformer blocks; default 0 leaves native demand-driven forward behavior |
| `--backward-prefetch auto` or `N` | Native reverse-order backward prefetch, or explicit depth at least 1. Default `auto` |
| `--gradient-sync {per-microbatch,last-microbatch}` | Reduce each microbatch into owned FP32 shards (default), or keep full gradients and reduce on the final microbatch |
| `--[no-]reshard-after-forward` | Release each block's gathered weights after forward, or keep them through backward |
| `--[no-]reshard-after-backward` | Reshard between accumulation microbatches, default on. Always reshard on the final backward before optimizer updates |
| `--compile-blocks` | Enable `torch.compile` on each FSDP block; default off until GPU qualification |
| `--warmup-steps`, `--steps` | Excluded warmups and measured complete optimizer updates |
| `--allocator-cap-gib` | Optional PyTorch caching-allocator limit; excludes external allocations |
| `--cpu-threads` | Intra-op CPU threads per rank, including offloaded optimizer work; single-case default 1 |
| `--learning-rate`, `--weight-decay`, `--seed` | Explicit optimizer and synthetic-data initialization settings |
| `--nsys-capture` | Bracket measured steps with Nsight profiler start/stop; profiled timings are not throughput results |
| `--tiny --verify-reference` | Small numerical qualification; never use its timings as full-model performance |
| `--output` | New case directory; existing rank directories are never overwritten |

Global update geometry is always checked:

```text
global_tokens = world_size * microbatch_tokens * gradient_accumulation
```

Changing accumulation independently requires setting `--global-tokens` to the
matching value; the harness never silently changes the amount of training work.
For eight GPUs and 2,097,152 tokens, 4K/8K/16K/32K/64K/128K microbatches require
64/32/16/8/4/2 accumulation microbatches respectively.

The current public PyTorch API treats an empty backward-prefetch list as
**default prefetch**, not disabled prefetch. Accordingly, backward depth zero
is rejected rather than mislabeled as an off switch. Forward zero disables
explicit lookahead but retains native stream overlap. The root FSDP unit owns
embedding/head parameters and retains their gathered copy through backward.

## Sweep harness

The example matrix has six microbatch sizes, five whole-block recomputation
fractions, and two offload modes (60 cases). Inspect its expanded commands first:

```bash
python -m baselines.fsdp2.sweep \
  --config baselines/fsdp2/configs/qwen30b.json \
  --gpus 8 --output runs/baselines/fsdp2/qwen30b --dry-run
```

Run or resume exactly that manifest:

```bash
python -m baselines.fsdp2.sweep \
  --config baselines/fsdp2/configs/qwen30b.json \
  --gpus 8 --output runs/baselines/fsdp2/qwen30b --resume
```

Use `configs/qualification.json` for the six-case tiny numerical matrix first.
The driver launches a fresh single-node `torchrun` for each configuration,
with automatic worker restarts disabled. Launch the driver inside the GPU
allocation, preferably from tmux. It does not reserve GPUs itself.

Configuration JSON has `schema_version: 1`, a `defaults` mapping, and either a
Cartesian `grid` or explicit `cases` list. Keys use the CLI names with underscores.
Every trainer control above can be a grid axis. For example:

```json
{
  "schema_version": 1,
  "defaults": {"microbatch_tokens": 16384, "recompute": 1},
  "grid": {
    "offload": ["none", "cpu"],
    "forward_prefetch": [0, 1, 2],
    "backward_prefetch": ["auto", 1, 2],
    "reshard_after_forward": [true, false]
  }
}
```

An `allocator_cap_gib` grid can add memory limits. They remain allocator caps,
not guarantees on NCCL/external memory or a replacement for ShadowSpill's pool
budget. `last-microbatch` synchronization and retaining gathered parameters can
substantially increase GPU memory. CPU offload also consumes host memory.

| Sweep argument | Meaning |
|---|---|
| `--config`, `--output`, `--gpus` | Matrix, isolated output root, and GPUs on one node |
| `--python` | Python interpreter for torchrun, default the launcher's interpreter |
| `--dry-run` | Save expanded configs, exact commands, provenance, and index; no GPU launch |
| `--resume` | Continue the identical matrix, code, dependencies, and Python environment |
| `--retry-failed` | With resume, retry OOM/error/timeout/interrupted cases in new attempt directories |
| `--case-timeout` | Wall-clock seconds per case, including setup/compilation; default 3,600 |
| `--continue-on-error` | Continue after non-OOM failures. Default stops for diagnosis; device OOM cases are recorded and skipped |
| `--limit` | Maximum new attempts in this invocation, useful for bring-up |

Completed cases are skipped on resume. Failed cases retain exact settings,
tracebacks, command, and logs; nothing silently changes the batch or policy.
An output lock prevents concurrent launchers from using the same sweep root.
Code/environment changes require a new output root, preserving earlier evidence.

```text
output/
  README.md             # linked configuration/results table
  manifest.json         # full settings, source hashes and Python dependencies
  results.json          # machine-readable outcomes and comparable metrics
  cases/case-NNNN-HASH/
    config.json
    command.txt
    attempt-001/
      command.txt
      console.log
      outcome.json
      results.json      # only after all ranks finish
      rank-00000/       # request, result or failure; one directory per rank
```

Parameters are created on meta, sharded before materialization, and initialized
in place on GPU or CPU as requested. Compute buffers remain on GPU. By default,
gradients reduce-scatter after every microbatch and accumulate in FP32 owned
shards, avoiding the full-model gradients that deferred synchronization retains.
FP32 moments and master shards are checked after the first optimizer update.

Each case saves `rank-XXXXX/request.json`, `result.json` or `failure.json`, and an
all-rank `results.json`. Useful TFLOP/s uses the same Qwen shape-based count as
the existing ShadowSpill sweep and excludes recompute/optimizer/communication.
All measured losses must be finite. The summary requires every expected rank and
every measured update. Setup time, GPU allocated/reserved peaks, and host peak
RSS are retained separately from timed complete optimizer steps. Summed peak
RSS is a sum of per-process high-water marks, not a simultaneous node measurement.
