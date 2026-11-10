# Qwen3 MoE full training

This recipe uses the [shared Python runner](../../README.md) with sharded DP or
expert parallelism. The default model is Qwen3-30B-A3B: BF16 compute weights, FP32 gradients,
FP32 master weights and FP32 AdamW moments. DP shards weights, gradients and
optimizer data across eight GPUs. EP owns each expert shard locally, replicates
shared weights and gradients, and shards the shared optimizer state. The recipe
starts from random weights and reads real,
already-tokenized text from `train.bin` plus `meta.json`.

Start with the [commented Python script](example.py):

```python
from workloads.recipes import Model, train

model = Model.from_config("qwen3moe-30b-a3b")
train(model, {
    "data": "/data/fineweb-qwen3",
    "run_dir": "runs/qwen3-dp8",
    "ranks": 8,
    "parallelism": "dp",
    "sequence_length": 4096,
    "sequences_per_step": 64,  # 2,097,152 global tokens/update.
    "microbatch_sizes": [4, 16],
    "execution_gib": 20,
    "spill_gib": 96,
})
```

Launch this script once, without an outer torchrun. DP uses one
GPU for build/profile, hides GPUs for planning, and launches all eight workers
for communication calibration and execution. Defaults are 4K tokens, 64 sequences
per rank, five updates, and 77 GiB GPU / 96 GiB host spill per rank. The latter
requires 768 GiB of host pools across the node.

Set `parallelism="ep"` to distribute experts across the rank group. EP build and
profiling require every resource participant; each microbatch candidate gets its
own token-capacity resource. EP execution chunks are currently refused because
the provider has no declared axis-chunk contract. This is independent of DP
execution chunking and checkpoint recomputation. The new public EP campaign is
pending; the historical performance table below covers DP only.

For full-training DP, chunk candidates currently split the surrounding dense
projections. Its opaque MoE operations contain sequence-length metadata and
nonlinear statistics, so they remain whole-sequence barriers. `moe_chunk_sizes`
does not split those opaque operations. The LoRA adapters have separate provider
contracts that support chunked expert execution.

Profiling reserves 768 MiB of external GPU headroom (configurable with
`--external-headroom-mib` when creating the run), in addition to calibrated
communication overhead. This is included in the total GPU budget. Library and
backend setup on an H100 SonicMoE worker takes more than 512 MiB outside the
pool; the margin reduces the memory available to planning rather than relaxing
runtime admission. Changing this reservation requires fresh profiles.

Each phase also runs independently with `--phase build`, `profile`, `communicate`,
`plan`, or `execute`. Start build with the data path and configuration. Subsequent
phases read saved settings; a new build requires a fresh directory. Change memory
budgets only during planning:

```python
saved = {"run_dir": "runs/qwen3-dp8"}
train(model, {**saved, "execution_gib": 40}, phase="plan")
train(model, saved, phase="execute")
```

The distributed `communicate` phase measures host copies and collectives, then
paired task/NCCL interference for each saved batch candidate. It saves
`overlap/{candidate}/training.ssp` and merges the measurements into the profiles.
Planning uses these measured progress rates when transfers overlap computation;
the original isolated task timing and memory contracts stay intact. Changing
budgets reuses this evidence. Changing task implementations or the communication
setup requires compatible new measurements. See
[distributed calibration](../../../../docs/python/api/distributed.md#communication-calibration-and-cpu-planning)
for the public API and measurement limits.

The shared workers have one purpose each; recipe modules delegate to them:

| File | Responsibility | Saved output |
| --- | --- | --- |
| `build.py` | Adapter setup, task compilation, resource declarations | `programs/{candidate}/training.ssp` |
| `profile.py` | Task timing and memory on one representative GPU | `profiles/`, `profiled/` |
| `communicate.py` | Host copies, collectives, communication memory and paired task interference | `communication/`, `overlap/`, enriched `profiles/` and `profiled/` |
| `plan.py` | Microbatch, chunk, retention and ordering search under memory budgets | `search.json`, `selected.json`, `plans/training.ssp`, `reports/` |
| `execute.py` | Fresh worker, bounded initialization, Trainer updates/checkpoint | `training/rank-NNNNN/` |
| `../initialization.py` | Identical random tensor values across shards/chunk sizes | Fills Executor buffers |

`--microbatch-sizes` lists the exact candidates to build and profile. The default
is `1 2 4 8 16 32`; for example, `--microbatch-sizes 4 16` builds only those two
sizes. Every size must be a positive integer dividing `--sequences-per-step`.
Repeated entries are built once. When reducing the step size, supply a list of
valid sizes as well. Changing the list requires a fresh build directory; later
phases reuse the saved list. The former maximum-size option is removed.
Planning searches all factor orderings for each size without rebuilding.
Inputs are rank-local
and disjoint. Loss normalization uses the global token count. Random initialization
draws Hugging Face's initial distributions, from a stable per-tensor RNG stream
rather than the module-reset traversal order.

`--model qwen3moe-tiny --ranks 1` supports local pipeline checks. Single-rank
profiling also measures host copies, so it needs no communicate phase. Such a
check does not qualify the 30B model's performance or memory use on eight GPUs.
The underlying common DP Executor is qualified on two H100s, and the recipe's
test runs the tiny model through every phase on one and two GPUs, with separate
compiler caches for each phase. The recipe opts into MLOps' supported
[operator preparation](../../../../docs/operator-preparation.md): BF16 SonicMoE
builds archive their selected provider kernels and tuning choices. Profiling
and execution load those artifacts without retuning inside a task. An eager
compiler cache is not a substitute for these saved bindings. Other providers
keep their own preparation contracts.

## Measured eight-H100 run

The November 9, 2026 full-model check used FineWeb-Edu text encoded with the
official Qwen tokenizer, 4,096-token sequences, 64 sequences per rank per update,
and **20 GiB execution / 96 GiB pinned host spill per rank**. All 27 batch/order
combinations were considered; 25 were admitted. Both batch-32 choices exceeded
the required input/output capacity. The lowest predicted duration selected
batch 16 with `1x4rp` ordering.

| Same-node comparison | Predicted update | Measured update |
| --- | ---: | ---: |
| Original `2x2rp` Plan without paired rates | 18.163 s | 19.568 s |
| Recalibrated `2x2rp` Plan | 19.934 s | 19.416 s |
| Recalibrated `1x4rp` Plan | 19.344 s | 18.705 s |
| Selected `1x4rp` through the normal public CLI | 19.344 s | **18.732 s** |

Each case completed seven updates. Measured time is the median maximum-rank
Trainer wall time over updates 2–6, excluding startup, final tracing and
checkpoint I/O. The public winner reached approximately **111,956 tokens/s**;
measured time was 3.2% below prediction. Calibration used independent paired
task/collective measurements and refreshed communication measurements, with no
correction fitted to the training step. All eight public-run checkpoints passed
metadata, optimizer-counter and finite-state sample checks.

This search used the default graph-pair policy and no explicit axis-chunk policy.
It searched batch/order choices and PressureFit's placement/variant decisions.
The current shared runner also accepts independent `chunk_sizes`,
`moe_chunk_sizes` and `graph_pair_policies` lists for DP; see its
[search contract](../../README.md#search-and-saved-phases). The table above does
not qualify those additional candidates.

The eight host pools alone require 768 GiB. The test used a 980 GiB node host
limit to allow process memory and checkpoint page cache; observed whole-job host
peak was 943.93 GiB. The public execution phase, including initialization and
writing its full checkpoint to local SSD, took 595 s. Checkpoint throughput is
separate from steady training throughput.
