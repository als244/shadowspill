# Qwen3 MoE full training

This recipe uses the [shared Python runner](../../README.md) with sharded DP or
expert parallelism. The example selects Qwen3-30B-A3B: BF16 compute weights, FP32 gradients,
FP32 master weights and FP32 AdamW moments. DP shards weights, gradients and
optimizer data across eight GPUs. EP owns each expert shard locally, replicates
shared weights and gradients, and shards the shared optimizer state. The recipe
starts from random weights and reads real,
already-tokenized text from `train.bin` plus `meta.json`.

Start with the [commented Python script](example.py):

```python
from shadowspill_workloads.recipes import Model, train

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
    "learning_rate": 3e-4,
    "eval_every": 0,  # Enable when val.bin is available.
})
```

Launch this script once, without an outer torchrun. DP uses one
GPU for build/profile, hides GPUs for planning, and launches all eight workers
for communication calibration and execution. The script explicitly supplies the
eight-GPU geometry; `Model.from_config` otherwise uses the common
[single-GPU defaults](../../README.md#data-defaults-and-search). Its 96 GiB spill
ceiling per rank permits up to 768 GiB of host pools across the node.

Set `parallelism="ep"` to distribute experts across the rank group. EP build and
profiling require every resource participant; each microbatch candidate gets its
own token-capacity resource. EP execution chunks are currently refused because
the provider has no declared chunkable axes. This is independent of DP
execution chunking and checkpoint recomputation. Both DP8 and EP8 have completed
the budget sweep below; its EP runs use unchunked execution.

For full-training DP, chunk candidates currently split the surrounding dense
projections. Its opaque MoE operations contain sequence-length metadata and
nonlinear statistics, so they remain whole-sequence barriers. `moe_chunk_sizes`
does not split those opaque operations. The LoRA adapters have separate provider
contracts that support chunked expert execution.

Profiling reserves 768 MiB of external GPU headroom (configurable with
`external_headroom_mib` when creating the run), in addition to calibrated
communication overhead. This is included in the total GPU budget. Library and
backend setup on an H100 SonicMoE worker takes more than 512 MiB outside the
pool; the margin reduces the memory available to planning rather than relaxing
runtime admission. Changing this reservation requires fresh profiles.

Each phase also runs independently with `phase="build"`, `"profile"`, `"communicate"`,
`"plan"`, or `"execute"`. Start build with the data path and configuration. Subsequent
phases read saved settings; a new build requires a fresh directory. Change memory
budgets only during planning:

```python
saved = {"run_dir": "runs/qwen3-dp8"}
train(model, {**saved, "execution_gib": 40}, phase="plan")
train(model, saved, phase="execute")
```

The distributed `communicate` phase measures host copies and collectives. For
DP it also measures paired task/NCCL interference for each saved batch candidate,
saves `overlap/{candidate}/training.ssp` and merges the measurements into the
profiles. Planning uses these measured progress rates when transfers overlap
computation; the original isolated task timing and memory contracts stay intact.
EP task timings include their internal collectives; extra EP contention
calibration is not implemented. Changing budgets reuses compatible evidence.
Changing task implementations or the communication setup requires compatible
new measurements. See
[distributed calibration](../../../../../../../docs/python/api/distributed.md#communication-calibration-and-cpu-planning)
for the public API and measurement limits.

The shared workers have one purpose each; recipe modules delegate to them:

| File | Responsibility | Saved output |
| --- | --- | --- |
| [Shared build](../../../../../../engine/src/shadowspill/training/preparation/build.py) | Adapter setup, task compilation, resource declarations | `programs/{candidate}/training.ssp` |
| [Shared profile](../../../../../../engine/src/shadowspill/training/preparation/profile.py) | Task timing and memory; one representative DP worker or all EP participants | `profiles/`, `profiled/` |
| [Shared communicate](../../../../../../engine/src/shadowspill/training/preparation/communicate.py) | Host copies, collectives, communication memory and supported paired task interference | `communication/`, `overlap/`, enriched `profiles/` and `profiled/` |
| [Shared plan](../../../../../../engine/src/shadowspill/training/preparation/plan.py) | Microbatch, chunk, retention and ordering search under memory budgets | `search.json`, `selected.json`, `plans/training.ssp`, `reports/` |
| [Shared execute](../../../../../../engine/src/shadowspill/training/preparation/execute.py) | Fresh worker, bounded initialization, Trainer updates/checkpoint | `training/rank-NNNNN/` |
| [Text initialization](../initialization.py) | Identical random tensor values across shards/chunk sizes | Fills Executor buffers |

`microbatch_sizes` lists the exact positive divisors of `sequences_per_step` to
build and profile. Repeated entries are built once. Changing the list requires a
fresh run; later phases reuse the saved list. Planning searches all factor
orderings for each size without rebuilding.
Inputs are rank-local
and disjoint. Loss normalization uses the global token count. Random initialization
draws Hugging Face's initial distributions, from a stable per-tensor RNG stream
rather than the module-reset traversal order.

`Model.from_config("qwen3moe-tiny")` with short sequences supports local pipeline checks. Single-rank
profiling also measures host copies, so it needs no communicate phase. Such a
check does not qualify the 30B model's performance or memory use on eight GPUs.
The underlying common DP Executor is qualified on two H100s, and the recipe's
test runs the tiny model through every phase on one and two GPUs, with separate
compiler caches for each phase. The recipe opts into MLOps' supported
[operator preparation](../../../../../../../docs/python/advanced/operator-preparation.md): BF16 SonicMoE
builds archive their selected provider kernels and tuning choices. Profiling
and execution load those artifacts without retuning inside a task. An eager
compiler cache is not a substitute for these saved bindings. Other providers
keep their own preparation contracts.

## Eight-H100 budget sweep

The November 10–11 campaign completed seven updates at every budget with
4,096-token sequences, 64 sequences per rank and 2,097,152 global tokens per
update. Both layouts used real tokenized FineWeb-Edu data, random host
initialization, BF16 compute weights, FP32 gradients/master weights/Adam moments,
and a 96 GiB maximum pinned spill pool per rank. Evaluation and checkpoint I/O
were disabled for this throughput protocol. Measured time is the median of
the slowest rank's `Trainer.step` time over updates 2–7.

| Device budget per rank | DP8 predicted | DP8 measured | EP8 predicted | EP8 measured |
| --- | ---: | ---: | ---: | ---: |
| 20 GiB | 18.313 s | 18.712 s | 23.105 s | 25.390 s |
| 40 GiB | 17.761 s | 18.556 s | 22.964 s | 25.183 s |
| 77 GiB | 17.619 s | 18.181 s | 22.500 s | 24.540 s |

DP searched microbatch sizes `[8, 16, 32]`, projection chunks `[0, 1024]` and all
three graph-pair policies: 18 built/profiled Programs. EP searched the same
microbatch and policy lists with unchunked execution: nine Programs. Both
searched factor orderings and crossed tail counts `[0, 1, 2, 4, 8, 16]` with
resolution options `["0", "1/4", "1/2", "3/4", "1"]`. All winners selected
`cost-input` and unchunked execution. DP selected batch 16 / `1x4rp` at 20 GiB
and batch 32 / `1x2rp` at 40 and 77 GiB. EP selected batch 16 at every budget,
with `2x2rp` at 20/40 GiB and `4x1rp` at 77 GiB.

The DP record uses engine `17c48169`; EP uses `af404765`; both use MLOps
`403cb240`. These are results at the recorded revisions, not measurements of
every later change. All ranks completed their updates with finite losses,
consistent global loss reductions and admitted spill capacities. These short
throughput runs do not establish convergence or a full tensor-reference check.
EP measured about 9–10% above prediction. A subsequent seven-update replay of
the 20 GiB EP Plan measured 25.292 s versus 23.105 s predicted, with two Nsight
capture steps on every rank and a separate final runtime trace. It used engine
`cd197853` and the same provider `403cb240`.

The final runtime trace places about 2.025 s of rank 0's prediction error inside
task durations and only 0.015 s in additional inter-task idle. EP dispatch and
combine kernels are substantial parts of those tasks, but this does not identify
a particular kernel or contention mechanism as the cause of the prediction gap.
Additional EP contention calibration remains deferred. Capture start/stop and
report conversion add considerable time outside steady updates; they are
excluded from the throughput numbers above.

## Earlier 20 GiB comparison

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
[search contract](../../configuration.md#search-and-saved-phases). The table above does
not qualify those additional candidates.

The eight host pools alone require 768 GiB. The test used a 980 GiB node host
limit to allow process memory and checkpoint page cache; observed whole-job host
peak was 943.93 GiB. The public execution phase, including initialization and
writing its full checkpoint to local SSD, took 595 s. Checkpoint throughput is
separate from steady training throughput.
