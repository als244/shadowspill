# Qwen3 MoE full training

This recipe uses the public build/profile/plan/execute pipeline with ordinary
sharded DP. The default model is Qwen3-30B-A3B: BF16 compute weights, FP32 gradients,
FP32 master weights and FP32 AdamW moments. Weights, gradients and optimizer data
are sharded across eight GPUs. It starts from random weights and reads real,
already-tokenized text from `train.bin` plus `meta.json`.

```bash
python -m workloads.recipes.text.train_qwen3_distributed \
  --data /path/to/qwen3-tokens --run-dir runs/qwen3-dp8
```

Launch this command once, without an outer torchrun. The orchestrator uses one
GPU for build/profile, hides GPUs for planning, and launches all eight workers
for communication calibration and execution. Defaults are 4K tokens, 64 sequences
per rank, five updates, and 77 GiB GPU / 96 GiB host spill per rank. The latter
requires 768 GiB of host pools across the node.

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

```bash
CUDA_VISIBLE_DEVICES= python -m workloads.recipes.text.train_qwen3_distributed \
  --run-dir runs/qwen3-dp8 --phase plan --execution-gib 60
```

The distributed `communicate` phase measures host copies and collectives, then
paired task/NCCL interference for each saved batch candidate. It saves
`overlap/batch-N.ssp` and merges the measurements into the corresponding profiles.
Planning uses these measured progress rates when transfers overlap computation;
the original isolated task timing and memory contracts stay intact. Changing
budgets reuses this evidence. Changing task implementations or the communication
setup requires compatible new measurements. See
[distributed calibration](../../../../docs/python/api/distributed.md#communication-calibration-and-cpu-planning)
for the public API and measurement limits.

The files have one purpose each:

| File | Responsibility | Saved output |
| --- | --- | --- |
| `build.py` | Meta model, local task compilation, DP declarations | `programs/batch-N.ssp` |
| `profile.py` | Task timing and memory on one representative GPU | `profiles/`, `profiled/` |
| `communicate.py` | Host copies, collectives, communication memory and paired task interference | `communication/`, `overlap/`, enriched `profiles/` and `profiled/` |
| `plan.py` | All factor sizes/orderings under the supplied memory budget | `search.json`, `plans/`, selected `plan.ssp`, `report/` |
| `execute.py` | Fresh worker, bounded initialization, Trainer updates/checkpoint | `training/rank-NNNNN/` |
| `../initialization.py` | Identical random tensor values across shards/chunk sizes | Fills Executor buffers |

The default candidates contain 1, 2, 4, 8, 16 and 32 sequences per example;
planning selects their execution order without rebuilding. Inputs are rank-local
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
For independent chunk-size and graph-pair-policy candidates, use the
[quickstart candidate search](../../../../benchmarking/quickstart.md).

The eight host pools alone require 768 GiB. The test used a 980 GiB node host
limit to allow process memory and checkpoint page cache; observed whole-job host
peak was 943.93 GiB. The public execution phase, including initialization and
writing its full checkpoint to local SSD, took 595 s. Checkpoint throughput is
separate from steady training throughput.
