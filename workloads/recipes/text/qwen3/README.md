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

Each phase also runs independently with `--phase build`, `profile`, `communicate`,
`plan`, or `execute`. Start build with the data path and configuration. Subsequent
phases read saved settings; a new build requires a fresh directory. Change memory
budgets only during planning:

```bash
CUDA_VISIBLE_DEVICES= python -m workloads.recipes.text.train_qwen3_distributed \
  --run-dir runs/qwen3-dp8 --phase plan --execution-gib 60
```

The files have one purpose each:

| File | Responsibility | Saved output |
| --- | --- | --- |
| `build.py` | Meta model, local task compilation, DP declarations | `programs/batch-N.ssp` |
| `profile.py` | Task timing and memory on one representative GPU | `profiles/`, `profiled/` |
| `communicate.py` | Host copies, declared collectives and communication memory | `communication/batch-N.ssp` |
| `plan.py` | All factor sizes/orderings under the supplied memory budget | `search.json`, `plans/`, selected `plan.ssp`, `report/` |
| `execute.py` | Fresh worker, bounded initialization, Trainer updates/checkpoint | `training/rank-NNNNN/` |
| `initialize.py` | Identical random tensor values across shards/chunk sizes | Fills Executor buffers |

The default candidates contain 1, 2, 4, 8, 16 and 32 sequences per example;
planning selects their execution order without rebuilding. Inputs are rank-local
and disjoint. Loss normalization uses the global token count. Random initialization
preserves the model's distributions, with a stable per-tensor RNG stream rather
than the old module-reset traversal order.

`--model qwen3moe-tiny --ranks 1` supports local pipeline checks. Single-rank
profiling also measures host copies, so it needs no communicate phase. Such a
check does not qualify the 30B model's performance or memory use on eight GPUs.
The underlying common DP Executor is qualified on two H100s; qualification of
this migrated MoE recipe on multiple GPUs remains separate. External autotuners
such as Quack still need saved-choice integration before claiming complete
cache-independent kernel replay.
