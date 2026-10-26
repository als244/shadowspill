# ShadowSpill over the catalog models

This engine uses the public build → profile → plan → execute pipeline with the
same objective, precision settings, data and measurement clock as the other
baseline engines. Each phase has its own process. Execution loads a saved Plan
into `Trainer(plan)`; preparation never opens ShadowSpill pools.

Run from the repository root with the ShadowSpill environment and this checkout's
native libraries:

```bash
python -m baselines.common.launch --engine shadowspill --gpus 1 \
  --output results/llama \
  --model llama3-8b --tokens-per-step 65536 --tokens-per-microbatch 8192 \
  --execution-gib 16 --spill-gib 128
```

The tiny numerical matrix is
`baselines/engines/shadowspill/configs/qualification.json`; run it with
`python -m baselines.sweep --engine shadowspill --config <file> --output <dir> --gpus 1`.

## Phase responsibilities

| Module | Responsibility |
| --- | --- |
| `train.py` | Compose preparation and launch fresh measurement workers |
| `factory.py` | Catalog model, local objective, precision and microbatch examples |
| `preparation.py` | Reuse quickstart's build/profile/calibration/CPU-plan workers |
| `execute.py` | Load the selected Plan, check resources, initialize and admit |
| `initialize.py` | Fill bounded model chunks from initialization or a checkpoint |
| `measurement.py` | Run warmup and measured updates under the common baseline clock |
| `reference.py` | Export tiny-model tensors for the independent numerical oracle |
| `reporting.py` | Record budgets, peaks and measured-versus-simulated results |

The model starts on meta. The catalog initializer fills Executor-provided chunks
and preserves derived buffers. It matches the model's initialization distributions,
with chunk-independent deterministic seeds; it does not reproduce
`reset_parameters`' traversal RNG sequence. Checkpoint readers also fill chunks
directly. Master weights and optimizer state use the saved Program's initializers.

Distributed cases use `Parallelism.dp(ranks_per_replica=world_size)`: weights,
gradients and optimizer data are sharded. Build and compute profiling use one
representative GPU. Communication calibration and execution use all ranks.
The objective divides local loss by world size before gradient summation.

## Settings and search

The common model/data/precision/run flags are described in
[the FSDP2 guide](../fsdp2/README.md). ShadowSpill adds:

| Flag | Meaning |
| --- | --- |
| `--execution-gib` | Total GPU allowance, including fixed costs and external headroom |
| `--spill-gib` | Per-rank pinned spill capacity |
| `--external-headroom-gib` | GPU external reserve inside the allowance; default 0.5 |
| `--host-headroom-gib` | Per-rank host allowance checked beside spill storage; default 2 |
| `--tokens-per-microbatch` | Fix one size, or omit to search whole-sequence divisors |
| `--min-tokens-per-microbatch`, `--max-tokens-per-microbatch` | Bound candidate sizes |
| `--orderings` | Search all depth/breadth factors, or only `depth-first` |
| `--resolution-options` | Planner save/recompute shares: `quarters`, `halves`, `eighths`, or fractions |
| `--artifact-store` | `default`: artifacts beside the case/shared preparation; `none`: disable sharing; a path: custom parent |

The artifact-location flag retains its baseline spelling. There is no global
artifact-store dependency. Compiler caches stay outside these output folders.

Planning chooses geometry, ordering and save/recompute alternatives together.
`--recompute` and `--allocator-cap-gib` are refused: the planner controls
recomputation and Executor enforces the total GPU allowance.

[Precision settings](../../PRECISION.md) retain model/gradient/master/moment
dtypes, optional fused narrow-gradient accumulation and explicit rounding.
Unspecified rounding is nearest. The numerical oracle checks the requested
compute dtype, master weights and Adam moments.

## Saved results and budget reuse

Each case writes `case.json`, an independent executable `plan.ssp`, per-rank
requests/results/failures, and the common `results.json`/`row.json`.
Preparation writes `programs/`, `profiles/`, `profiled/`, `communication.ssp`,
`planning/sweep.ssp`, selected `plans/`, `report/` and one log per phase.
Tensor/code payloads are packed into archives.

Single-rank sweeps group settings that differ only in execution budget. They
save preparation under `preparations/<identity>/artifact`. Adding a budget
reuses the saved profiles and runs CPU planning; it does not require a larger
profiling pool. A case with an existing winning Plan skips preparation/search
and loads it in a fresh worker. Each case keeps its own Plan archive, so later
replanning cannot change an earlier result's executable.

Source, environment, host and settings identify shared preparations. Reference
checks, distributed cases and `artifact_store=none` use individual preparation.
Typed candidate OOMs remain in the planning report; other build/profile errors
stop the case. An infeasible budget is recorded as the sweep's memory boundary.

## Measurement and numerical checks

Only `Trainer.step` and its loss/metric collection are inside measured updates.
Warmup, model loading, compilation, task/communication profiling, CPU planning
and admission are outside. Nsight's capture starts at the first measured update.

Reference verification is tiny-model-only and never a throughput result. After
each update it exports bounded tensor chunks, reconstructs complete values and
checks loss, masters and Adam moments against ordinary PyTorch AdamW with a
whole-model compiled objective. It writes no intermediate checkpoint snapshots.
Reference compilation/checks are outside the measurement window. Verification
uses short task-profile windows; throughput runs retain normal windows.

Rows report measured execution-pool occupancy, total GPU allowance, both pool
statistics, planned spill peaks and observed device usage. Host peak for
comparisons is the sum of planned spill peaks; RSS is recorded separately.
The median update time excludes warmup. Simulator error is
`100 * (measured / predicted - 1)`. Full simulated timelines are in `report/`.
