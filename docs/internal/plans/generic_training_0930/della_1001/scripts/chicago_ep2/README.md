# Chicago OLMoE recipe on Della EP2

Run in the reserved `codex` pane:

```bash
bash docs/internal/plans/generic_training_0930/della_1001/scripts/chicago_ep2/launch.sh
```

`train.py --config FILE --steps N` changes the configuration or short-run length.
The LR schedule always uses `schedule_total_steps`, independently of `steps`.

## Matching Chicago

- 16 layers; width 1,024; 16 attention/KV heads of width 64.
- 192 experts per layer, top-4, expert width 1,280; no shared expert.
- GPT-2 tokens, model vocabulary 50,304; maximum sequence length 2,048.
- 1,048,576 token slots per global update; normalization uses actual trained
  targets across both ranks. Each rank owns different packed microbatches.
- BF16 weights, gradients, and AdamW moments; no master parameters; stochastic
  rounding for parameter and moment stores, weight decay zero.
- LR 3e-4, 100 warmup updates, cosine decay to 3e-5 over 9,537 updates.
- Evaluation every 100 updates, checkpoints every 500 and at the final update.
- Cross-entropy, auxiliary loss, routing summaries, parameter/gradient health,
  step time and throughput are recorded. W&B has per-rank and aggregate runs.

This is a fresh initialization with the same distributions, not a replay of
Chicago's current checkpoint. Distributed reductions and local microbatch
routing statistics can differ from Chicago's single-device execution.

## Resource choices

Planning searches 16K, 32K, 64K, 128K and 256K tokens per rank per microbatch,
with 32, 16, 8, 4 and 2 accumulated microbatches per rank respectively. The workload owns
one correctly sized MoonEP token buffer shared by every layer, plus shared
expert publication/reduction banks. Each candidate runs in fresh worker
processes so its communication resources are released before the next capacity.
The ordinary Trainer searches recomputation fractions and depth-first/breadth-two
factor orderings for each capacity; the outer experiment compares their admitted
predicted step times. `planning_max_breadth=2` bounds CPU search time for this
reservation; set it to `null` to search every factor ordering.
Candidate progress is saved immediately and completed candidates are resumable.

Distinct expert parameters occupy 11.25 GiB per rank across all 16 layers.
Communication banks are allocated once: 1.40625 GiB BF16 home/replica weights
and 1.40625 GiB FP32 replica-gradient scratch, totaling 2.8125 GiB per GPU.
Returned expert gradients and optimizer state are BF16. The execution budget is
50 GiB with 6 GiB external headroom, leaving 44 GiB for the managed execution
pool, and 80 GiB host spill per rank. Physical measurements at model construction
and admission are recorded for each capacity. No MoonEP-specific behavior is
added to ShadowSpill's planner or allocator.

The data subset is under
`/home/as1669/storage/datasets/fineweb_edu_gpt2_chicago_sample`.
It contains the first 268,434,637 tokens of Chicago's prepared train stream,
ending at a document boundary, plus its validation stream. Packing uses the
ordinary text recipe with `long_documents="splice"` and window 1,024.

All run artifacts are under
`/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-olmoe12b-ep2-1002`:
console output and `planning-progress.json`; `candidates/tokens-N/` contains
each configuration’s per-rank artifact stores and graph pairs, plans, startup
diagnostics, metric JSONL files, W&B files, and checkpoints. W&B HTTPS is relayed
through the Della head node because the compute node has no public DNS/network.
Each rank's `plan-diagnostics.json` retains both graph-pair variants, their
input/mutation/output/workspace sizes, measured runtimes, and object mappings.
`launch.sh` expects that loopback SSH relay on port 18375; online machines can
run `train.py` directly without it.

## Resuming the October 2 capacity retry

The 16K training baseline is complete. After the MoonEP compiler and generic
profiling-lifetime fixes, retry the larger capacities in `codex:0.0`:

```bash
bash docs/internal/plans/generic_training_0930/della_1001/scripts/retry_chicago_capacities.sh
```

This preparation-only command defaults to 32K, 64K, 128K and 256K. Use
`--tokens 131072 262144` to select particular capacities. It skips passed
cases and preserves per-case `result.json`, console logs and complete artifact
stores under `~/storage/shadowspill/generic_training_0930/della_1001/chicago-ep2-capacity-fixes-1002`.
Interrupted or failed cases are retried. Worker processes are recreated between
capacities. Training uses the admitted plan with the lowest predicted step time;
online W&B authentication and the relay are checked before launching it.

After every candidate has finished, launch the fastest admitted one with:

```bash
/home/as1669/.conda/envs/shadowspill/bin/python -u \
  docs/internal/plans/generic_training_0930/della_1001/scripts/train_best_capacity.py \
  --steps 100
```

`--select-only` prints the choice without running it. A longer run can use
`--data /path/to/prepared/tokens` and a larger `--steps`; the launcher checks
that the prepared prefix is long enough. It retains the original LR schedule,
uses the winning artifact store, refuses to overwrite previous training, and
places the aggregate/rank W&B runs in the same experiment group. Checkpoints,
metrics and `training-console.log` remain beneath the winning case directory.

For an allocation-length run, use `--until-allocation-end` instead of `--steps`:

```bash
/home/as1669/.conda/envs/shadowspill/bin/python -u \
  docs/internal/plans/generic_training_0930/della_1001/scripts/train_best_capacity.py \
  --until-allocation-end \
  --data /home/as1669/storage/datasets/fineweb_edu_gpt2_chicago_2b \
  --training-outdir /path/to/fresh/run
```

`--training-outdir` separates a fresh run's logs, checkpoints and W&B files
from its reused planning artifacts. Evaluation is admitted and executed once
before real updates, so an evaluation planning failure cannot first appear at
the 100-update cadence. This preflight is not logged as a training update.

After evaluation preflight, the normal warmup and traced zero-LR step run once. Their
slower-rank runtime and the predicted runtime determine a common update count,
with 15% timing headroom and a ten-minute checkpoint/sync reserve before the
Slurm deadline. The count is bounded by the prepared data and original LR
horizon. This estimates a safe run length; it does not change the training
engine or place a timer inside compiled tasks. `training-window.json` records
the decision on each rank. The two-billion-token prefix has been checked
against the complete original planning sample and ends at a document boundary.

## Validation before launch

Both save and recompute passed a two-block EP2 test with a BF16 router and one
shared communication buffer: three updates, identical replicated parameters,
and an independent PyTorch full-model loss/gradient reference. Data checks
verify disjoint microbatch assignment, global target normalization and the
unchanged LR schedule endpoints.

## Nsight profiling

The profiling client restores the completed run's step-900 checkpoint, warms up
three updates, then captures five full training updates with EP2 and 128K tokens
per rank per microbatch. Run it in the allocated `codex` pane:

```bash
/home/as1669/.conda/envs/shadowspill/bin/python -u \
  docs/internal/plans/generic_training_0930/della_1001/scripts/profile_chicago_ep2.py \
  --outdir /path/to/fresh/profile
```

Use `--config`, `--checkpoint`, `--steps` and `--warmup` to change those inputs.
The existing planned-call option `profiler_annotations=True` enables ShadowSpill
task/transfer ranges. Nsight captures CUDA, NVTX, OS runtime and cuBLAS events,
plus both GPUs' device metrics. Setup and warmup occur before collection starts.
The profile writes no new checkpoints or W&B runs. Commands, validation and the
completed report are documented in [NSYS_128K.md](../../NSYS_128K.md).
