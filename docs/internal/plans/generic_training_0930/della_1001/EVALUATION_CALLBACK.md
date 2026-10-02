# Evaluation callback partitioning: October 2

## Observed failure

The 128K/rank EP2 run completed 100 updates at a median 6.412 seconds/update.
Its first scheduled evaluation failed during physical admission, before an
evaluation invocation ran. Both ranks reported `PlanInfeasibleError` with
`constraint: unplaceable`. The 50 GiB execution budget and 80 GiB spill pool
were unchanged. No checkpoint existed yet: checkpoint cadence was 500 updates.
Training and W&B evidence is retained in `tokens-131072/` under the capacity
sweep's storage directory. The failed run is not a completed long run.

## Root cause and generic fix

`Forward` selects a caller's `forward_fn` on a capture-only model copy. A
shallow view of that copy lets the callback call the original `model(...)`
without recursively invoking itself. Strict PyTorch Export sometimes describes
the shared child modules through this view's closure path, for example
`forward.__func__.__closure__[1].cell_contents.blocks.0`, although the registered
name is `blocks.0`.

The automatic partition policy matches registered names. None of those closure
paths matched, so evaluation became one whole-model forward task. Training's
objective wrapper retained correct paths and was partitioned normally.

The fix resolves Export's module-identity stack keys back to the capture
model's registered names before archiving and partitioning. The capture view
maps to the root. It changes provenance only: operations, state names, tensors
and communication remain unchanged. It applies to any model using a forward
callback, including custom partition policies; it has no expert-layer rules.

CPU regression: an ordinary repeated Linear network using a forward callback
lost all automatic boundaries before the fix. The regression now verifies four
stages, unchanged state names, equal numerical output, and one stage when the
caller explicitly chooses `partition="whole"`. All 97 capture/partition checks and
the production file's mypy/Ruff checks pass. The full EP2 model captures 18
evaluation tasks, passes physical admission, and evaluates on both GPUs.

## Memory evidence and separate limitation

Rank 0's archived whole-model evaluation has one executable task:

- Input objects: 12,429,432,836 bytes (11.576 GiB).
- Measured workspace: 5,892,744,672 bytes (5.488 GiB).
- Logical simulated peak: 18,322,202,172 bytes (17.064 GiB).
- Physical layout required: 118,916,001,628 bytes (110.749 GiB).
- Available layout capacity: 46,515,901,448 bytes (43.321 GiB).

These physical-layout bytes were a planning requirement, not an actual device
allocation. The current fixed-layout derivation holds each distinct task-local
allocation slot for the whole task. It honors exact recorded slot reuse, but
many differently sized allocations across sixteen layers acquire separate slots.
This overreservation is a separate generic limitation for large unpartitioned
tasks. Restoring automatic boundaries avoids it here. Improving intra-task
workspace packing needs its own design and runtime correctness checks; this
change does not claim to solve that broader problem.

`scripts/replay_eval_admission.py` reproduces the failure on the head node from
the archived program and measured allocation trace. Logical planning passes;
physical placement fails at both the original capacity and an 80 GiB trial.
The replay logs, normalized facts and diagnostics live beside the original
rank artifacts. `SHADOWSPILL_LAYOUT_TRACE=1` records the byte derivation.

## Retry

`training-128k-v2/` uses the selected case's compilation/profile store and a
fresh output directory. Its client evaluates once before W&B/training starts,
then performs the usual zero-LR diagnostics. Evaluation still runs every 100
training updates and checkpointing every 500 plus the final update. The
original 9,537-update LR schedule is unchanged. Allocation-sized run length is
recalculated after preparation, with ten minutes and 15% timing headroom.

Tübingen passed suite, numerical 5/5 and performance 3/3 for the earlier
compiler/profiling fixes. Its suite and numerical gates also passed for this callback correction:
1,110 Python checks, one skip, 48 CUDA CTests, 20 neutral CTests, and numerical
5/5 against the existing references. Tested source hashes match Della exactly.

At 12:53 UTC the retry had completed eleven training updates with finite loss.
Its startup trace selected 900 updates, leaving the checkpoint/sync reserve.
Median update after the first: 6.425 seconds. W&B's API confirmed the aggregate
and both rank runs were running with live metrics:

- Aggregate: https://wandb.ai/andrew-sheinberg-princeton-university/shadowspill-della-ep-training/runs/26c6iq9k
- Rank 0: https://wandb.ai/andrew-sheinberg-princeton-university/shadowspill-della-ep-training/runs/hcmiagvb
- Rank 1: https://wandb.ai/andrew-sheinberg-princeton-university/shadowspill-della-ep-training/runs/z7xtua5y

The head-node watcher verified its wakeup connection and delivered the
10-update milestone. These were preliminary measurements; the completed result
is recorded below. Tübingen qualification for the callback change has passed.

At update 100 the scheduled evaluation completed on both ranks (aggregate loss 6.43952, 2.37 seconds) and training continued. The first 100 training losses differ from the preserved pre-fix run by at most 0. The median update time is 6.410 seconds versus 6.398 seconds predicted. Evidence: `training-128k-v2/milestone-0100.json`.

At 13:19 UTC training had reached update 248, with aggregate loss 4.849. Scheduled evaluations at 100 and 200 passed. The qualification summary and tested source hashes are saved in `evidence/callback_qualification.json`.

## Completed validation

The fresh run completed all 900 updates on both ranks at approximately
14:32 UTC. All nine scheduled evaluations passed, including the final
evaluation loss of 3.80065. Median training step time was 6.398974 seconds,
versus the planning prediction of 6.398133 seconds. Training loss decreased
from 11.15658 to 3.69174. Checkpoints at 500 and 900 are saved for both ranks;
the final manifest and archive central directories are readable. Full checkpoint
restore was not part of this run.

The aggregate and both per-rank W&B runs report `finished` with update 900.
`training-128k-v2/final-summary.json` and the tracked copy
`evidence/ep2_training_900.json` contain the final metrics and validation details.
The source fix is published as `f77950ee`; the experiment/preflight changes are
`6f5473ff`. Both are pulled on Tübingen with the qualified source hashes unchanged.
