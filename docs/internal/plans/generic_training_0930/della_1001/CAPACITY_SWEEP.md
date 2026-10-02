# Chicago EP2 capacity sweep after the fixes

The model has 16 layers, width 1024, 192 routed experts, top-4 routing,
expert width 1280, vocabulary 50304 and sequence length 2048. The global
update contains 1,048,576 token slots. Each candidate retains the same 50 GiB
execution budget, 6 GiB external headroom and 80 GiB host spill per rank.
One workload-owned MoonEP buffer and two projection banks are shared by all
layers. Source revisions: ShadowSpill `17c0a2d1`, MLOps `e61e7b2`.

| Tokens per rank per microbatch | Microbatches per rank per step | Status | Predicted step, slower rank |
| ---: | ---: | --- | ---: |
| 16,384 | 32 | Prior 100-step run complete | 8.464 s |
| 32,768 | 16 | Plan and physical admission pass on both ranks | 6.674 s |
| 65,536 | 8 | Plan and physical admission pass on both ranks | 6.573 s |
| 131,072 | 4 | Plan and physical admission pass; selected for training | 6.398 s |
| 262,144 | 2 | Plan and physical admission pass on both ranks | 6.878 s |

The 16K measured median was 7.785 s/update. The other rows are planning
results until explicitly marked as measured training. New 32K preparation took
865 seconds including model initialization, two ordering searches and final
plan construction. The final cached construction measured 132 cache hits,
zero misses and zero retained saved-value snapshots.

## Results and resumption

All new evidence is under
`/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-ep2-capacity-fixes-1002`.
Each `tokens-N/` has `result.json`, the worker console and process logs,
and per-rank configurations, communication memory, selected plan, search
results, diagnostics and the complete build artifact store. Readable
`graph-pairs.csv` and `graph-pairs.md` sit beside each rank's diagnostics.
Planning progress is written after each case. The retry command is:

```bash
bash docs/internal/plans/generic_training_0930/della_1001/scripts/retry_chicago_capacities.sh
```

Passed cases are skipped; interrupted or failed cases are retried. The updated command was restarted at 11:41 UTC after the original 32K/64K
parent finished; it skipped those two cases and started 128K. The original
shell wrapper printed an EOF parsing error after both workers completed
successfully: editing the shell file while its pipeline was running changed
its read offset. The updated script passed `bash -n` and launched normally.
Do not edit a shell wrapper while it is running.

## Reservation and logging

Job 14864915 completed the capacity sweep. Replacement 14868213 is active with
two GPUs, 240 GiB host RAM, 32 CPUs and three hours through 15:09:42 UTC. Its
`afterany:14864915` dependency avoided overlapping allocations. The allocation watcher polls every ten seconds and
wakes the agent; it launches no workload. GPU work stays in `codex:0.0`.

The W&B relay and authenticated API query succeeded from della-j15g1 on
October 2. Reconnect and verify the reverse SSH relay on any replacement node.
The training recipe records separate per-rank and aggregate runs in project
`shadowspill-della-ep-training`; planning-only trials create no training runs.
The LR horizon remains 9,537 updates, independent of the allocated run length.

## Training under the replacement allocation

Allocation 14868213 started at 12:09:42 UTC on della-j15g1, ending at 15:09:42
UTC. The watcher reported RUNNING at 12:09:45. Its shell was moved into
`codex:0.0`; the W&B tunnel was reconnected and authenticated before launch.

The fastest admitted case is 128K tokens/rank, four microbatches per rank.
Training uses the separately copied 1,999,999,310-token GPT-2 prefix, which
matches the full original planning sample byte for byte. Data stays under
`~/storage/datasets/fineweb_edu_gpt2_chicago_2b`.

The normal zero-LR warmup and traced step ran once. The slower-rank trace was
7.327 s, which selected 1,100 training updates with 15% timing headroom and a
ten-minute final checkpoint/sync reserve. LR still follows the original
9,537-update schedule; evaluation is every 100 and checkpointing every 500
plus the final update. This is an estimated run length, not a hard time stop.

The first attempt completed 100 updates: median 6.412 s/update, aggregate loss
11.157→6.234. Its first evaluation then failed during physical admission.
The callback capture had hidden registered module paths, collapsing evaluation
into one whole-model task. See [EVALUATION_CALLBACK.md](EVALUATION_CALLBACK.md).
The following links belong to that failed attempt, whose complete evidence
is preserved in `tokens-131072/`:

- Aggregate W&B: https://wandb.ai/andrew-sheinberg-princeton-university/shadowspill-della-ep-training/runs/8uqx2s62
- Rank 0 W&B: https://wandb.ai/andrew-sheinberg-princeton-university/shadowspill-della-ep-training/runs/vayee76b
- Rank 1 W&B: https://wandb.ai/andrew-sheinberg-princeton-university/shadowspill-della-ep-training/runs/sspu1hic

The generic callback provenance correction passes 97 CPU checks and the full
EP2 evaluation preflight. The retry uses `training-128k-v2/` and is training
900 updates with online W&B. At update 11, the measured median was 6.425 s.
[Live aggregate run](https://wandb.ai/andrew-sheinberg-princeton-university/shadowspill-della-ep-training/runs/26c6iq9k).
Its per-rank `training-window.json` records the common run length and unchanged
9,537-update LR horizon. The preflight and startup trace occur before W&B
logging; they are not training updates. The head-node
watcher polls every ten seconds for errors, milestones and completion.
Tübingen passed all three capacity-fix gates and the additional callback-fix
suite (1,110 passed, one skip, 48 CUDA CTests) and numerical gate (5/5). Allocation-length control lives only in the
experiment client; no optimizer behavior changed.
