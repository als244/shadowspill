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
| 131,072 | 4 | Running | Pending |
| 262,144 | 2 | Queued in resumable retry | Pending |

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

Current job 14864915 ends at 12:07:27 UTC. Follow-up 14868213 requests two GPUs,
240 GiB host RAM, 32 CPUs and three hours, with `afterany:14864915` to avoid
overlapping allocations. The allocation watcher polls every ten seconds and
wakes the agent; it launches no workload. GPU work stays in `codex:0.0`.

The W&B relay and authenticated API query succeeded from della-j15g1 on
October 2. Reconnect and verify the reverse SSH relay on any replacement node.
The training recipe records separate per-rank and aggregate runs in project
`shadowspill-della-ep-training`; planning-only trials create no training runs.
The LR horizon remains 9,537 updates, independent of the allocated run length.
