# Prepared commit groups

The user approved committing and pushing this batch on October 8, 2026, then
updating Della, fatnode, Tubingen, and Chicago. The groups below keep the
independently reviewed pieces separate. Full Qwen EP8 validation and the
small-model DP4 bias investigation remain outstanding.

1. **Planner cache retention:** restore every requested resolution on cache hits;
   reject incomplete retention through a completion manifest. Includes tests and
   artifact-store documentation.
2. **Verified symmetric planning:** opt-in public setting, requirement checks,
   conservative timing, one-owner CPU searches, receiver admission, retained
   schedules, generic quickstart controls, and regression tests/documentation.
3. **Qwen MoE workloads:** Qwen3-30B-A3B and Qwen3.5-35B-A3B text architectures,
   workload registration, and independent CPU numerical checks. Full EP8 GPU
   validation remains pending the Della allocation.
4. **Experiment record:** scripts, agenda, design notes, and small summary
   evidence. Excludes logs, checkpoints, build stores, source backups, and the
   unrelated `benchmarking/quickstart_reports` path.

Preview without changing the index or repository history:

```bash
python docs/internal/plans/qwen_moe_ep8_1007/scripts/commit_changes.py
```

After approval, `--execute` writes the selected groups on `master`; for example,
`--groups 1 2 --execute` commits only the planner changes. The script does not
push or create any branches/worktrees. MLOps has no changes in this batch.

## Known validation limits

- The separate DP4 toy-model bias discrepancy reproduces with symmetric planning
  disabled. It remains an open issue, with its evidence preserved.
- Real DP4 Llama training and a fresh independent-search control both passed
  finite-state, parameter-update, and within-run replica checks. This is not a
  claim that the unresolved toy case or Qwen EP8 training passed.
- See `SYMMETRIC_PLANNING.md`, `REAL_MODEL_DP4.md`, and `PROGRESS.md` for exact
  validation and the post-refactor rerun.
