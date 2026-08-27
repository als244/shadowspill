# Planning evaluation

This package consumes immutable `StepProgram` inputs and evaluates PressureFit,
simulation, and physical admission. It never constructs a model, captures a
graph, compiles a task, or profiles a kernel.

Run the complete 168-Program, 2,520-point baseline:

```bash
PYTHONUNBUFFERED=1 python -m benchmarking.planning_eval.evaluate \
  --config benchmarking/planning_eval/configs/full_pressurefit_frontier_v1.json \
  --corpus-dir benchmarking/datasets/input_programs/full_model_program_corpus_v1 \
  --output-dir benchmarking/planning_eval/results/full_pressurefit_frontier_v1 \
  --artifact-store benchmarking/program_collection/planning_caches/full_model_program_corpus_v1
```

The v1 matrix evaluates 15 budget/bandwidth points per Program. All 2,520
points use three global bidirectional-concurrent transfer pairs: 1/2x, 1x, and
2x the calibrated fetch/evict bandwidths. PressureFit is cold by default; the
saved planning cache is used only where the configuration explicitly permits
it.

Each point log contains:

- `[program/168]` and `[point/2520]` progress;
- model/provider identity;
- one grouped `DATA GEOMETRY` block (sequence length, tokens and sequences per
  microbatch, gradient accumulation rounds, and tokens per optimizer step);
- execution and spill budgets;
- fetch and evict bandwidths;
- UTC `START`, `STOP`, and `DURATION: <seconds>` records.

Blank lines separate points and Programs. Output is line-buffered to stdout and
duplicated in `collection.log`, so the same command is easy to follow in tmux.

Every point is journaled before PressureFit begins and atomically publishes one
of `succeeded`, `infeasible`, `search_exhausted`, or `error`. A worker exit or
300-second active-point timeout is attributed to that point; there is no point
retry. The controller advances to the next point/Program and preserves the
failure evidence. Use `--resume` only when intentionally continuing an existing
baseline.

Resume uses the same launch command plus `--resume`. It locates the incomplete
baseline from the config and corpus identities, validates every terminal point,
and starts at the first pending point.

The repository revision does not gate it. A run that was stopped days and many
commits ago resumes and finishes, because finishing it is the point; requiring
a matching revision only means replaying hours of planning to learn the same
thing. What does gate resume is what is being measured: the frontier config and
the corpus manifest must match, and a baseline whose either differs is not the
same baseline.

Instead of refusing, the baseline records what changed. Every point carries the
revision that produced it, so a mixed-revision run says so per point rather
than looking uniform. The resume record classifies the relationship as
`exact_source`, `harness_only`, `planner_changed`, `unrelated_revision`, or
`dirty_worktree`, and lists the files that differ. Resume commands and those
relationships are appended to `resume-commands.log` and `resume-history.jsonl`.

Read that before comparing a resumed baseline's wall times against another:
points from different revisions were produced by different code.

`--revision <sha>` names the revision a run records, and on `--resume` selects
the baseline recorded under it - which is how you pick one when several
baselines share a config and corpus. It labels the run; it does not check
anything out, so the code that runs is whatever is in the worktree. The corpus
collector takes the same option, and records the revision on every case.

If the controller was interrupted while a point was running, that attempt stays
in the journal with status `interrupted` but does not consume the point's attempt
budget. Timeouts, worker failures, and completed planner errors remain charged
and are never silently retried.

## Result layout

```text
<output>/<baseline-identity>/
├── config.json
├── corpus-manifest.json
├── provenance.json
├── collection.log
├── frontier.csv
├── frontier.jsonl
└── cases/<program>/
    ├── points/<point>/point.json
    ├── annotated-plans/<budgets>/<bandwidths>/<plan>/<artifact>/
    │   ├── manifest.json
    │   └── annotated_program_plan.json
    └── logs/worker-NNNN.log
```

Complete annotated plans include the source Program, selections, schedule,
simulator timeline, PressureFit diagnostics, admission refinements, and the
PressureFit/admission/orchestration wall-time split. Compact CSV/JSONL rows link
back to those canonical artifacts.

`plan_digest` excludes wall-clock/cache-hit observations and identifies a
semantic planner decision. `artifact_sha256` covers the full measured JSON,
including timing and diagnostics. A new planner revision creates a new baseline
identity and can be compared row-for-row with prior results.
