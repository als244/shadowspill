# Planning evaluation

This harness **plans**. It reads a frozen corpus of `StepProgram` inputs
built by [program collection](../program_collection/README.md) and runs
graph-pair selection, PressureFit, simulation, and physical admission over a
grid of budgets and transfer bandwidths. It constructs no model, captures no
graph, compiles no task, and profiles no kernel, so it takes no build
arguments beyond `--corpus-dir`, the path the corpus lives at.

It needs no GPU. The task costs it plans against were measured when the
corpus was collected and are carried in each `StepProgram`, so a run measures
the planner on the machine it runs on against costs recorded on the machine
that collected the corpus, and those need not be the same machine. The
frontier config names no worker count, so the planner's default applies and
the search uses every logical CPU; wall time is dominated by selection, so a
host with more cores finishes sooner.

## Running it

```bash
PYTHONUNBUFFERED=1 python -m benchmarking.planning_eval.evaluate \
  --config benchmarking/planning_eval/configs/full_pressurefit_frontier_v1_repairs256.json \
  --corpus-dir benchmarking/datasets/input_programs/full_model_program_corpus_<rev> \
  --output-dir benchmarking/planning_eval/results/full_pressurefit_frontier_<rev>_repairs256 \
  --artifact-store benchmarking/planning_eval/planning_caches/frontier_<rev>
```

A corpus is named for the revision that collected it, and it holds whole
Programs, so it stops loading when the program schema moves on: point a run
at a corpus collected at or after the current schema, read from
`input_programs/` rather than from a name copied from here. Give the output
directory the revision being measured, so two baselines can be told apart
later.

`--config`, `--corpus-dir`, `--output-dir`, and `--artifact-store` are
required. The rest select subsets or describe the run:

| Argument | Meaning |
|---|---|
| `--resume` | Continue an existing baseline: validate every terminal point and start at the first pending one. Without it, a baseline directory that already holds case state is refused. |
| `--dry-run` | Print the expanded matrix, including the resume relationship, and exit. |
| `--case GLOB` | Evaluate only matching case IDs; repeatable. |
| `--start-at CASE_ID` | Start at one exact case ID after filtering. |
| `--limit N` | Keep the first N selected Programs. |
| `--verbose-search` | Forward each planning call's own phase reporting. |
| `--revision SHA` | The revision recorded on every point this run produces; on `--resume`, the baseline recorded under it. It labels the run and checks nothing out. |

`--artifact-store` is where a point's planning artifacts land: the selection
request, the result, the plan manifest, and an archived copy of the program
that was planned. What the run may do with what is already there is the
config's `plan_store_mode`.

## The configuration

One JSON file, validated strictly — an unknown or missing key is an error.

| Field | Meaning |
|---|---|
| `name` | Names the baseline, with the config digest and revision. |
| `expected_programs` | The corpus size this config is written for; a corpus of another size is refused. |
| `expected_points_per_program` | Must equal the expanded grid size, so a grid edit that changes the point count fails at load rather than mid-sweep. |
| `program_role` | Which program in each `StepProgram` is planned: `recurrent`, `initial`, or `forward`. |
| `point_timeout_seconds` | Required, with no default; the shipped configs set 300. |
| `max_point_attempts`, `max_worker_restarts_per_program` | How often a point may be retried, and how often its worker may be restarted. |
| `plan_store_mode` | `contribute`, `reuse`, `require`, or `refresh`. |
| `transfer_bandwidths` | One global fetch/evict pair, with its provenance, frozen across the corpus so points from different Programs are comparable. |
| `grids` | Cartesian products of execution budgets, spill budgets, and exact rational bandwidth scales. |

Four optional fields reach the planner and are part of the config digest, so
two runs that differ only in one of them are told apart:
`capacity_refinement_bytes`, `max_repair_attempts`, `split_write_backs`, and
`deterministic`. Absent means the planner's own default.

The v1 grids expand to 15 points per program: four execution budgets at one
spill budget across three bandwidth scales — half, one, and twice the frozen
calibration — plus three spill budgets at one execution budget and the
unscaled rates. The shipped configs set `plan_store_mode` to `refresh`, so
every point is searched afresh and its plan written over whatever the store
held: a baseline measures the planner, not the store.

## What a run prints

Each point logs `[program/N]` and `[point/M]` progress, the model and
provider, one grouped `DATA GEOMETRY` block (sequence length, tokens and
sequences per microbatch, gradient accumulation rounds, and tokens per
optimizer step), the execution and spill budgets, the fetch and evict
bandwidths, and UTC `START`, `STOP`, and `DURATION: <seconds>` records.
Blank lines separate points and Programs. Output is line-buffered to stdout
and duplicated in `collection.log`, so the same command is easy to follow in
tmux.

Every point is journaled before PressureFit begins and atomically publishes
one of `succeeded`, `infeasible`, `search_exhausted`, or `error`. A worker
exit or an active-point timeout is attributed to that point; the controller
then advances and preserves the failure evidence.

## What it writes

```text
<output>/<baseline-identity>/
├── config.json
├── manifest.json
├── summary.json
├── case-failures.json
├── collection.log
├── frontier.csv
├── frontier.jsonl
├── git-status.txt
├── launch-command.txt
├── planner.patch
└── cases/<program>/
    ├── points/<point>/point.json
    ├── annotated-plans/<budgets>/<bandwidths>/<plan>/<artifact>/
    │   ├── manifest.json
    │   └── annotated_program_plan.json
    └── logs/worker-NNNN.log
```

Complete annotated plans include the source program, selections, schedule,
simulator timeline, PressureFit diagnostics, admission refinements, and the
PressureFit/admission/orchestration wall-time split. Compact CSV/JSONL rows
link back to those canonical artifacts.

`plan_digest` excludes wall-clock and store-hit observations and identifies a
semantic planner decision. `artifact_sha256` covers the full measured JSON,
including timing and diagnostics. A new planner revision creates a new
baseline identity and can be compared row-for-row with prior results.

## Resuming

Resume uses the same launch command plus `--resume`. It locates the
incomplete baseline from the config and corpus identities and starts at the
first pending point.

The repository revision does not gate it. A run that was stopped days and
many commits ago resumes and finishes, because finishing it is the point;
requiring a matching revision only means replaying hours of planning to learn
the same thing. What does gate resume is what is being measured: the frontier
config and the corpus manifest must match, and a baseline whose either
differs is not the same baseline.

Instead of refusing, the baseline records what changed. Every point carries
the revision that produced it, so a mixed-revision run says so per point
rather than looking uniform. The resume record classifies the relationship as
`exact_source`, `harness_only`, `planner_changed`, `unrelated_revision`, or
`dirty_worktree`, and lists the files that differ. Resume commands and those
relationships are appended to `resume-commands.log` and
`resume-history.jsonl`. Read that before comparing a resumed baseline's wall
times against another: points from different revisions were produced by
different code.

`--revision <sha>` is also how one baseline is picked when several share a
config and corpus.

If the controller was interrupted while a point was running, that attempt
stays in the journal with status `interrupted` but does not consume the
point's attempt budget. Timeouts, worker failures, and completed planner
errors remain charged and are never silently retried.
