# Phase quickstart

Quickstart composes the public library APIs in separate processes:

```text
build_step → Program
measure_program → Profiles → apply_profiles → ProfiledProgram
measure_communication → CommunicationInfo
plan_program over geometries/orderings/budgets → Plans + simulated reports
Trainer(Plan) → execution measurements
```

Build and profiling use ordinary PyTorch allocation. Planning is CPU-only.
Each execution budget gets a fresh worker with Executor's allocator installed.
Compiler caches stay outside the result directory. No stage implicitly invokes
the previous one to fill missing artifacts.

The migrated path supports single-GPU training and single-node sharded DP,
including examples with different shapes and static arguments within one update.
The [catalog factory](../workloads/recipes/text/benchmark/README.md) supplies
ordinary text models, precision and LoRA settings. The GLM families use this
same runner with their own model/input/initialization helpers. Model-owned EP
resources still use the old `python -m benchmarking.quickstart.cli` path until
migrated; its [migration reference](quickstart_legacy.md) is separate.
The core APIs can be used directly without quickstart; see the
[public phase guide](../docs/architecture/offline-pipeline.md).

## Complete example

```bash
python -u -m benchmarking.quickstart \
  --factory workloads.recipes.regression.benchmark:experiment \
  --factory-args '{"rows":64,"width":16,"outputs":8}' \
  --search-budget-gib 2 3 --run-budget-gib 2 3 --spill-gib 2 \
  --steps 5 --device cuda:0 --output-dir runs/regression
```

This builds two example geometries, profiles every unique task form, measures
host transfer rates, plans both budgets, writes simulated timelines and runs the
selected plans. Budgets mean total GPU allowances in GiB. `--run-budget-gib`
with no values skips execution; omitted values run every requested search budget.
Infeasible cases stay in the planning report and are skipped for execution.

A Python caller can use `benchmarking.quickstart.run` with the same argument
names, using underscores and numeric lists instead of CLI syntax. The factory
must be importable (a module function or `module:function` string), because
workers reconstruct setup independently. Closures returned *by* that factory
are fine; no callback pickle or model object is sent between workers.

## One phase at a time

Start with build alone, without selecting planning budgets yet:

```bash
python -m benchmarking.quickstart --phase build \
  --factory workloads.recipes.regression.benchmark:experiment --output-dir runs/regression
python -m benchmarking.quickstart --phase profile --output-dir runs/regression
python -m benchmarking.quickstart --phase communicate --output-dir runs/regression
CUDA_VISIBLE_DEVICES='' python -m benchmarking.quickstart --phase plan \
  --output-dir runs/regression --search-budget-gib 2 3 --spill-gib 2
python -m benchmarking.quickstart --phase execute \
  --output-dir runs/regression --run-budget-gib 2 3
```

Once profiled, repeat only `--phase plan` to change budgets or communication costs.
It reads the saved ProfiledPrograms and does not import a model, Torch, runtime
pools, or the factory. The process orchestrator also explicitly hides CUDA for
this worker. `--no-plots` keeps numeric reports/HTML while skipping figure files.

`--phase profile` loads saved Programs without the model factory. Use
`--warmup-seconds` and `--measure-seconds` to control measurement; defaults are
0.05 and 0.18 seconds. `--phase communicate` calibrates host copies and the union
of collective probes required by the candidates. This calibration can be replaced with
`--communication-info /path/to/communication.ssp`, saved by
`CommunicationInfo.save`. The same option overrides communication inputs during
CPU replanning. Supplied task durations or memory estimates should be applied
through `Profiles` / `apply_profiles` and saved to the corresponding `profiled/`
archive; estimated memory produces simulation-only plans and cannot execute.

## Factory inputs

The [regression factory](../workloads/recipes/regression/benchmark.py) is a complete example.
It returns a plain mapping:

| Entry | Purpose |
| --- | --- |
| `model_factory` | Construct the ordinary model, preferably on `meta`; called only by build. |
| `optimizer` | Callable taking model parameters and returning the optimizer used by `build_step`. |
| `objective` | Called as `objective(model, *args, **kwargs)`. It owns loss normalization. |
| `candidates` | Names mapped to `{"args": [...], "kwargs": [...]}`, the same per-example input sequences as Executor. Inputs stay on CPU. |
| `build_options` | Optional direct builder controls such as `partition`, `operator_recompute`, `master_dtype`, `grad_dtype`. |
| `fill_tensor` | Execution callback `(TensorChunk, destination)` filling the bounded CPU buffer supplied by Executor. Checkpoint slices or deterministic random initialization belong here. |
| `hyperparams` | Optional per-call runtime values using the names declared in the Program. |
| `metric_reducer` | Optional reducer of host observations to named scalars or MetricSummary. |
| `context` | Optional context-manager factory around build and execution initialization, for scoped compiler settings and closing readers. Saved tasks do not require it for profiling or execution. |
| `units_per_step`, `unit_label` | Optional throughput numerator and label; quickstart does not assume tokens. |
| `metadata` | Optional JSON description retained with the candidate inventory. |

Each candidate supplies every example call, including its shapes and static
arguments, through the builder's `example_calls` interface. Identical task forms
still share compiled code and profiles; differing examples retain their own
contracts. All candidates should describe the same normalized update. Example
ordering and optimizer ordering are selected during planning. Both save/recompute variants
remain available. Runtime hyperparameters must be declared during build—for
example, construct the optimizer with a tensor-valued learning rate to change it
at execution time. There is no automatic alias from `lr` to an optimizer group.

The factory runs separately for build and execution. It should return a lazy
model factory, or declare only meta tensors when the initializer needs their
geometry. It must not allocate a complete model's payload in its own body. Execution
never calls `model_factory` or `optimizer`. Initializers must fill supplied chunks;
quickstart does not create an anonymous full-model host copy. Each budget starts
from the same contents when the initializer is deterministic.

## Distributed workers

The factory can declare `Parallelism` through `build_options`. Build and task
profiling still launch one representative GPU process for ordinary DP.
Communication calibration and execution launch the Program's worker count
using `torchrun`; they create actual process groups only there. The launcher
supports one node and symmetric DP tasks. Model-owned distributed operators
still need their own resource binding and qualification.

Distributed factories accept `logical_rank=0`. Execution passes each worker's
logical rank so inputs can differ while state initialization stays identical for
matching parameter slices. Objectives and displayed losses must be normalized
contributions to the global update; quickstart sums them across ranks. Set
`units_per_step` to the global count. Reported step time is the maximum worker
time, and the result retains every rank's measurement.

Use `device="cuda:0"` for distributed builds. Workers expose their assigned GPU
as `cuda:0` so all use the saved code's device ordinal. Select physical devices
through `CUDA_VISIBLE_DEVICES`. The default logical rank mapping is identity;
`rank_map={0: 1, 1: 0}` in Python, or `--rank-map '{"0":1,"1":0}'`, supplies a
complete alternative. Executor inherits the Plan's recorded overhead reservation,
including calibrated communication memory, and retains physical admission checks.

## Results

```text
request.json                 # reproducible orchestration settings
programs/                    # one Program archive per geometry + small index
profiles/                    # reusable measured task records
profiled/                    # independent Program + Profiles archives
communication.ssp            # measured or supplied transfer information
planning/sweep.ssp           # all cases, plans and failures
plans/                       # one selected portable Plan per feasible budget
report/index.html            # simulated timelines, memory and transfer tables
logs/                        # separate phase and execution worker logs
execution/budget-*/run-*/     # host metrics for each execution attempt
                             # distributed attempts contain rank-0000/, ...
execution/budget-*/result.json
execution/summary.json       # latest matching results and predicted/actual times
execution/summary.csv
execution/performance.svg    # measured throughput and simulated/measured times
execution/performance.png
```

Execution summaries record every step plus the median duration and optional
throughput; the first step can include startup overhead and remains visible.
These small regression runs qualify the pipeline, not model performance.
Rerunning execution creates another attempt directory. Plan identities prevent
old measurements from being reported as belonging to a newly selected plan.
With `--plots`, the execution comparison uses matching measured runs only.
`--no-plots` keeps JSON/CSV summaries and removes older comparison figures.
Per-step `loss` honors the metric reducer's display override; `objective_loss`
always retains the differentiated objective. Detailed reducer metrics live in
each attempt's `metrics.jsonl`, copied to host before the reducer is called.
The numerical report exporters for the old quickstart format remain under the
private migration path until updated.
