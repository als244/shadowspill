# Phase quickstart

Quickstart composes the public library APIs in separate processes:

```text
build_step → Program
measure_program → Profiles → apply_profiles → ProfiledProgram
measure_communication → CommunicationInfo
measure_overlap → Profiles overlay → enriched ProfiledProgram
plan_program over build policies/geometries/orderings/budgets → Plans + simulated reports
Executor(Plan) → execution measurements
```

Build and profiling use ordinary PyTorch allocation. Planning is CPU-only.
Each execution budget gets a fresh worker with Executor's allocator installed.
Compiler caches stay outside the result directory. No stage implicitly invokes
the previous one to fill missing artifacts.

Quickstart supports single-GPU training and single-node sharded DP, including
examples with different shapes and static arguments within one update.
The [catalog factory](factories/text/README.md) supplies
ordinary text models, precision and LoRA settings. The GLM families use this
same runner with their own model/input/initialization helpers. The
[EP recipe](factories/expert_parallel/README.md) declares generic
provider resources and uses distributed build/profiling when its tasks communicate.
Ordinary DP qualification does not certify its transport.
The core APIs can be used directly without quickstart; see the
[public phase guide](../docs/python/phases.md).

## Complete example

```python
from benchmarks.quickstart import run

run(
    # An importable factory supplies the model, objective, data and candidates.
    factory="benchmarks.factories.regression:experiment",
    factory_args={"rows": 64, "width": 16, "outputs": 8},

    # Search each device budget with the same measured candidates and host pool.
    search_budget_gib=[2, 3],
    spill_gib=2,

    # Execute every admitted winner for five updates and keep the evidence.
    steps=5,
    output_dir="runs/regression",
)
```

This builds two example geometries, profiles every unique task form, measures
host transfer rates, plans both budgets, writes simulated timelines and runs the
selected plans. Budgets mean total GPU allowances in GiB. `--run-budget-gib`
with no values skips execution; omitted values run every requested search budget.
Infeasible cases stay in the planning report and are skipped for execution.

`benchmarks.quickstart.run` accepts ordinary Python values, using underscores
in argument names and numeric lists for budgets. The equivalent CLI is
`python -m benchmarks.quickstart`. The factory
must be importable (a module function or `module:function` string), because
workers reconstruct setup independently. Closures returned *by* that factory
are fine; no callback pickle or model object is sent between workers.

For full-checkpoint text training with evaluation and checkpointing, start with
the [public-client demos](../examples/demos/README.md)
and the [model catalog](../docs/python/catalog.md).

## Spill storage

Pinned host memory is the default. Select SSD with `--ssd-spill /local-nvme`,
or an RDMA peer with `--remote-spill HOST:PORT`. The corresponding Python argument
is `spill=SSDPool(...)` or `spill=RemotePool(...)`, with capacity equal to
`spill_gib`. SSD tuning flags are `--ssd-staging-mib`, `--ssd-chunk-mib` and
`--ssd-queue-depth`; defaults are 256 MiB, 2 MiB and 16.

Only communication calibration and execution open the selected backend.
Build/profile use ordinary allocation; CPU planning reads CommunicationInfo
without importing SSD/network code or contacting the peer. Calibration measures
the complete route, including staging, in a bounded scratch pool. Executor uses
bounded buffers for initialization, controls and checkpoint I/O on every backend.
To change backend, rerun communication calibration and planning before execution.

Single-GPU SSD and RDMA execution are qualified. Compound sharded transfers
use the same spill backend interface; that qualification does not cover them on
several GPUs. Saved Executor plans use bounded tensor loaders.
See [SSD storage](../docs/python/api/ssd.md) for configuration and lifetime details.

## One phase at a time

Start with build alone, without selecting planning budgets yet:

```bash
python -m benchmarks.quickstart --phase build \
  --factory benchmarks.factories.regression:experiment --output-dir runs/regression
python -m benchmarks.quickstart --phase profile --output-dir runs/regression
python -m benchmarks.quickstart --phase communicate --output-dir runs/regression
CUDA_VISIBLE_DEVICES='' python -m benchmarks.quickstart --phase plan \
  --output-dir runs/regression --search-budget-gib 2 3 --spill-gib 2
python -m benchmarks.quickstart --phase execute \
  --output-dir runs/regression --run-budget-gib 2 3
```

Once profiled, repeat only `--phase plan` to change budgets or communication costs.
It reads the saved ProfiledPrograms and does not import a model, Torch, runtime
pools, or the factory. The process orchestrator also explicitly hides CUDA for
this worker. `--no-plots` keeps numeric reports/HTML while skipping figure files.

The Python `run` accepts `tail_save_counts=[0, 1, 2, 4, 8, 16]` and
`resolution_options=["0", "1/4", "1/2", "3/4", "1"]`. Each tail count crosses
every resolution for each measured candidate and ordering. A count saves the
last K eligible forward invocations within each forward/backward pass, repeated
at every depth. The resolution's recompute fraction applies to the remaining
flexible invocations. Required saves do not consume K; zero retains the original
distribution. Defaults are `(0,)` and quarter shares. These choices reuse the
same build/profile artifacts; they only expand CPU planning. Saved values can
still spill according to PressureFit, and every selected schedule must pass
physical admission. See the [complete contract](../docs/architecture/graph-pair-selection.md).

For a saved quickstart run, set `--tail-save-counts 0 1 2 4 8 16` and
`--resolution-options 0,1/4,1/2,3/4,1` during `--phase plan`. The choices are
recorded in `request.json` and the Plan's search key. Execution uses that Plan;
changing these choices requires replanning.

`--phase profile` loads saved Programs without the model factory. Use
`--warmup-seconds` and `--measure-seconds` to control measurement; defaults are
0.05 and 0.18 seconds. `--phase communicate` calibrates host copies and the union
of collective probes required by the candidates. For sharded DP, it then calls
`measure_overlap` for each successfully profiled candidate, using its saved
tasks and actual collective routes. Repeated task-only, transfer-only and paired
trials produce computation/communication progress rates for the simulator.
The phase saves `overlap/` overlays and merges them into `profiles/` and
`profiled/`, preserving isolated timing and memory evidence. It does not rebuild
kernels or fit costs to a complete training step. Unsupported or unstable
measurements remain missing; see the
[measurement contract](../docs/python/api/artifacts.md#profiles-and-taskprofile).

The measurement phase can be replaced with
`--communication-info /path/to/communication.ssp`, saved by
`CommunicationInfo.save`. That path copies supplied communication information
without measuring paired interference; reuse compatible enriched profiles or
measure and merge an overlay explicitly. The same option overrides communication
inputs during CPU replanning. Supplied task durations or memory estimates should be applied
through `Profiles` / `apply_profiles` and saved to the corresponding `profiled/`
archive; estimated memory produces simulation-only plans and cannot execute.
For a manual profile pass, copy `programs/index.json` to `profiled/index.json`
after saving every successful candidate's ProfiledProgram; the normal profile
phase writes this index automatically.

Typed device out-of-memory failures during build or profiling are recorded for
that candidate; the sweep continues with the others. The saved sweep and HTML/CSV
report retain each failure at every requested budget and ordering. Other errors
stop the phase. Communicating build/profile workers stop together on any failure;
a single rank cannot safely skip a task while its peers remain inside it.
A profile rerun retries built candidates, and an interrupted
build/profile pass cannot leave its old completion index eligible for planning.

## Factory inputs

The [regression factory](factories/regression.py) is a complete example.
It returns a plain mapping:

| Entry | Purpose |
| --- | --- |
| `model_factory` | Construct the ordinary model, preferably on `meta`; called only by build. |
| `optimizer` | Callable taking model parameters and returning the optimizer used by `build_step`. |
| `objective` | Called as `objective(model, *args, **kwargs)`. It owns loss normalization. |
| `candidates` | Names mapped to `{"args": [...], "kwargs": [...], "build_options": {...}}`. `args`/`kwargs` are the same per-example input sequences as Executor; inputs stay on CPU. Candidate `build_options` are optional overrides of the factory defaults. |
| `build_options` | Optional shared direct builder controls such as `partition`, `chunking`, `graph_pair_policy`, `gradient_destination_rules`, `master_dtype`, `grad_dtype`. |
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
ordering and optimizer ordering are selected during planning. Save/recompute variants
permitted by each candidate's policies remain available. Runtime hyperparameters must be declared during build—for
example, construct the optimizer with a tensor-valued learning rate to change it
at execution time. There is no automatic alias from `lr` to an optimizer group.

Microbatch size is part of each candidate's input shapes; the number of calls
is its microbatch count. Declare only the geometries you want built. The ordinary
Python runner tries every depth/breadth factorization of that count during CPU
planning, with paired loss and reverse backward order. An explicit ordering
allowlist can instead be implemented by looping over
`plan_program(ordering=StepDataOrdering(...))`; quickstart's Python `run` does
not expose an ordering selector, and candidate fields do not accept one.
Reordering reuses the built tasks and profiles. The selected Plan fixes the
input geometry and traversal that Executor executes.
See [step ordering](../docs/architecture/step.md).

The launcher reads the factory's declarations before build to select its worker
count; build and execution also reconstruct the setup. It should return a lazy
model factory, or declare only meta tensors when the initializer needs their
geometry. It must not allocate a complete model's payload in its own body. Execution
never calls `model_factory` or `optimizer`. Initializers must fill supplied chunks;
quickstart does not create an anonymous full-model host copy. Each budget starts
from the same contents when the initializer is deterministic.

### Searching chunk and retention policies

Declare a named candidate for each combination of input geometry, chunk policy
and graph-pair policy to measure. For example, given `geometries`
containing the `args`/`kwargs` sequences, `chunk_policies` containing generic
[axis-chunking policies](../docs/architecture/axis-chunking.md), and
`graph_pair_policies` containing the builder's
[graph-pair policies](../docs/python/api/frontend.md#graph-pair-policies):

```python
candidates = {
    f"{geometry}-{configuration}-{retention}": {
        **inputs,
        "build_options": {
            "chunking": policy,
            "graph_pair_policy": graph_pair_policy,
        },
    }
    for geometry, inputs in geometries.items()
    for configuration, policy in chunk_policies.items()
    for retention, graph_pair_policy in graph_pair_policies.items()
}
```

A policy can select different chunk sizes for different execution regions,
including bounded compositions. Candidate overrides replace individual
factory-level options; they are not recursively merged. An explicit
`{"chunking": None}` measures the unchunked builder behavior. Keep the objective,
initial state and numerical policy fixed when comparing throughput.
`parallelism` and `resources` must be declared at factory level: candidate
overrides for those controls are refused before selecting workers.

Provider accumulation contracts can be shared across these candidates through
the factory's `build_options["gradient_destination_rules"]`. For example, MLOps
versions with the optional ShadowSpill integration expose
`shadowspill.integrations.mlops.sonicmoe_gradient_destination_rules()` for
SonicMoE LoRA. Include its separate `sonicmoe_chunk_rules()` alongside the other
operator rules in each applicable chunk policy. The rules do not pick a chunk
size or retention policy; every candidate still passes ordinary compilation,
profiling and admission. See [gradient destination writes](../docs/integrations/mlops.md#gradient-destination-writes)
for precision permissions and unsupported paths that retain ordinary accumulation.

Each graph-pair policy defines independent `save` and `recompute` cuts; the
planner chooses among the variants each chunk policy permits. Retained values
remain eligible for ordinary residency and spill decisions. Candidate names
should identify both chunk and graph-pair policies. There is no need to enumerate
every stage's save/recompute choice as a separate candidate.

For dense-attention retention, declare three independent policy pairs. Given
the provider's exact `attention_operators` overload names:

```python
from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule

cost = RetentionPolicy()
inputs_only = RetentionPolicy(default="recompute")
attention_only = RetentionPolicy(
    default="recompute",
    rules=tuple(RetentionRule(op, "retain") for op in attention_operators),
)
graph_pair_policies = {
    "cost_inputs": GraphPairPolicy(save=cost, recompute=inputs_only),
    "cost_attention": GraphPairPolicy(save=cost, recompute=attention_only),
    "attention_inputs": GraphPairPolicy(save=attention_only, recompute=inputs_only),
}
```

Cross all three with the chunk configurations in the comprehension above.
Each pair also retains supported routing choices by default, including the
`inputs_only` policy above. The name describes its base cut; top-k IDs and
selected routing weights remain ordinary saved tensors eligible for spill.
Each pair gets its own chunk-size search; the winner is selected over the full
set at each budget. Do not first choose a retention pair at one chunk size and
reuse that choice for the others. The best combination can change with sequence
length, hardware, communication costs and memory budget. Changed task geometry
or hardware requires compatible build/profile evidence; changed transfer costs
can be supplied for offline replanning.

The runner measures the declared candidates and does not invent provider
targets or expand per-stage combinations. These are factory declarations for
normal training search, not a fixed policy chosen at execution. See the
[graph-pair policy contract](../docs/python/api/frontend.md#graph-pair-policies)
for mandatory state and terminal-SAVE behavior. A factory may deliberately use
a smaller explicit candidate set when preparation latency matters.

Build finalizes each candidate and profile measures its task implementations.
Planning searches every saved candidate and ordering at each budget, including
its permitted save/recompute and offload choices. The admitted Plan with the
lowest simulated update time wins. Candidate names identify the configuration
in the full sweep/report, the winning Plan index and execution results; archives
retain the resolved task contracts. Build/profile OOMs and planning refusals
remain in the report. No candidate is inferred from a smaller task's workspace
or an isolated kernel's throughput.

Execution loads the winner's exact saved Plan; it does not resize chunks or
forward build controls as model inputs. Replanning different budgets reuses
the measured candidates without rebuilding. To try a new policy, build and
profile it in a fresh run before comparing its Plans.

## Distributed workers

The factory can declare `Parallelism` through `build_options`. Build and task
profiling still launch one representative GPU process for ordinary DP.
Communication calibration and execution launch the Program's worker count using
`torchrun`. If `build_options.resources` names participant groups, build and
profiling also use that worker count and recreate the described resources.
Only launcher rank zero writes the shared archives. Ordinary DP still needs no
process group for build/profiling. The launcher supports one node and symmetric
tasks; each model-owned distributed backend requires its own qualification.

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
profiled/                    # independent Program + Profiles archives + outcome index
communication.ssp            # measured or supplied transfer information
overlap/                     # paired task/collective measurements and coverage
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

## Nsight captures of saved winners

After a budget has executed successfully, capture its exact saved Plan:

```bash
python -m tools.pressurefit_trace.profile_winners runs/regression \
  --output runs/regression-nsys --budget-gib 2 \
  --warmup-steps 3 --capture-steps 3
```

Omit `--budget-gib` to capture every completed winner. The saved Program determines
the worker count and the request supplies the complete rank map. Each fresh worker
loads the same Plan and initializes through the original bounded loader. Capture
does not build, profile, calibrate or replan, and starts from initialized values
rather than continuing the benchmark's final weights.

Each budget gets separate attempt logs, an `.nsys-rep`, a SQLite export and checked
plan identities. Validation requires GPU kernels, device metrics and semantic
task/transfer/step ranges from every worker. Existing validated captures are reused
only when their inputs and output hashes match. These captures explain execution;
use the ordinary quickstart measurements for throughput. Device metrics require
permission on the selected GPUs.

## Optional pool-memory calibration

Pass `--pool-memory pool-memory.ssp` (or `pool_memory=profile` to `run`) to reuse
one full-capacity pool measurement across all policy/chunk/order/budget candidates.
The Plan embeds this evidence for cold startup. Measurement is separate from
transfer timing and never repeats per candidate. See the
[pool-memory contract](../docs/python/api/pool-memory.md) for measurement,
distributed composition and the distinction between process and physical limits.
