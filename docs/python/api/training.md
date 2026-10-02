# Generic training and forward execution

`shadowspill.training` accepts ordinary PyTorch modules, functions and iterables.
It has no model catalog or text-input convention. Optional recipes under
`workloads/recipes/` are clients of this API. The lower-level
[planning API](frontend.md) remains available for direct runtime ownership and
custom execution loops.

## Trainer

`Trainer` prepares one objective and one optimizer update. One item from the
data source represents one complete update. `trainer.step(data)` executes it;
`trainer.fit(source, steps=N)` supplies an optional loop. A user does not need
to subclass a model, implement a data-source interface, or register an objective.

```python
import torch
from shadowspill.training import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import WarmupCosine

with ShadowSpill(execution_gib=20, spill_gib=64) as backend:
    with Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.AdamW,
        optimizer_args={"weight_decay": 0.01},
        schedules={"lr": WarmupCosine(3e-4, 3e-5, 100, 10000)},
        backend=backend,
    ) as trainer:
        trainer.prepare(example_data)
        trainer.fit(
            source, steps=10000, run_dir="runs/example",
            log_every=10, eval_data=validation, eval_every=100,
            eval_batches=20, checkpoint_every=500,
            checkpoint_dir="/storage/checkpoints/example",
        )
```

`model`, `objective`, `example_data`, `source` and `validation` are caller inputs.
The [complete regression example](../../examples/generic-training.md) shows how
to construct them without an external model library.

| Constructor argument | Meaning |
| --- | --- |
| `model` | Ordinary initialized module, or a meta module with explicit initialization at preparation |
| `objective` | `objective(model, data)` returning a scalar loss or `(loss, metrics)`; metrics are a supported tensor pytree |
| `optimizer` | Optimizer constructor or callable returning an optimizer from parameters |
| `optimizer_args` | Constant constructor keyword arguments; defaults to an empty mapping |
| `schedules` | Mapping from hyperparameter name to `fn(update_index)`; default empty |
| `hyperparams` | Additional dynamic names accepted by `step`; schedule keys are declared automatically |
| `parameter_groups` | Optional `fn(model)` returning ordinary optimizer group dictionaries after state import |
| `microbatches` | Optional `fn(data)` yielding `(microbatch_data, loss_scale)`, or named candidate functions |
| `backend` | `PyTorch()` by default, or an explicitly entered `ShadowSpill(...)` context |
| `master_dtype`, `grad_dtype` | Optional optimizer master and gradient accumulation dtypes; neither changes model construction |
| `parameter_metrics` | Optional pure tensor function `(parameter, gradient) -> metrics`, before optimizer updates |
| `metric_reducer` | Optional host function of `StepObservations`, returning a scalar mapping or `MetricSummary` |
| `eval_fn` | Evaluation objective; defaults to the training objective |
| `distributed` | Optional explicit participant/parameter-replica ownership; see the [distributed guide](distributed.md) |
| `shard_optimizer` | `True` by default: shard optimizer state and optional masters over parameter replicas when distributed |

A live optimizer instance belongs to the low-level `plan_step` API. The trainer
constructs its optimizer after state import, preserving group assignments and
scheduled tuple values. Unscheduled group-specific constants remain unchanged.

## Preparation and initialization

`trainer.prepare(example_data, initialize=None, checkpoint=None)` performs
initialization, capture, profiling, planning and admission as needed by the
backend. It returns the trainer and does not increment its update count.
Representative input shapes and structures must match subsequent calls.

Already-initialized state is retained. A meta model requires `initialize=fn` or
a checkpoint. `reset_parameters` is an explicit supplied initializer; no reset
occurs merely because a runner is prepared. Meta materialization preserves tied
parameter identities and shared-storage views. Mixing meta and initialized
state is rejected. An initializer can build nonpersistent buffers before a
checkpoint is applied; the checkpoint then supplies the saved state.

Enter `ShadowSpill` before allocating accelerator state or constructing model
resources on the device. CPU models may be constructed earlier. The backend
owns the imported model state; use `trainer.model` after preparation. Close
runners before the backend. The contexts in the example enforce that order.

Fresh meta initialization currently materializes ordinary CPU storage before
importing state into the pinned spill pool. Budget for that staging copy;
`prepare(initialize=...)` does not initialize directly into the pool or stream a
large model in bounded chunks.

## Data sources

Pass an ordinary iterable or PyTorch `DataLoader`. Each yielded item is the data
for one update in this process; `microbatches` splits that item for execution.
Preparation uses a caller-supplied example and does not advance the source. When
taking that example from an iterator, include it again for the first update.

Distributed callers supply rank-local items and their normalization. ShadowSpill
does not insert a sampler or shard the source. See
[distributed data loading](distributed.md#data-loading) for partition choices.
Optional `state_dict()`/`load_state_dict()` methods preserve source progress in
checkpoints; an ordinary DataLoader alone does not guarantee exact resume.

## Microbatches and loss weighting

The default is one microbatch with scale `1`. With a callback, the runner sums
microbatch gradients after multiplying each scalar loss by its supplied scale.
There is no implicit mean over microbatches, data items, tokens, or ranks.

For a mean squared error over a whole update with 24 target elements, each
microbatch can return its **summed** squared error and scale `1 / 24`. Unequal
microbatch sizes still have the correct weight. An already-normalized objective
uses scale `1`. A token recipe instead divides by the number of valid targets.
Other objectives may normalize by observations or arbitrary caller weights.

Scales are finite Python numbers or scalar CPU tensors without gradients. They
are runtime tensor inputs, so their values can change without recompilation.
No host `.item()` occurs inside the captured objective.

A mapping such as `{"rows_8": split8, "rows_16": split16}` supplies named search
candidates to the ShadowSpill backend. Every candidate must represent the same
update, and the callbacks must remain callable for later source items.
`selected_candidate` names the chosen function; `planning` is its search report
and `plan` is the admitted plan report. The PyTorch backend takes one candidate.

## Updates, schedules and results

`step(data, hyperparams=None)` completes one update, collects small returned
summaries and returns `StepResult`:

- `step`: completed updates, beginning at 1.
- `losses`: per-microbatch normalized contributions; `loss` is their sum.
- `metrics`: one objective-metric pytree per microbatch, copied to CPU.
- `parameter_metrics`: named summaries of compute weights and final gradients.
- `seconds`: execution and host summary-collection wall time.
- `hyperparams`: values applied for this update.
- `summary`: the optional host reducer's scalar metrics and tables.

Schedules receive the zero-based index of the update about to run. Explicit
step hyperparameters override scheduled values for that update. Undeclared
names are rejected before execution. `Constant(value)` and
`WarmupCosine(lr, min_lr, warmup_steps, total_steps)` are supplied schedules;
ordinary callables work as well.

`StepObservations`, `MetricSummary` and `MetricTable` live in
`shadowspill.training.observations`. `parameter_norms` returns raw tensor L2
norms; `parameter_scalars` derives host RMS, gradient/weight ratios and module
shares. Observers must not mutate their inputs. The compiled tasks return
summary tensors; conversion to Python values happens after completion.
Observation tasks use the selected process device. Host collection batches
asynchronous transfers by device and dtype, waits for those transfers, then
copies the small summaries from pinned staging into ordinary CPU storage.
Returned results can safely outlive the trainer and runtime streams.

## Optional loop policy

`fit(source, steps=N, ...)` targets N total completed updates, including any
restored updates. It does not infer duration from data cardinality. Premature
source exhaustion raises a clear error. Default evaluation and checkpointing
are off.

| Argument | Default | Behavior |
| --- | --- | --- |
| `run_dir` | `None` | JSONL metrics and optional plan/search artifacts |
| `log_every` | `1` | Stdout and optional logger cadence; zero disables |
| `tables_every` | `100` | Detail tables at logged updates |
| `logger` | `None` | Any `logger(record)` callable |
| `callbacks` | `()` | Functions `(trainer, result)` after each update |
| `eval_data` | `None` | Iterable, or function creating an iterable per evaluation |
| `eval_every`, `eval_batches` | `0`, `None` | Evaluation cadence and optional source-item limit |
| `checkpoint_every` | `0` | Checkpoint cadence |
| `checkpoint_dir` | `<run_dir>/checkpoints` | Checkpoint destination when enabled |
| `checkpoint_weights` | `"master"` | Save masters where available; `"compute"` saves compute weights instead |
| `keep_last` | `3` | Number of completed checkpoints to retain |
| `startup_diagnostics` | `False` | One zero-LR warmup then one traced step before training; requires a compatible optimizer and ShadowSpill backend |

The final update is included for each enabled cadence. `elapsed_seconds` is
active wall time since preparation, including data/logging/evaluation time and
excluding startup diagnostics and downtime before a resume. The records use completed `step`,
`train/loss`, `train/step_seconds`, `train/elapsed_seconds`, `eval/loss` and
`hyperparameters/<name>`. Reducer names are prefixed by `train/` or `eval/`.

The optional `shadowspill.training.logging.Wandb` helper logs all records with
the actual step axis and retains the SDK's system telemetry defaults. Importing
the trainer does not import or start the logging SDK.

`evaluate(source, batches=None)` uses the current model state and eval mode,
then restores the model's mode. `EvaluationResult` contains update losses,
metrics, elapsed seconds, `mean_loss` and a reducer summary. `mean_loss` is the
arithmetic mean of caller-normalized update losses; use a reducer for another
aggregation rule.

## Startup diagnostics

With MLOps AdamW, enable `fit(..., startup_diagnostics=True, run_dir=...)` to
execute one warmup step and one traced step before the first update of that fit
call. Both use LR=0. The same compiled tasks, optimizer kernels, transfers and
collectives execute; optimizer values are preserved at their stores. No model
weight, master-weight or optimizer-moment snapshot is created.

The optimizer must declare `zero_lr_preserves_state=True`. Ordinary PyTorch
AdamW does not satisfy this contract: it changes moments and counters at zero LR.
The prepared ShadowSpill plan has already initialized optimizer state before
these diagnostic invocations. Models that mutate parameters outside optimizer
tasks are rejected. Model buffers that the selected plan mutates are preserved
separately in host memory, along with RNG state; read-only buffers are not copied.
Side effects outside captured model/optimizer state remain the caller's responsibility.

The first data item is reused for actual update 1. Diagnostics do not advance
training counters or schedules, trigger callbacks/evaluation/checkpoints, enter
training JSONL/W&B history, or count toward reported training elapsed time.
They finish before normal training starts. All distributed ranks enable the same
setting and keep their usual task-local communication behavior.

Each rank writes `startup/step.json`, `selection.json`, `program.json`,
`summary.json`, and `timelines/{index,simulated,traced}.html` under its run
directory (`rank-00000/startup/` etc. for distributed runs). Timeline generation
uses the occupancy tool and the final admitted layout, including distributed
program choices. Kernel structure is retained; zero LR does not promise identical
hardware timing to a positive-LR update.

Custom loops can call:

```python
trainer.diagnose(example_data, directory="runs/example/startup", warmup=1)
```

This method never advances a data source or calls a logger. `warmup=0` requests
only the traced invocation. Plain `trainer.step(data, hyperparams={"lr": 0})`
still counts a loop step; use `diagnose` for uncounted diagnostics.

## Checkpoints

`save(path, source=None, weights="master")` writes a new checkpoint directory atomically.
It contains `state.pt`, `loop.pt` and a version-1 `manifest.json`. It will not
replace an existing checkpoint. Each parameter is saved in one representation:
`weights="master"` saves its master when available and otherwise its compute
weight. `weights="compute"` saves compute weights, even when masters exist.
Loading casts saved values into compute weights and any configured masters;
upcasting compute weights cannot recover discarded master precision. Optimizer
state is saved independently in both cases. `fit(checkpoint_weights=...)` selects
the same policy for periodic saves.

The backend also saves the completed-update count. Loop state includes RNG, elapsed time,
selected candidate and optional source/schedule progress.

Resume with `prepare(example, checkpoint=path)` or `load(path)` on an already
prepared trainer. Preparation pins the saved microbatch candidate before
planning. Exact continuation of an iterable requires caller-managed progress,
or its optional `state_dict()` and `load_state_dict(state)` methods. Stateful
schedules may expose the same optional methods. Ordinary iterables and pure
schedule functions require no wrappers.

Checkpoint directories are trusted local application state; source and schedule
capabilities may serialize ordinary Python objects.

## Forward

`Forward(model, forward_fn=None, backend=None, training=False)` is the matching
forward-only runner. The default call is `model(data)`; an explicit
`forward_fn(model, data)` can unpack arbitrary input structures. The result
retains its pytree structure.

Use `prepare(example, initialize=None, checkpoint=None)`, then `forward(data)`
or `forward.map(source)`. `synchronize()` explicitly waits for completion.
`training` controls module mode; this runner does not build an autograd update.
Close it with `close()` or a context. Passing `trainer.model` and the same
backend shares current imported state and its admitted execution slab. Close
the borrowing forward runner before its training owner.

The low-level `plan_forward(..., forward_fn=...)` likewise captures the supplied
function without renaming model state or changing the original module's forward.

## Backends and devices

`PyTorch(compile=True, device="auto")` executes the same objective with ordinary
PyTorch autograd; `compile=False` selects eager execution. It accepts CPU and
accelerator devices. `ShadowSpill(execution_gib=..., spill_gib=..., device="auto")`
installs its pools on entering the context. It also accepts `artifact_store`,
`search_options`, `profiling_options`, `partition`, `orderings`,
`round_accumulation_once`, `numa_binding=True`, `external_headroom_gib=0.5`, and
`reject_overbudget=False`.

`device` can be an explicit local device/index. Automatic selection uses the
only visible device, or `LOCAL_RANK` when several are visible. It does not map
a global rank modulo device count. Selecting a local device alone does not synchronize training across processes;
provide the `Distributed` binding described below.

## `Distributed`

The optional `Distributed` binding supplies process groups and logical parameter
replicas to preparation. It is available from both `shadowspill.pytorch` and
`shadowspill.training`. Runtime takes a caller-owned Gloo `control_group` before
allocating pools; accelerator communication groups are created afterwards. Optimizer/master sharding
is enabled by default with `shard_optimizer=True`. See the
[distributed guide](distributed.md) for initialization, normalization, task
completion, checkpointing, and the current validation limits.


## Distributed reporting

The trainer always preserves each rank's own console and JSONL records. Add
`DistributedLogger` to also produce aggregate records and grouped W&B runs:

```python
from shadowspill.training.logging import DistributedLogger

with DistributedLogger(
    cpu_control_group,
    run_dir="runs/example",
    device=backend.device,
    wandb={"project": "experiment", "group": "run-001"},  # optional
) as logger:
    trainer.fit(source, steps=1000, run_dir="runs/example", logger=logger)
```

Every rank creates the logger in the same order, and closes it before destroying
the CPU group. The first rank writes `aggregate/metrics.jsonl` and prints lines
prefixed `[aggregate]`. With W&B enabled there is one `rank-NNNNN` run per process
and one `aggregate` run, all in the same group. Each record uses the completed
training step. Detailed tables remain in the corresponding rank's run.

Each rank's W&B system telemetry monitors only its selected GPU. Pass
`device=backend.device` explicitly, or use the current accelerator device (launch-device
resolution before accelerator initialization). W&B's GPU indices are system-monitor
indices, not process-local ordinals or training ranks; UUID-based lookup handles device
masking and reordering.

Rank run names include `node-NNNNN/rank-NNNNN`. The aggregate's
`system/gpu_mapping` table and `aggregate/devices.json` map **node → GPU → rank**:
node ID, hostname, global rank, process-local device, GPU model/UUID, rank
metric prefix, and aggregate metric prefix. Node IDs are assigned in first-rank
order for this run. Each rank also saves `device.json` and its mapping in W&B
config. This makes `gpu.0.*` unambiguous across hosts.

The aggregate's automatic system telemetry covers participating GPUs accessible
on its own host. It does not collect remote hosts' GPU samples. Their entries
have no aggregate metric prefix; follow their per-rank W&B runs. GPU system
series retain W&B's original names and sampling cadence. Device discovery and
the mapping exchange occur once at logger initialization, outside training.

Default aggregation sums `train/loss` contributions, which must already use the
caller's global normalization. It reports `train/step_seconds` as the maximum
rank-local duration, and `train/elapsed_seconds` as the maximum rank-local elapsed
time. These are not timestamps spanning a globally synchronized step. Supplying
`work_units` through the ordinary metric reducer also reports total work divided
by the maximum step duration. Units can be samples, tokens, or another caller
choice; local throughput estimates are never summed.

Pass `reduce=callable` to combine other metrics or implement weighted evaluation.
It receives CPU record dictionaries in group-member order and must retain their
`step`. Evaluation loss, rank-specific parameter metrics and arbitrary summaries
are not implicitly averaged. For EP/CP or replicated observations, the reducer
must avoid counting the same logical contribution multiple times.

Reporting has a separate CPU mailbox and background threads. It adds no runtime
GPU collective or task/step barrier. Bounded `max_pending=256` queues fail clearly
if logging falls behind; `timeout=120` bounds missing-peer records and final
flush. Ranks must use the same record cadence and step order. Initialization and
shutdown coordinate reporting; neither belongs inside a compiled task. Existing
local records remain available after an aggregate reporting failure.

W&B uses explicit Run objects and `reinit="create_new"` for rank 0's two runs,
as described in [W&B's multiple-run API](https://docs.wandb.ai/ref/python/experiments/run/).

Runtime setup automatically discovers device-local host NUMA placement.
`ShadowSpill(..., numa_binding=False)` opts out; see
[host placement](frontend.md#host-numa-placement) for scope and fallback warnings.
