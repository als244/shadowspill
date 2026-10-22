# Generic training and forward execution

`shadowspill.training` accepts ordinary PyTorch modules, functions and iterables.
It has no model catalog or text-input convention. Optional recipes under
`workloads/recipes/` are clients of this API. The lower-level
[planning API](frontend.md) remains available for direct runtime ownership and
custom execution loops.

`TransferSharding` is also exported from `shadowspill.training`. It configures
local weight/gradient spill sharding independently of global optimizer/master
ownership; see [distributed ownership](distributed.md).

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

trainer = Trainer(
    model,
    objective=objective,
    optimizer=torch.optim.AdamW,
    optimizer_args={"weight_decay": 0.01},
    schedules={"lr": WarmupCosine(3e-4, 3e-5, 100, 10000)},
    backend=ShadowSpill(execution_gib=20, spill_gib=64),
)
trainer.prepare(example_data)  # Opens pools, searches and admits a callable.
trainer.fit(
    source, steps=10000, run_dir="runs/example",
    log_every=10, eval_data=validation, eval_every=100,
    eval_updates=20, checkpoint_every=500,
    checkpoint_dir="/storage/checkpoints/example",
)
trainer.close()  # Releases the callable and the backend opened by prepare.

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
| `microbatches` | One fixed `fn(data)` yielding `(microbatch_data, loss_scale)`; omitted means the whole update is one microbatch |
| `microbatch_candidates` | Named mapping of split functions; search these candidates during preparation and retain the winner |
| `backend` | `PyTorch()` by default, or a `ShadowSpill(...)` configuration; an already-open backend is borrowed |
| `master_dtype`, `grad_dtype` | Optional optimizer master and gradient accumulation dtypes; neither changes model construction |
| `parameter_metrics` | Optional pure tensor function `(parameter, gradient) -> metrics`, before optimizer updates |
| `metric_reducer` | Optional host function of `StepObservations`, returning a scalar mapping or `MetricSummary` |
| `eval_fn` | Evaluation objective; defaults to the training objective |
| `distributed` | `DataParallel(...)` for managed torchrun groups, or `Distributed(...)` for explicit group/parameter-replica ownership |
| `shard_optimizer` | `True` by default: shard optimizer state and optional masters over parameter replicas when distributed |
| `frozen_state_id` | Optional immutable base identity; omit unchanged frozen weights from all this Trainer's checkpoints and require the same identity on restore |

A live optimizer instance belongs to the low-level `plan_step` API. The trainer
constructs its optimizer after state import, preserving group assignments and
scheduled tuple values. Unscheduled group-specific constants remain unchanged.

Use `microbatch_candidates=data.candidates` to search sizes: the text helper
maps sizes to split functions, so every candidate still consumes the same full
optimizer update. ShadowSpill also searches each size's execution orderings.
Use `microbatches=split` when the application fixes its own splitting function.
Supply only one argument. The older mapping form of `microbatches` remains
accepted for existing callers.

For MLOps AdamW, Trainer defaults BF16 moment storage to stochastic rounding;
FP16/FP32 moments and parameter storage retain nearest rounding. Set
`optimizer_args={"opt_state_rounding": "nearest"}` to override this. Explicit
rounding in a partial or parameter group also wins. With
`opt_state_dtype="parameter"`, mixed-dtype groups are split only where rounding
differs, preserving parameter order and the other group settings. Other
optimizers and caller-owned optimizers passed to `plan_step` keep their settings.

## Data parallel training and resource ownership

`DataParallel` is a configuration; constructing it does not initialize the
accelerator or process groups. `rank` and `world_size` are available from torchrun before
preparation, so a data source can partition its input.

```python
from shadowspill.training import DataParallel

parallel = DataParallel()  # Replicated compute weights; sum gradient contributions.
# Or, shard compute weights and gradients in the spill pool:
parallel = DataParallel(shard_weights=True, shard_gradients=True)
# Pass distributed=parallel to Trainer, using an unopened ShadowSpill backend.
```

| Mode | Compute weights / gradients | Optimizer state / masters |
|---|---|---|
| `DataParallel()` | Replicated weights; gradient reductions across replicas | Sharded by default |
| `DataParallel(shard_weights=True, shard_gradients=True)` | Weight all-gather on fetch, gradient reduce-scatter on eviction | Sharded by default |

`Trainer(shard_optimizer=True)` is the independent default in both modes; set it
false for replicated optimizer state and masters. Transfer sharding requires
symmetric planning; `DataParallel` enables it by default when sharding weights.
Its `timeout` (seconds) controls group creation and distributed preparation.
The caller normalizes loss by the global batch/token count; reductions use sums.

`prepare()` creates Gloo before opening the runtime, then NCCL after allocator
installation. An existing Gloo default group is borrowed. The created groups
span all torchrun ranks; use explicit `Distributed`/`TransferSharding` with an
already-open backend for subgroups or model-specific ownership.

`prepare()` and `prepare_artifacts()` open an unopened backend automatically.
`trainer.close()` closes its callables, then its owned backend, then its owned
process groups. It is idempotent. Preparation failures perform the same cleanup.
A backend that was already open remains caller-owned, so a trainer can share it
with other trainers or `Forward` instances. `ShadowSpill.open()` and `close()`
provide explicit backend control; context managers remain optional.

## Preparation and initialization

`trainer.prepare(example_data, initialize=None, checkpoint=None, prepared=None)` performs
initialization, capture, profiling, planning and admission as needed by the
backend. It returns the trainer and does not increment its update count.
Representative input shapes and structures must match subsequent calls.

### Reuse preparation across budgets and processes

`trainer.prepare_artifacts(example_data, budgets=None, initialize=None)` initializes
state, captures and profiles each candidate, and searches the requested budgets
without admitting a training callable or taking an optimizer step. `budgets` is
a sequence of `(execution_bytes, spill_bytes)` pairs; omitted means the backend's
budget. Every pair must fit the preparation runtime. An exhausted candidate is
recorded as infeasible and other sizes are still considered.

The returned `PreparedTraining` holds `candidates` (named `PreparedStep` artifacts)
and `search` (a `StepSearchReport`). `search.winner_plans[(execution, spill)]` is
the complete selected plan, including its estimated `summary`. Infeasible budgets
have points in the report but no winning plan. `save(directory)` and
`PreparedTraining.load(directory)` persist the captured computations and selected
plans; no model weights, gradients, optimizer state, runtime or live callable is
saved. Load only trusted local artifacts.
For one case, use `PreparedTraining.load(directory, budget=(execution, spill))`:
it reads only that budget's winning candidate, ordering and selected plan, while
retaining its search outcomes and summaries. This avoids parsing every other
budget's plan in each fresh process. Load without `budget` when extending search
over all the prepared candidates and orderings.

Create a new backend with the actual case budget and a new Trainer with matching
model structure, optimizer, precision and microbatch functions. Call
`trainer.prepare(example_data, initialize=..., prepared=artifacts)` to bind fresh
state and admit that budget's selected plan. This bypasses objective export and
search; it still reconstructs task bindings, validates the program and certifies
the physical layout for the current runtime. Use the same artifact store to reuse
graph pairs, task profiles and compiler artifacts. A checkpoint can independently
restore training state. An artifact's computation is authoritative: loading one
does not compare a new Python objective's source code with the original.

Additional budgets within the prepared domain can be searched on the CPU with
`shadowspill.pytorch.search_prepared_steps(artifacts.candidates, budgets=...)`.
Save `PreparedTraining(artifacts.candidates, new_report)` to retain those plans.
Larger capacities or a different model, precision, optimizer or input geometry
require preparation again. Persistent preparation currently supports one rank.
Ordinary distributed `prepare()` retains coordinated capture, search and admission.
The [reusable planning example](../../examples/reusable-planning.md) shows both
the Trainer and lower-level boundaries.

### Preparation and diagnostic random state

`Trainer.prepare()` and `Forward.prepare()` preserve default random streams
after model initialization, including representative microbatch construction.
Initialization retains its ordinary random-number consumption. Startup
`Trainer.diagnose()` restores the same streams after its zero-LR warmup and
traced steps, and restores mutable model buffers. With an optimizer advertising
`zero_lr_preserves_state`, diagnostics leave parameters, moments and step
counters unchanged; the first real update uses the same random stream with
or without diagnostics. See [frontend RNG boundaries](frontend.md#random-state-during-preparation)
for explicit-generator and custom-operator limitations.

Already-initialized state is retained. A meta model requires `initialize=fn` or
a checkpoint. `reset_parameters` is an explicit supplied initializer; no reset
occurs merely because a runner is prepared. Meta materialization preserves tied
parameter identities and shared-storage views. Mixing meta and initialized
state is rejected. An initializer can build nonpersistent buffers before a
checkpoint is applied; the checkpoint then supplies the saved state.

With the ShadowSpill backend, meta initialization runs **after** pool allocation:
pinned-host weights are filled directly in their pool, and non-addressable
pools stage only the roots used by one initialization operation. No full CPU
model is allocated first. The same applies to Forward and quickstart. Callbacks
must write the provided tensors in place and should initialize them individually.

`ShadowSpill(execution_gib=..., spill_pool=...)` accepts a configured spill pool,
including [SSD storage](ssd.md). `spill_gib`, when supplied with a pool, limits
the planner within its capacity; otherwise capacity is the limit. The existing
`spill_gib=...` form constructs pinned host memory. `calibrate=False` permits
explicit calibration through `backend.runtime.calibrate_transfer_capabilities`,
for example with smaller probes on an SSD.

Enter `ShadowSpill` before allocating accelerator state or constructing model
resources on the device. CPU models may be constructed earlier. The backend
owns the imported model state; use `trainer.model` after preparation. Close
runners before the backend. The contexts in the example enforce that order.

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
| `eval_every`, `eval_updates` | `0`, `None` | Cadence in training updates and number of complete validation source items per evaluation |
| `checkpoint_every` | `0` | Checkpoint cadence |
| `checkpoint_dir` | `<run_dir>/checkpoints` | Checkpoint destination when enabled |
| `checkpoint_weights` | `"master"` | Save masters where available; `"compute"` saves compute weights instead |
| `checkpoint_include_optimizer` | `True` | Save optimizer state; `False` resumes with a freshly initialized optimizer |
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

`evaluate(source, updates=None)` uses the current model state and eval mode,
then restores the model's mode. `EvaluationResult` contains update losses,
metrics, elapsed seconds, `mean_loss` and a reducer summary. `mean_loss` is the
arithmetic mean of caller-normalized update losses; use a reducer for another
aggregation rule.

One validation update is one complete source item, split into microbatches using
the selected splitter. It performs no gradient or optimizer update. For
`TokenUpdates(sequences_per_step=8, sequence_length=32768)`, `eval_updates=2`
evaluates 16 sequences and 524,288 target tokens each time evaluation runs.

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

`save(path, source=None, weights="master", include_optimizer=True)` writes a new checkpoint directory atomically.
It contains `state.pt`, `loop.pt` and a version-1 `manifest.json`. It will not
replace an existing checkpoint. Each parameter is saved in one representation:
`weights="master"` saves its master when available and otherwise its compute
weight. `weights="compute"` saves compute weights, even when masters exist.
Loading casts saved values into compute weights and any configured masters;
upcasting compute weights cannot recover discarded master precision.

Optimizer state is included by default. Set `include_optimizer=False` to omit it;
this is independent of the weight precision. For example,
`trainer.save(path, weights="compute", include_optimizer=False)` saves lower
precision parameters without optimizer tensors. `fit(checkpoint_weights=...,
checkpoint_include_optimizer=...)` selects these policies for periodic saves.

A checkpoint without optimizer state can resume through a new Trainer's
`prepare(example, checkpoint=path)`. It restores weights, update count, schedules,
RNG and source progress, while the optimizer starts with fresh moments/counters.
This continues training, but cannot reproduce the original optimizer trajectory.
Loading such a checkpoint into a callable that has already stepped is rejected;
prepare a fresh Trainer so existing optimizer state cannot be reused accidentally.

For partial training, configure `Trainer(..., frozen_state_id="base-revision-and-config")`.
Both periodic saves and `trainer.save()` then omit unchanged frozen parameters.
Selection uses `requires_grad` and mutations, without relying on adapter names or
model families. Each trainable parameter is saved once (its master by default),
alongside optional optimizer state, persistent buffers and mutated frozen state.
ShadowSpill streams only the
selected pool objects; it does not copy the frozen model to a CPU snapshot.
The PyTorch backend retains frozen parameters whose version counters changed.

Resume with the same `frozen_state_id` and initialize the same base before loading:
`trainer.prepare(example, initialize=load_base, checkpoint=path)`. This is required
for a meta model whose checkpoint omits frozen weights. `trainer.load(path)` also
checks the identity before restoring into an already prepared model. The ID is
caller supplied and should identify an immutable base revision plus configuration;
the library compares the ID and state inventory, without hashing the base's bytes.
Omitting `frozen_state_id` keeps full-model checkpoints as the default. The
[LoRA example](../../../workloads/recipes/text/train_qwen35_lora.py) uses partial
checkpoints and reads its frozen base from the original HF directory on resume.

For tensor-wrapper weights, compute checkpoints retain the exact component
payloads and scales. Loading reconstructs their logical type from the receiving
model; the file itself contains ordinary tensors and remains compatible with
`torch.load(..., weights_only=True)`. Use the same representation configuration
when restoring a compute checkpoint.

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
accelerator devices. `ShadowSpill(execution_gib=..., spill_gib=...,
device="auto")` installs its pools when `Trainer.prepare()` opens it, or when
the caller explicitly calls `backend.open()`. It also accepts
`artifact_store`, `search_options`, `profiling_options`, `partition`,
`orderings`, `fuse_narrow_gradient_accumulation=True`, `numa_binding=True`,
`external_headroom_gib=0.5`, and `reject_overbudget=False`. Every step it
prepares, one microbatch list or a search over several, builds and plans in
`artifact_store`, or the library's own store (`~/.cache/shadowspill`) when it
is `None`, in `contribute` mode: what the store holds is read, and what it
lacks is built and written back ([the artifact store](../artifact-store.md)).

`orderings` defaults to every depth/breadth factor pair for a candidate's
microbatch count. Pass a list such as `[StepDataOrdering(depth=1, breadth=4)]`
to fix it for four microbatches, or a function of the microbatch count to select
orderings across different candidate sizes. Even with a fixed size and ordering,
preparation still plans storage, transfers and recomputation within the budget.

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

## Quantized parameter representations

The model chooses its compute representation. Use the existing
`master_dtype=torch.float32` and `grad_dtype=torch.float32` options when training
quantized wrapper parameters with a dense optimizer such as AdamW or SGD.
A wrapper gets a dense master even when its nominal dtype is already FP32.
Its payloads and scales are separate physical state; its gradient remains one
logical tensor. No FP8 flag belongs to the generic Trainer.

The representation must expose its components through PyTorch's tensor
flatten/unflatten protocol and provide traceable `copy_` conversion between
the logical value and dense masters. See the
[state contract](../../architecture/state-import.md#tensor-representations) and
[optimizer ownership](../../architecture/optimizer.md#master-copies-and-the-dtype-gradients-are-kept-at).
