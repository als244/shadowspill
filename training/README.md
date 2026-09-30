# Training

A small, generic harness for training a model on real text: the same loop on
plain PyTorch or on ShadowSpill, with packed documents, a learning-rate
schedule, evaluation, checkpoints that resume, and metrics on stdout, in a
JSON-lines file and on W&B. It knows no model, tokenizer, dataset or optimizer
of its own: the caller names them -- in Python, or in a JSON config. The
example configs train the repository's reference workloads (`workloads/`) on
FineWeb-Edu.

## Contents

1. [Quick start](#quick-start)
2. [Setup](#setup)
3. [Data](#data)
4. [The trainer](#the-trainer)
5. [Configs](#configs)
6. [Backends](#backends)
7. [What a run writes](#what-a-run-writes)
8. [Checkpoints and resuming](#checkpoints-and-resuming)
9. [Comparing runs](#comparing-runs)
10. [Extending it](#extending-it)
11. [Layout](#layout)

## Quick start

From Python, a run is one `Trainer` and one call:

```python
import torch

from training.backends.shadowspill import ShadowSpill
from training.data import PackedTokens
from training.models import build_on_meta
from training.objectives import model_loss
from training.schedules import WarmupCosine
from training.trainer import Trainer

trainer = Trainer(
    "training/runs/demo",
    model=build_on_meta(MyModel, dtype="bfloat16", config=my_config),
    objective=model_loss,
    optimizer=torch.optim.AdamW,
    optimizer_args={"betas": (0.9, 0.95), "weight_decay": 0.1},
    data=PackedTokens("training/data/tokens/llama3"),
    steps=1000,
    max_seq_len=2048,
    max_tokens_per_step=65536,
    max_tokens_per_microbatch=8192,
    schedule=WarmupCosine(lr=3e-4, min_lr=3e-5, warmup_steps=100),
    backend=ShadowSpill(execution_gib=8, spill_gib=32),
)
trainer.train()
```

From the command line, the same run is a JSON config:

```bash
training/scripts/launch.sh training/runs/llama3_1b training/configs/llama3_1b.json
```

and any config value can be replaced for one run by its dotted key, here a
20 GiB budget and a run without W&B:

```bash
training/scripts/launch.sh training/runs/llama3_1b_20gib training/configs/llama3_1b.json \
    backend.execution_gib=20 wandb_project=null
```

`training/scripts/pair.sh` runs a config on PyTorch and on ShadowSpill and
compares the two; `training/scripts/smoke.sh` checks the whole pipeline in a
few steps on both.

## Setup

Install ShadowSpill as the [repository README](../README.md) says. Its setup
script installs the harness's dependencies with it -- the package's `training`
extra -- and `mlops`, the operation library the example configs' models run
on; the harness itself does not import `mlops`. To send metrics to W&B, log in
once:

```bash
wandb login
```

## Data

`training.prepare_data` downloads a text dataset from the Hugging Face hub and
tokenizes it once per tokenizer:

```bash
python -m training.prepare_data --tokenizer NousResearch/Meta-Llama-3-8B \
    --dataset HuggingFaceFW/fineweb-edu --files 'sample/10BT/{:03d}_00000.parquet' \
    --out training/data/tokens/llama3
```

It writes `train.bin` and `val.bin` -- every document's token ids followed by
the tokenizer's end-of-text id, back to back, as uint32; the first
`--val-tokens` tokens (5M by default) are validation -- and `meta.json`, which
records the tokenizer, its end-of-text id and vocabulary size, and what was
read. `--shards` reads more files; downloads go to the Hugging Face cache
(`HF_HOME`). The example configs read `training/data/tokens/<family>`, from the
tokenizers `NousResearch/Meta-Llama-3-8B` (`llama3`),
`Qwen/Qwen3.5-9B-Base` (`qwen35`) and `allenai/OLMoE-1B-7B-0924` (`olmoe`).

`PackedTokens(directory, long_documents="drop", min_tokens_per_seq=128,
window=64)` packs those documents into microbatches as sequences of at most
`max_seq_len` tokens, the trainer's sequence-length limit.

The sequence capacity must leave at least one document slot after reserving
padding slots. An invalid `min_tokens_per_seq`/microbatch combination raises a
clear error; lower `min_tokens_per_seq` when using very small microbatches.

- A document of at most `max_seq_len` tokens (its end-of-text included) is one
  sequence. A longer one, as `long_documents` says, is dropped (`"drop"`);
  truncated to its first `max_seq_len` tokens (`"truncate"`); or spliced into
  consecutive pieces of `max_seq_len` tokens, the last taking the rest, each a
  sequence of its own (`"splice"`).
- Sequences are packed back to back into microbatches of at most
  `max_tokens_per_microbatch` tokens, each taking the longest of the next
  `window` sequences that still fits; the tail a microbatch cannot fill is
  padding.
- Each microbatch is `[tokens, targets, seq_lens]`: tokens and targets of shape
  `[1, max_tokens_per_microbatch]`, and the sequences' lengths as a zero-padded
  int32 tensor, so attention stays inside each sequence and one planned step
  serves every packing. It has room for one sequence per `min_tokens_per_seq`
  tokens -- its `seq_slots` -- which bounds how many sequences it holds.
- A sequence's targets are its tokens shifted by one. At its last position the
  target is the document's next token when the document goes on (a truncated
  document, or a spliced piece before the last), and `-100` -- which the loss
  does not read -- at the document's end, whose next token is another
  document's; the padding is `-100` too.
- Packing is deterministic: microbatch *i* holds the same sequences in every
  run, so both backends train on identical data and a resumed run continues
  where it stopped.

## The trainer

`Trainer(run_dir, ...)` takes the run's directory -- where its records go --
and, by keyword:

| Argument | What it is |
|---|---|
| `model` | The model on `meta`: structure only. `training.models.build_on_meta(model, dtype, **arguments)` builds one. The backend materializes it and every module initializes its own storage from `seed`, in module order, so both backends start from the same weights. |
| `objective`, `objective_args` | The loss a step differentiates, `objective(model, tokens, targets, seq_lens, **objective_args)`, returning the model's loss summed over the positions the targets train. `training.objectives.model_loss` calls the model's own `loss(tokens, targets, seq_lens=..., reduction="sum", **objective_args)`. The module wrapping the model divides that sum by its `trained_total` buffer, the trained positions of the whole step, which the trainer sets before each step as a hyperparameter beside the learning rate. So each microbatch returns its share of the step's mean loss over trained tokens, a step's loss is the sum of its microbatches' shares whatever the geometry, its gradient weighs every trained token by that one total, and padding counts nowhere. |
| `optimizer`, `optimizer_args` | The optimizer class, built as `optimizer(parameters, **optimizer_args)`. |
| `master_dtype` | A dtype -- `torch.float32` -- to keep a master copy of every weight trained at another dtype at. The optimizer steps the masters in the weights' place and each step writes the weights from them, rounded to nearest; without it the optimizer steps the weights themselves and rounds each update to their dtype -- mlops AdamW to nearest, or stochastically, which keeps small updates in expectation, with `"parameter_rounding": "stochastic"` in `optimizer_args`. |
| `grad_dtype` | The dtype gradients are summed at over a step's microbatches, the weights' own when not given; normally `torch.float32` with fp32 masters. A model on mlops kernels asks them for its weight gradients at the same dtype among the `settings` -- `mlops.dispatch:set_weight_gradient_dtype` -- so those they sum come back unrounded; the fp32 config does. The optimizer must read the gradients at that dtype too: mlops AdamW reads them at its own `gradient_dtype`, `torch.bfloat16` unless given, so without it in `optimizer_args` every update rounds fp32 gradients to bf16 first, and nothing reports it. `"parameter"` reads them at the dtype of what the optimizer steps -- right over fp32 masters, as in the fp32 config; without masters, name the dtype. |
| `optimizer_args.opt_state_dtype` | The dtype the optimizer keeps its state at -- AdamW's moments -- which is the optimizer's own setting, since only the optimizer can keep its state apart from its parameters. mlops AdamW takes `opt_state_dtype`: `torch.bfloat16` unless given, whatever `master_dtype` is, or `torch.float32`, or `"parameter"` for the dtype of what it steps; `opt_state_rounding` rounds what it stores to nearest, or `"stochastic"`. `torch.optim.AdamW` has neither: its moments take the dtype of what it steps, the masters or else the weights. ShadowSpill plans the state at whatever dtype the optimizer makes it. |
| `data` | A `PackedTokens`. |
| `steps` | The optimizer steps the run takes. |
| `max_seq_len` | The longest sequence a microbatch holds; the data drops, truncates or splices longer documents, and planning assumes sequences of this length. |
| `max_tokens_per_step`, `max_tokens_per_microbatch` | Tokens a step and a microbatch hold at most; both multiples of `max_seq_len`, the second dividing the first. The PyTorch backend runs `max_tokens_per_microbatch`; ShadowSpill does too unless its backend names planning bounds, and searches for the fastest at its budget when neither is given. |
| `schedule` | `training.schedules.WarmupCosine(lr, min_lr, warmup_steps)` or `Constant(lr)`, setting the optimizer's rate every step; without one, the optimizer's own rate stays. |
| `backend` | `training.backends.pytorch.PyTorch()` (the default) or `training.backends.shadowspill.ShadowSpill(...)`; see [Backends](#backends). |
| `seed` | Seeds the initial weights. |
| `eval_every`, `eval_batches` | Evaluate on the first `eval_batches` validation microbatches every `eval_every` steps and after the last; 0 never does. |
| `checkpoint_every`, `checkpoint_dir` | Checkpoint every so many steps and after the last (0 never does), into `checkpoint_dir` -- by default the run's directory. |
| `artifact_store` | Where planning keeps what it builds -- captures, compiled graphs, profiles, plans -- for later runs to reuse; by default `artifact_store` inside the run's directory. Runs that share one plan faster. |
| `wandb_project`, `wandb_mode` | Send the metrics to this W&B project too, `online` or `offline`. |
| `parameter_metrics` | Optional pure `(compute_weight, accumulated_gradient) -> tensor pytree` callback. `training.observations.parameter_norms` returns FP32 L2 norms of both. Observed once per step, before updates, with the same semantics on both backends. |
| `metric_reducer` | Optional CPU reducer of the objective's per-microbatch metrics into `MetricSummary(scalars, tables)`. Required when an objective returns metrics. |
| `metric_tables_every` | Write detail tables every this many steps, plus the first and last (default 100; 0 disables). Scalar metrics are logged every step. |

### Loss, routing and parameter observations

An objective can return `ObjectiveResult(summed_loss, metrics)`. The wrapper
divides only the loss by the step's trained-token total; it detaches the metric
leaves without changing their values. Both backends return each microbatch's
metrics in order and once-per-step `parameter_metrics` separately. The
ShadowSpill callable returns raw device tensors. The backend copies only
these summaries to pinned CPU storage and waits at the existing step boundary;
CPU reduction, `.item()` and logging happen afterwards. No logging or host
synchronization runs inside captured tasks.

For the mlops OLMoE workload, add these fields to a training config:

```json
{
  "objective": "@training.objectives:model_loss_with_metrics",
  "metric_reducer": "@training.olmoe_metrics:reduce_metrics",
  "parameter_metrics": "@training.observations:parameter_norms",
  "metric_tables_every": 100
}
```

The model exposes statistics already produced by `mlops.moe`, without a second
router pass. The reducer weights CE and auxiliary loss by trained targets and
sums counts across microbatches **before** computing expert-load entropy.
Routing counts cover all router rows, including any packed padding. Load
entropy describes aggregate expert usage, not mean per-token router entropy.

W&B and local JSON use these names:

| Section | Values |
|---|---|
| `train/loss/`, `eval/loss/` | `cross_entropy`, `auxiliary` (raw), `weighted_auxiliary`, `total` |
| `train/routing/layer_00/` (and `eval/`) | Load entropy, normalized entropy, effective experts, maximum/mean load, unused experts, assignments, auxiliary loss |
| `train/routing/expert_counts` (and `eval/`) | Indexed table: layer, expert, count, share, probability sum |
| `param_norm/<module>/<parameter>` | Pre-update compute-weight L2 norm, reduced in FP32 |
| `grad_norm/<module>/<parameter>` | Final accumulated-gradient L2 norm, reduced in FP32 |
| `param_norm/global/l2`, `grad_norm/global/l2` | Square root of the sum of squared per-parameter norms |
| `parameters/norms` | Indexed table with full parameter name, shape, dtype, metric and value |

W&B's internal `_step` and the logged `step` both use the zero-based training
step. Scalar and table calls for that step accumulate into one W&B history row,
including any evaluation performed after that update. The row is committed
when the next step is logged or the run closes; setup metrics join the first
step's row. Evaluation every 100 updates therefore appears at steps 99, 199,
and so on, rather than advancing a separate logging counter.

Scalar results are flushed immediately to
`metrics.jsonl`; tables are flushed to `observations.jsonl`. Stdout keeps its
short step lines, while detailed metrics go to files and W&B. Norms require
reading the parameters and gradients; their GPU cost is profiled and planned,
even though the returned scalars are small.

For stock `torch.optim.AdamW`, use FP32 masters when training an FP16 model
with FP32 optimizer state. The optimizer steps the masters; ShadowSpill writes
the updated values back to the FP16 model weights. Set `master_dtype` and
`grad_dtype` to `"@torch:float32"` in a training config, or pass
`master_dtype=torch.float32, grad_dtype=torch.float32` to `plan_step` or
`build_step`. AdamW's `exp_avg` and `exp_avg_sq` then remain FP32, including
through a ShadowSpill checkpoint restore. Stock AdamW has no independent
`opt_state_dtype` argument. `mlops.optim.AdamW` supports FP32 moments with FP16
parameters directly, without master parameters, by explicitly setting
`opt_state_dtype=torch.float32` (its default remains BF16).

mlops workloads use automatic operation selection, including during graph
capture. On GPUs below compute capability 8.0, attention selects PyTorch SDPA
instead of the Ampere-or-newer FlashAttention provider. Explicit mlops
`use_implementations` settings remain available for deliberate overrides.

`train()` trains from wherever the run stands to its last step and returns the
last metrics it logged. `setup()` builds the backend without training -- on
ShadowSpill it plans the step -- and returns `plan`: ShadowSpill's
`PlanSummary`, what the plan promises. `trainer.plan.simulated_step_seconds` is
the step time the plan expects, made exactly of
`unconstrained_step_seconds` (the compute alone), `recomputation_overhead_seconds`,
`idle_seconds` and `terminal_writeback_seconds`; it also has the transfer
volumes and the spill pool's peak. `trainer.planning` is what the plan was made
for: geometry and ordering. `close()` releases the backend; `train()` does that
when it ends, however it ends.

## Configs

A config is one JSON object of `Trainer` arguments. Three forms name Python
objects, so a config can choose the model, objective, optimizer and data
without the harness knowing any of them:

- `"@module:name"` is the object `name` in `module` -- `"@torch.optim:AdamW"`;
- `{"@call": "module:name", ...}` is what calling it with the other keys
  returns -- `{"@call": "training.data:PackedTokens", "directory": ...}`;
- `{"@partial": "module:name", ...}` is `functools.partial` of it with the
  other keys.

`name` may be dotted, and the forms nest. A config's `settings` is a list of
calls made first, in order, for state a run needs set process-wide: the
example configs choose which kernels the reference models' operations run and
ask for ordered ones. From Python, make those calls before building the trainer.

```bash
python -m training.train <config.json> [key=value ...]
```

trains the run a config describes, in its `run_dir`. Each override replaces a
value by dotted key, read as JSON or else taken as a string: `run_dir=...`,
`steps=100`,
`backend.execution_gib=12`, `wandb_project=null`,
`'backend={"@call": "training.backends.pytorch:PyTorch"}'`.
`training/scripts/launch.sh <run dir> <config.json> [key=value ...]` does the
same for the run directory it is given, from the repository root so a config's
relative paths are the repository's, and keeps the run's output in `stdout.log`
as well.

The example configs in [`configs/`](configs/) train the reference workloads'
~1B models with mlops AdamW and a warmup-cosine schedule, on ShadowSpill at an
8 GiB budget: `<family>_1b.json` for 1000 steps of 16K tokens, and
`<family>_1b_300m.json` for 300M tokens at 64K tokens a step, checkpointing
every 50M -- all at bf16, weights and moments alike. Two more train the
300M-token Llama-3 run at a 20 GiB budget with other precisions:
`llama3_1b_300m_fp32.json` with fp32 masters, gradients and moments, and
`llama3_1b_300m_bf16_sr.json` at bf16 throughout, rounding both the weights'
and the moments' updates stochastically.

### Configuring FP16 training on an older GPU

The model dtype is `model.dtype`, passed to `build_on_meta` before any weight
storage is allocated. Both the PyTorch and ShadowSpill backends preserve those
parameter dtypes when materializing the model. The example configs retain their
BF16 defaults; choose FP16 explicitly on a GPU without BF16 support:

```bash
python -m training.train training/configs/llama3_1b.json \
  run_dir=training/runs/llama3_fp16 \
  model.dtype=float16 \
  optimizer_args.opt_state_dtype=@torch:float32 \
  optimizer_args.gradient_dtype=parameter \
  grad_dtype=@torch:float16 master_dtype=null \
  backend.execution_gib=8 backend.spill_gib=32 wandb_project=null
```

This uses FP16 weights and accumulated gradients, FP32 AdamW moments, and no
master weights. `master_dtype=@torch:float32` adds FP32 masters while leaving
model weights in FP16; `grad_dtype=@torch:float32` independently selects FP32
accumulation. Omit `grad_dtype` to use the weights' dtype. Optimizer state has
its own setting and does not inherit the master dtype in mlops AdamW.

When requesting FP32 weight gradients directly from mlops kernels, also select
`mlops.dispatch:set_weight_gradient_dtype` in the config's `settings`, as in
`configs/llama3_1b_300m_fp32.json`. The trainer stays independent of a particular
operation library. An FP16 config can omit this setting because mlops kernels
then return weight gradients at the weights' dtype. The run's saved
`config.json` retains the model, master, gradient, and optimizer settings.

From Python, the same choices are:

```python
import mlops
import torch

model = build_on_meta(MyModel, dtype="float16", config=my_config)
trainer = Trainer(
    ...,
    model=model,
    optimizer=mlops.optim.AdamW,
    optimizer_args={"gradient_dtype": "parameter", "opt_state_dtype": torch.float32},
    master_dtype=None,  # or torch.float32
    grad_dtype=torch.float16,  # or torch.float32, independently
)
```

## Backends

A step is `max_tokens_per_step` tokens in microbatches of at most
`max_tokens_per_microbatch`: the geometry. Both backends run the same
microbatches through the same objective, add up gradients across them, and
return each microbatch's share of the step's loss. A step is given two
values every time, the way a learning-rate schedule gives its rate: `lr`,
under a schedule, and `trained_total`, the step's trained positions, which
the objective divides by; an evaluation is given its set's total the same
way. ShadowSpill sets both as hyperparameters of the planned step, PyTorch
writes them into the optimizer and the module.

`PyTorch(compile=True, device=None)` keeps the model, gradients and optimizer
state on the device (by default, PyTorch's current accelerator) and compiles
the objective with `torch.compile` unless `compile=False`. It needs
`max_tokens_per_microbatch`.

`ShadowSpill(execution_gib, spill_gib, eval_execution_gib=None,
round_accumulation_once=False, planning_min_tokens_per_microbatch=None,
planning_max_tokens_per_microbatch=None, resolution_options="quarters",
orderings="factors")` keeps the state in a pinned host pool of
`spill_gib` and plans every step to fit `execution_gib` of the device, which is
the whole device pool. Evaluation's forward pass shares the step's slab
(`share_slab_with`): the two run in turn, so the pool holds the bytes once. It
plans within the step's own budget -- the whole slab they share -- unless
`eval_execution_gib` names less; a budget larger than the slab stops the run at
setup. `round_accumulation_once` is `plan_step`'s: with bf16 gradients a matrix
multiply adds its product into the running gradient as it writes it, rounding
the sum once where the PyTorch backend rounds it twice, so the two backends'
steps no longer agree bit for bit.

- **Planning.** A run's first launch searches for its plan with
  `plan_step_search` and records what it chose in `planning.json`: the
  geometry, the ordering, the search options, and the transfer bandwidths and
  budgets it was planned against; the search's whole table is `search.json`
  beside it. Every later launch of the run plans that choice directly with
  `plan_step`, from the artifact store's captures, compiled graphs and
  profiles, without searching; it refuses other budgets or other search
  options than the record's, since a run keeps the plan it started with.
- **The geometry.** The search plans every split of `max_tokens_per_step`
  whose microbatch holds between `planning_min_tokens_per_microbatch` and
  `planning_max_tokens_per_microbatch` tokens and runs the fastest; equal
  bounds pin one split, a missing bound is open. Without either bound the
  trainer's `max_tokens_per_microbatch` pins the geometry, and without that
  too every split is searched -- the narrow ones included, which cost the
  most to plan, so a floor is worth naming. The data is then packed at the
  geometry the search chose.
- **Search options.** `resolution_options` names the shares of the flexible
  graph-pair groups the search plans recomputing, one resolved program each:
  `quarters` (the library default), `eighths`, `halves`, or a list of exact
  fractions such as `["0", "1/2", "1"]`. `orderings` names the walks tried
  per geometry: `factors`, every depth x breadth factor pair, or `depth-first`
  alone. More of either plans more programs per point, so the first launch's
  search wall grows with them; both are part of every plan's identity in the
  store.
- **Optimizer state.** ShadowSpill creates the optimizer's state in its pool
  before any step runs, each entry where the optimizer's own first step starts
  it -- moments at zero, for instance -- and a resumed run's checkpoint
  replaces it. The PyTorch backend builds nothing ahead: the optimizer makes
  its own state on the first step.
- **Masters.** ShadowSpill keeps them with the optimizer's state in its pool
  (`plan_step(master_dtype=..., grad_dtype=...)`); the PyTorch backend keeps
  them on the device and does the same around an ordinary optimizer step.
- **Evaluation** runs a forward pass planned at setup, right after the
  training step and over the same weights, so a run that cannot evaluate stops
  before it trains.

## What a run writes

Everything a run writes is in its directory:

| File | What it holds |
|---|---|
| `config.json` | The config the run was launched with, or a description of the trainer's arguments. |
| `metrics.jsonl` | One JSON record per line, each with its `step` and `time`. |
| `packing.jsonl` | The sequences each step trained on, each as `[ordinal, length]` -- its document's place among the file's documents, and its length -- with the offset in the document added for a spliced piece that does not start it. |
| `stdout.log` | Everything the run printed, when launched through `launch.sh`. |
| `plan.json`, `planning.json`, `search.json` | On ShadowSpill: the plan's summary, what it was made for, and the geometry search's table. |
| `checkpoint.pt` | The latest checkpoint, in `checkpoint_dir` when that is elsewhere. |
| `wandb/`, `wandb_id.txt` | W&B's files, when a project is given. |

Each step prints one line and records the same values: `loss`, the mean over
the positions the targets train; `lr`; `step_seconds`; and `tokens_per_second`,
the trained targets over the step's time -- padding and each document's last
position are not tokens trained on. A step's time runs from its start to the
next step's start, because a backend returns from a step before the device has
finished it (ShadowSpill's end-of-step writeback overlaps what the trainer does
in between); before an evaluation, a checkpoint or the end of the run the
trainer waits for the step to finish, and its time runs to there. So a step's
line appears when the next step starts, and neither an evaluation nor a
checkpoint overlaps a step or is charged to one. Each evaluation adds `val_loss`,
`eval_seconds`, `device_peak_gib` and `host_rss_gib`. Off stdout, each step also
records what it packed under `packing/`: `sequences`, the fewest and most in one
microbatch, their mean and median length, `fill` (the share of the
microbatches' tokens they fill), and `trained_tokens` for the step and in total.
The first line records the setup: its seconds and the geometry, and on
ShadowSpill the planned step time, with the whole plan summary under `plan/`.
With a W&B project every value goes to W&B too, one row per step.

## Checkpoints and resuming

A checkpoint holds the model's and the optimizer's state and the steps taken.
A weight trained over a master copy is written as its master, under the
weight's name, so the training state is in it once and at full precision, and
it loads into a plain model of either precision as it stands. ShadowSpill
writes checkpoints straight from its pool, without a copy of the state first;
both backends write the same format. A checkpoint is
written to `checkpoint.partial` and then renamed, so `checkpoint.pt` is always
whole.

Training a run directory again resumes it: the trainer loads the checkpoint,
continues from its step with the same microbatches it would have reached, and
keeps counting trained tokens where they stood.

## Comparing runs

```bash
python -m training.compare <reference run dir> <run dir> [<run dir> ...]
```

prints, for each run against the first, whether they trained on the same
documents at every step, the step-0 loss difference, the largest difference
over the first 10 steps, the mean difference over each 100 steps and the final
validation loss, and draws the curves into the reference run's `compare.png`.

## Extending it

- **Another objective** is a function of the model and a microbatch that
  returns the loss summed over the positions the targets train; pass it, with
  its options in `objective_args`. The division by the step's trained total is
  the wrapping module's, so the objective never sees it.
- **Another data source** is one the packer can read: for each sequence, its
  `lengths`, `max_len` and `eos`, and its `tokens`, `targets`,
  `trained_tokens` and `record` -- as `training.documents.TokenDocuments` gives
  them for pretraining. A source whose targets ignore a prompt trains
  supervised fine-tuning with nothing else changed.
- **Another backend** is anything with the methods of
  `training.backends.Backend`.

## Layout

```text
training/
├── README.md
├── trainer.py        the Trainer: setup, the loop, evaluation, checkpoints, logs
├── train.py          the command line: a run from a JSON config
├── config.py         JSON configs: overrides, and the forms that name objects
├── data.py           PackedTokens, the data a run trains on
├── documents.py      sequences from a token file: tokens, targets
├── packing.py        sequences packed into microbatches, and their stats
├── models.py         building a model on meta, and initializing it
├── objectives.py     the objective and the module that computes it
├── schedules.py      learning-rate schedules
├── checkpoints.py    the checkpoint format
├── metrics.py        stdout, metrics.jsonl and W&B
├── compare.py        comparing runs step by step
├── prepare_data.py   tokenizing a dataset into the files a run packs
├── backends/
│   ├── __init__.py   what a backend is given and does
│   ├── pytorch.py    the PyTorch backend
│   └── shadowspill.py
├── configs/          example configs for the reference workloads
└── scripts/          launch.sh, pair.sh, smoke.sh
```

Data, runs and the artifact store live under `training/` too, but only the
harness, its configs, scripts and this guide are tracked.
