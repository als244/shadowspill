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

`training/scripts/pair.sh` runs a config on PyTorch and on ShadowSpill and
compares the two; `training/scripts/smoke.sh` checks the whole pipeline in a
few steps on both.

## Setup

Install ShadowSpill as the [repository README](../README.md) says, then the
harness's own dependencies:

```bash
pip install -e '.[training]'
wandb login  # only to send metrics to W&B
```

The example configs also need the operation library their models run on,
`mlops`, installed beside it. The harness itself does not import it.

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
`max_seq_len` tokens, the trainer's:

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
| `objective`, `objective_args` | The loss a step differentiates, `objective(model, tokens, targets, seq_lens, **objective_args)`. `training.objectives.model_loss` calls the model's own `loss(tokens, targets, seq_lens=..., **objective_args)`. By convention the loss sums over trained positions and divides by all the microbatch's positions, so a step weighs every trained token alike and the trainer can report the loss per trained token. |
| `optimizer`, `optimizer_args` | The optimizer class, built as `optimizer(parameters, **optimizer_args)`. |
| `data` | A `PackedTokens`. |
| `steps` | The optimizer steps the run takes. |
| `max_seq_len` | The longest sequence a microbatch holds; the data drops, truncates or splices longer documents, and planning assumes sequences of this length. |
| `max_tokens_per_step`, `max_tokens_per_microbatch` | Tokens a step and a microbatch hold at most; both multiples of `max_seq_len`, the second dividing the first. Without `max_tokens_per_microbatch`, ShadowSpill searches for the fastest at its budget. |
| `schedule` | `training.schedules.WarmupCosine(lr, min_lr, warmup_steps)` or `Constant(lr)`, setting the optimizer's rate every step; without one, the optimizer's own rate stays. |
| `backend` | `training.backends.pytorch.PyTorch()` (the default) or `training.backends.shadowspill.ShadowSpill(...)`; see [Backends](#backends). |
| `seed` | Seeds the initial weights. |
| `eval_every`, `eval_batches` | Evaluate on the first `eval_batches` validation microbatches every `eval_every` steps and after the last; 0 never does. |
| `checkpoint_every`, `checkpoint_dir` | Checkpoint every so many steps and after the last (0 never does), into `checkpoint_dir` -- by default the run's directory. |
| `artifact_store` | Where planning keeps what it builds -- captures, compiled graphs, profiles, plans -- for later runs to reuse; by default `artifact_store` inside the run's directory. Runs that share one plan faster. |
| `wandb_project`, `wandb_mode` | Send the metrics to this W&B project too, `online` or `offline`. |

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
every 50M.

## Backends

A step is `max_tokens_per_step` tokens in microbatches of at most
`max_tokens_per_microbatch`: the geometry. Both backends run the same
microbatches through the same objective, add up gradients across them, and
return one loss per microbatch.

`PyTorch(compile=True, device=None)` keeps the model, gradients and optimizer
state on the device (by default, PyTorch's current accelerator) and compiles
the objective with `torch.compile` unless `compile=False`. It needs
`max_tokens_per_microbatch`.

`ShadowSpill(execution_gib, spill_gib, eval_execution_gib=None)` keeps the
state in a pinned host pool of `spill_gib` and plans every step to fit
`execution_gib` of the device, which is the whole device pool. Evaluation plans
its forward pass into the step's slab (`share_slab_with`): the two run in turn,
so the pool holds the bytes once. It plans within `eval_execution_gib` when
given -- the example configs give 4 GiB of the step's 8 -- and within the whole
slab when not.

- **Planning.** A run's first launch searches for its plan with
  `plan_step_search` -- over microbatch sizes when the trainer gave none -- and
  records what it chose in `planning.json`: the geometry, the ordering, and the
  transfer bandwidths and budgets it was planned against. Every later launch of
  the run plans that choice directly with `plan_step`, from the artifact
  store's captures, compiled graphs and profiles, without searching; it refuses
  other budgets than the record's, since a run keeps the geometry it started
  with.
- **Optimizer state.** ShadowSpill creates the optimizer's state in its pool
  before any step runs, each entry where the optimizer's own first step starts
  it -- moments at zero, a master copy of the weights at the weights -- and a
  resumed run's checkpoint replaces it. The PyTorch backend builds nothing
  ahead: the optimizer makes its own state on the first step.
- **Evaluation** plans the forward pass once, beside the training step, over
  the same weights.

## What a run writes

Everything a run writes is in its directory:

| File | What it holds |
|---|---|
| `config.json` | The config the run was launched with, or a description of the trainer's arguments. |
| `metrics.jsonl` | One JSON record per line, each with its `step` and `time`. |
| `packing.jsonl` | The sequences each step trained on, each as `[ordinal, length]` -- its document's place among the file's documents, and its length -- with the offset in the document added for a spliced piece that does not start it. |
| `stdout.log` | Everything the run printed, when launched through `launch.sh`. |
| `plan.json`, `planning.json` | On ShadowSpill: the plan's summary, and what it was made for. |
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
A model entry that an optimizer entry reproduces bit for bit by a cast -- a
weight kept beside the higher-precision master it is the rounding of -- is
written once, as the optimizer's entry, and `model_from_optimizer` says where it
comes from. ShadowSpill writes checkpoints straight from its pool, without a
copy of the state first; both backends write the same format. A checkpoint is
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

- **Another objective** is a function of the model and a microbatch; pass it,
  with its options in `objective_args`.
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
