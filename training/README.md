# Training recipes

The installed trainer lives in `shadowspill.training`. Its public interface
takes a saved Plan and runs through Executor. Build accepts ordinary PyTorch
models and objectives; profiling and planning are separate. The library does
not import `workloads`, text data utilities, model catalogs or qualification
policies.

- [Example model catalog](../workloads/MODELS.md): architectures, dimensions,
  precision, trainable parameters, and expert parallelism.
- [Generic training API](../docs/python/api/training.md): initialization,
  custom loops, schedules, evaluation, logging and checkpoints.
- [Generic example](../docs/examples/generic-training.md): regression with
  nested data and explicit microbatch weighting.
- [Benchmark quickstart](../benchmarking/quickstart.md): search, execution,
  tracing and plots for a supplied experiment.
- [Qwen3.5 MoE LoRA](qwen35_lora.md): bounded HF initialization, independent
  phases, batch/ordering selection, held-out evaluation and frozen-base resume.

This directory holds example configs and launchers for the optional
`workloads.recipes.text` recipe. Text packing and its token-normalized objective
are recipe behavior, and can be replaced without changing the trainer.

## Getting started

A fine-tune of a local Hugging Face checkpoint is a config of six keys:

```json
{"checkpoint": {"family": "qwen3", "directory": "/models/Qwen3-1.7B"},
 "data": "data/qwen3", "optimizer_args": {"lr": 1e-5},
 "max_seq_len": 2048, "max_tokens_per_step": 16384, "steps": 1000}
```

`python -m training.prepare finetune.json --dataset <hub dataset> --files
<pattern>` tokenizes the corpus with the checkpoint's tokenizer into the
directory `data` names (`--messages-column` for a chat dataset, whose
assistant turns are then the only targets), and `python -m training.train
finetune.json run_dir=runs/qwen3` trains. `data` given as a directory alone
is the packed documents it holds; the objective defaults to next-token
prediction and the optimizer to the mlops AdamW; a key the runner does not
know is refused with the nearest known name, and a missing required key is
named. From Python, `training.run(config, run_dir=..., steps=...)` takes a
path or a mapping with the same overrides as keyword arguments and returns
the final step's result. `ranks=N` trains on N GPUs of one node, weights,
gradients and optimizer state sharded over them, each rank taking its share
of every step's tokens; the ranks' records sit under `rank-xxxxx/` and the
summed figures under `aggregate/`. `hosts` spreads the ranks over several
machines, each building for its own GPUs, with `shares` giving a faster card
more of every step; see the [configuration reference](../workloads/recipes/text/experiment/README.md).

## Run a supplied text recipe

The JSON-configured launchers use separate build, profile, plan and execution
workers. See the [configuration reference](../workloads/recipes/text/experiment/README.md)
for all fields and saved artifacts. Each phase can also be selected with
`--phase build`, `--phase profile`, `--phase plan` or `--phase execute`.

```bash
python -u -m training.train training/configs/olmoe_1b.json \
  run_dir=training/runs/olmoe_example \
  steps=1000 execution_gib=8 spill_gib=32
```

A budget left out comes from the profiled hardware: the device's memory less
2 GiB, three quarters of the host memory available when profiling ran. The
plan phase prints the budgets it planned and records them in `planning.json`.

The JSON file names constructors and functions explicitly. `@module:name`
resolves a Python object, `{"@call": "module:name", ...}` constructs it, and
`{"@partial": "module:name", ...}` binds its keyword arguments. Overrides use
`key=value` with dotted paths and JSON values; a key segment that holds dots,
such as a schedule's hyperparameter name, is quoted:
`'schedules."optimizer_group.0.lr".total_steps=300'`. The complete resolved
request is recorded in the run directory. A model preset is optional user code.

Data ingestion is also supplied as a recipe:

```bash
python -m workloads.recipes.text.data.prepare --help
```

`PackedTokens` reads the prepared token directory. The recipe packs documents,
chooses microbatch candidates within the token bounds, and supplies summed
losses weighted by the inverse number of valid targets in the entire update.
Packing statistics and token throughput are logged by the recipe's callback.
The generic trainer never infers tokens or units from an input tensor.

## Precision and initialization

Model construction sets compute dtypes. `master_dtype` and `grad_dtype` are
independent builder arguments; optimizer state dtype and rounding are arguments
to the optimizer constructor. For example, the supplied MLOps optimizer supports
FP16 weights and gradients with FP32 optimizer state without master weights:

```bash
python -u -m training.train training/configs/llama3_1b.json \
  model.dtype=float16 grad_dtype=@torch:float16 \
  optimizer_args.opt_state_dtype=@torch:float32 \
  master_dtype=null
```

For stock `torch.optim.AdamW`, use FP32 masters when FP16 compute weights need
FP32 moments. Set `master_dtype=@torch:float32`. This consumes more storage.
No dtype changes are inferred from the selected model or machine in this recipe.

When the build recipe constructs MLOps AdamW, it selects stochastic rounding for BF16
moments by default. FP16 and FP32 moments and parameter rounding retain the
optimizer's nearest-rounding default. Set
`optimizer_args.opt_state_rounding=nearest` to override this, or supply rounding
in an optimizer partial or parameter group. MLOps used directly and
`torch.optim.AdamW` keep their own defaults. Numerical qualification explicitly
uses nearest rounding so existing references remain valid.

The build recipe declares models on meta. Execution uses an explicit bounded
`initialize(request, destination)` callback or checkpoint. Supplied configs use
the shared catalog initializer, including derived buffers and deterministic
per-tensor seeds. `initialize.model` and its geometry must match `model.model`
and its geometry when overriding a preset. Initialization does not allocate an
anonymous full model alongside spill storage.

A fine-tune of a local Hugging Face checkpoint names its family and directory
once:

```json
"checkpoint": {"family": "qwen3", "directory": "/models/Qwen3-1.7B",
               "dtype": "bfloat16", "lora": {"rank": 16, "alpha": 32}}
```

The entry stands for the model, `workloads.models.catalog:build_from_checkpoint`,
declared at the checkpoint's configuration, and the initializer,
`workloads.recipes.text.initialization:from_checkpoint`, which streams the
parameters in bounded chunks and generates the rotary tables and any LoRA
factors as for a catalog model; the two explicit entries remain available.

## Instruction data and the transformers check

`python -m workloads.recipes.text.data.prepare_chat` renders a chat dataset's
conversations through the tokenizer's chat template and writes, beside the
token files, a mask marking the assistant's tokens; a run on that directory
trains only those, never the prompt. `python -m training.reference <run dir>`
runs the Hugging Face model the run was built from on the documents of a step
(the mask included) and reports its loss beside the run's recorded loss, the
check that a checkpoint loaded as transformers reads it.

## Run policy and results

`steps` is the target total completed update count. Schedules are mappings such
as `schedules["optimizer_group.0.lr"]`, evaluated with the zero-based update index. A warmup/cosine
schedule has an explicit `total_steps`; changing the run length does not silently
change the schedule.

`eval_every`, `checkpoint_every`, `log_every`, `metric_tables_every` and
`keep_last` control the optional loop behavior. Resume checkpoints retain model
weights, masters, optimizer state and loop/source progress, and bind to those
tensors rather than to the Plan: a rebuilt run directory, another microbatch
geometry or another budget resumes them. Compute-only or optimizer-omitting
resume policies are rejected. The directory contains:

| Path | Meaning |
| --- | --- |
| `config.json` | Configuration including explicit overrides |
| `programs/`, `profiles/`, `profiled/` | Reusable phase artifacts |
| `planning.json`, `search.json` | Selected geometry and candidate outcomes |
| `plans/`, `report/` | Saved Plans and simulated timeline |
| `logs/` | A log for each preparation/execution phase |
| `metrics.jsonl`, `observations.jsonl` | Completed losses, timing, schedules and optional tables |
| `packing.jsonl` | Document identities and packing statistics |
| `checkpoints/step_XXXXXXXX.ssp` | Model/optimizer/loop/source checkpoint archives |

`eval_microbatches` sets the number of packed examples in each held-out
validation item. Evaluation calls a separately built forward Plan sharing the
current model weights.

`wandb_project` enables the optional logger; the run's id and URL are recorded
in `wandb_run.json` and the runner prints the URL after execution. `step` is
the actual completed training step; evaluation, tables and training records
use the same axis. `wandb` passes additional SDK options. System telemetry keeps the SDK defaults.
Tensor summaries leave compiled tasks as tensors. Host collection and `.item()`
run after the step completes. `parameter_norms_every` enables bounded eager observations before optimizer
updates. These are off by default and can report compute-weight and final-gradient
statistics without compiled observation tasks.

Resume is explicit:

```bash
python -m training.train training/configs/olmoe_1b.json \
  run_dir=training/runs/olmoe_example \
  resume=training/runs/olmoe_example/checkpoints/step_00000500.ssp --phase execute
```

The checkpoint restores model and optimizer state, update count, RNG,
stateful schedules and the packed source's progress. The saved Plans fix the candidate;
execution does not repeat preparation. An ordinary iterable is also valid, but
exact data continuation requires caller-managed progress or the optional
`state_dict`/`load_state_dict` source methods.

## Layout and comparison

- `src/shadowspill/training/`: Trainer over saved Plans, schedules and logging.
- `src/shadowspill/execution/`, `checkpoint/`: execution and bounded state I/O.
- `workloads/recipes/text/`: optional model construction, token ingestion,
  packing, objectives and text experiment composition.
- `training/configs/`: example recipe requests.
- `training/train.py`: recipe CLI.
- `training/scripts/`: launch and comparison helpers.

`engine=pytorch` runs the same configuration in ordinary PyTorch, with one
fixed microbatch size and no saved phases, as the reference for a comparison:

```bash
python -m training.compare <reference run dir> <run dir> [<run dir> ...]
```

The comparison reads actual step numbers, losses and document records from the
saved runs. Model examples and qualification policies remain outside the
installed generic trainer.

## LoRA examples

The trainer uses ordinary `requires_grad` and has no LoRA-specific execution
mode. Apply [`workloads.lora.configure_lora`](../workloads/MODELS.md#trainable-parameters-and-lora)
before building the Program. The `build_on_meta` text recipe accepts a `lora`
configuration dictionary (for example `{"rank": 32, "alpha": 32, "head": "lora"}`)
so configuration-driven training can make the same selection. Optimizer updates and state are created for trainable parameters; frozen state
remains available to forward and input-gradient computation.

For example, reuse the Llama training config with rank-32 LoRA and FP32 factors,
gradients and optimizer state:

```bash
python -u -m training.train training/configs/llama3_1b.json \
  'model.lora={"rank":32,"alpha":32,"head":"lora","factor_dtype":"float32"}' \
  grad_dtype=@torch:float32 optimizer_args.gradient_dtype=@torch:float32 \
  optimizer_args.opt_state_dtype=@torch:float32 \
  run_dir=training/runs/llama3_lora
```

The same `model.lora` override applies to the supplied Qwen3.5 and OLMoE configs.
