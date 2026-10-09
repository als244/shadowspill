# Training recipes

The installed trainer lives in `shadowspill.training`. It accepts ordinary
PyTorch models, objectives and data iterables. It does not import `workloads`,
text data utilities, model catalogs, or qualification policies.

- [Example model catalog](../workloads/MODELS.md): architectures, dimensions,
  precision, trainable parameters, and expert parallelism.
- [Generic training API](../docs/python/api/training.md): initialization,
  custom loops, schedules, evaluation, logging and checkpoints.
- [Generic example](../docs/examples/generic-training.md): regression with
  nested data and explicit microbatch weighting.
- [Benchmark quickstart](../benchmarking/quickstart.md): search, execution,
  tracing and plots for a supplied experiment.

This directory holds example configs and launchers for the optional
`workloads.recipes.text` recipe. Text packing and its token-normalized objective
are recipe behavior, and can be replaced without changing the trainer.

## Run a supplied text recipe

```bash
python -u -m training.train training/configs/olmoe_1b.json \
  run_dir=training/runs/olmoe_example \
  steps=1000 schedules.lr.total_steps=1000 \
  backend.execution_gib=8 backend.spill_gib=32
```

The JSON file names constructors and functions explicitly. `@module:name`
resolves a Python object, `{"@call": "module:name", ...}` constructs it, and
`{"@partial": "module:name", ...}` binds its keyword arguments. Overrides use
`key=value` with dotted paths and JSON values. The complete resolved request
is recorded in the run directory. A model preset is optional user code.

Data ingestion is also supplied as a recipe:

```bash
python -m workloads.recipes.text.prepare_data --help
```

`PackedTokens` reads the prepared token directory. The recipe packs documents,
chooses microbatch candidates within the token bounds, and supplies summed
losses weighted by the inverse number of valid targets in the entire update.
Packing statistics and token throughput are logged by the recipe's callback.
The generic trainer never infers tokens or units from an input tensor.

## Precision and initialization

Model construction sets compute dtypes. `master_dtype` and `grad_dtype` are
independent trainer arguments; optimizer state dtype and rounding are arguments
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

When Trainer constructs MLOps AdamW, it selects stochastic rounding for BF16
moments by default. FP16 and FP32 moments and parameter rounding retain the
optimizer's nearest-rounding default. Set
`optimizer_args.opt_state_rounding=nearest` to override this, or supply rounding
in an optimizer partial or parameter group. MLOps used directly and
`torch.optim.AdamW` keep their own defaults. Numerical qualification explicitly
uses nearest rounding so existing references remain valid.

Initialized models retain their values. Meta models need an explicit initializer
or checkpoint; this recipe supplies `reset_parameters` with its configured seed.
An initializer can rebuild nonpersistent buffers before saved state is restored.
The generic APIs never silently reset an initialized model.

## Run policy and results

`steps` is the target total completed update count. Schedules are mappings such
as `schedules.lr`, evaluated with the zero-based update index. A warmup/cosine
schedule has an explicit `total_steps`; changing the run length does not silently
change the schedule.

`eval_every`, `checkpoint_every`, `log_every`, `metric_tables_every` and
`keep_last` control the optional loop behavior. Checkpoints go under
`checkpoint_dir`, or `<run_dir>/checkpoints`. `checkpoint_weights=master` is the
default: save masters where available, otherwise compute weights. Set
`checkpoint_weights=compute` to save compute weights alone; restore upcasts them
when masters are configured, without recovering the original higher precision.
Optimizer state is saved with either choice. The directory contains:

| File | Meaning |
| --- | --- |
| `config.json` | Configuration including explicit overrides |
| `planning.json` | Selected text microbatch geometry and skipped candidates |
| `search.json`, `plan.json` | Search and admitted plan, when applicable |
| `metrics.jsonl` | Completed `step`, `train/loss`, `train/step_seconds`, active elapsed time, schedules, evaluation and recipe metrics |
| `observations.jsonl` | Optional small observation tables |
| `packing.jsonl` | Document identities and packing statistics |
| `checkpoints/step_XXXXXXXX/` | Atomic model/optimizer/loop/source checkpoints |

`wandb_project` enables the optional logger. `step` is the actual completed
training step; evaluation, tables and training records use the same axis.
`wandb` passes additional SDK options. System telemetry keeps the SDK defaults.
Tensor summaries leave compiled tasks as tensors. Host collection and `.item()`
run after the step completes. Norm observers run before optimizer updates and
can report compute-weight and final-gradient statistics.

Resume is explicit:

```bash
python -m training.train training/configs/olmoe_1b.json \
  run_dir=training/runs/olmoe_example \
  resume=training/runs/olmoe_example/checkpoints/step_00000500
```

The checkpoint restores model and optimizer state, update count, RNG,
stateful schedules and the packed source's progress. The saved microbatch
candidate is pinned before planning. An ordinary iterable is also valid, but
exact data continuation requires caller-managed progress or the optional
`state_dict`/`load_state_dict` source methods.

## Layout and comparison

- `src/shadowspill/training/`: reusable trainer, forward runner, backends,
  schedules, checkpoint and logging helpers.
- `workloads/recipes/text/`: optional model construction, token ingestion,
  packing, objectives and text experiment composition.
- `training/configs/`: example recipe requests.
- `training/train.py`: recipe CLI.
- `training/scripts/`: launch and comparison helpers.

```bash
python -m training.compare <reference run dir> <run dir> [<run dir> ...]
```

The comparison reads actual step numbers, losses and document records from the
saved runs. Model examples and qualification policies remain outside the
installed generic trainer.

## LoRA examples

The trainer uses ordinary `requires_grad` and has no LoRA-specific execution
mode. Apply [`workloads.lora.configure_lora`](../workloads/MODELS.md#trainable-parameters-and-lora)
before creating the trainer. The `build_on_meta` text recipe accepts a `lora`
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
