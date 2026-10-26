# ShadowSpill

**Refactor branch:** the public API now separates build, measurement, CPU planning
and saved-plan execution. Start with the [phase API guide](docs/architecture/offline-pipeline.md)
or [quickstart](benchmarking/quickstart.md). `Trainer(plan, ...)` consumes an already
prepared Plan. The Llama and Qwen recipes below use these phases. The practical
GLM checkpoint-training recipe still awaits migration.

ShadowSpill turns a fixed-shape PyTorch forward or training step into a
memory-budgeted callable: it decides what to keep on the device, what to spill
and fetch back, and what to recompute, then proves the answer fits the declared
pools before the step runs. PyTorch still executes every kernel.

## Installation

```bash
./scripts/setup.sh                                    # fresh checkout
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python" # existing environment
```

The script creates `.venv`, installs the supported PyTorch and device-backend
stack, builds the C library with its backends and the PyTorch adapter, installs
the mlops operation library and the training harness's dependencies, and
verifies the install. Add `--expert-parallel quack`, `te` or `both` for
mlops's optional expert-parallel backends, which the expert-parallel workloads
use; that needs the mlops checkout beside this one.

## Model recipes

The Llama example separates build, profiling, CPU planning, and saved-plan
training into independent processes. `Trainer(plan, ...)` runs the resulting
plan and shares its current weights with a separately built evaluation plan.
The Qwen recipes use the same phases. The practical GLM recipe still uses the
private preparation implementation and is marked below.

| Example | Sequence length | Sequences per update | Execution | Microbatch selection |
|---|---:|---:|---|---|
| Llama 3 8B full training | 4,096 | 16 | 1 GPU | Fixed: four sequences, 1×4 ordering |
| GLM-5.3-Flash LoRA | 32,768 | 8 globally; 1 per GPU | Eight GPUs | Fixed: one sequence per rank |
| Qwen3.5-35B-A3B LoRA | 32,768 | 8 | 1 GPU | Search sizes and orderings |
| Qwen3-30B-A3B full training | 4,096 | 512 globally; 64 per GPU | Eight GPUs | Search sizes and orderings |

Run these scripts from the checkout after installation. Replace each example
path with a corpus tokenized using the matching model's tokenizer. `train.bin` and
`val.bin` contain
uint32 token IDs and `meta.json` records `vocab_size`. The
[text data preparation recipe](workloads/recipes/text/data/prepare.py) writes
this format. Supply at least five complete updates of data.

The small [TokenUpdates helper](workloads/recipes/text/data/fixed.py) reads text
windows, shifts targets by one token, and supplies every microbatch size that
divides the local sequence count. It gives each microbatch independent storage
and normalizes summed losses by the **global** token count. The Qwen examples
build one Program per example size and search orderings during planning; Llama
fixes both. The total update size stays fixed in every example.

These examples use FP32 masters/gradients/AdamW moments and a warmup-cosine
learning-rate schedule. GLM preserves the checkpoint's mixed frozen storage
precision, including FP8 routed experts; the other examples use BF16 compute
weights. Models are declared on `meta`;
initialization writes directly into pool-owned storage. The single-GPU examples request 28 GiB of GPU memory
and a 144 GiB pinned host pool. Adjust budgets for your machine; the planner
checks feasibility before training. These are example configurations, not
published performance or memory-fit results.

<details>
<summary><strong>Llama 3 8B: full training</strong></summary>

Start from random initialization and train all parameters on 4K-token sequences,
16 sequences per optimizer update. Each microbatch has four sequences (16K
tokens), so there are four microbatches per update. The fixed 1×4 ordering visits
all four at each layer before moving to the next layer. ShadowSpill still chooses
storage, transfers and recomputation within the budget. The
[complete script](workloads/recipes/text/train_llama3.py) is runnable with:

```bash
python -m workloads.recipes.text.train_llama3 --data /path/to/llama3-tokens
```

Its [build](workloads/recipes/text/llama3/build.py),
[profile](workloads/recipes/text/llama3/profile.py),
[plan](workloads/recipes/text/llama3/plan.py), and
[execute](workloads/recipes/text/llama3/execute.py) files call the public APIs
directly. No model is constructed during execution. The bounded initializer
writes weights and rotary tables into Executor-provided storage; evaluation
uses `build_forward(..., forward_fn=loss)` with the same model tensor names.
Programs, profiles, plans, reports, logs and checkpoints have separate folders
under `--run-dir`.

Each phase can also run separately. Later phases read the saved configuration:

```bash
python -m workloads.recipes.text.train_llama3 --data /path/to/llama3-tokens \
  --run-dir runs/llama3-8b --phase build
python -m workloads.recipes.text.train_llama3 --run-dir runs/llama3-8b --phase profile
# Replan on CPU with different memory budgets, without rebuilding or profiling:
python -m workloads.recipes.text.train_llama3 --run-dir runs/llama3-8b \
  --phase plan --execution-gib 28 --spill-gib 144
python -m workloads.recipes.text.train_llama3 --run-dir runs/llama3-8b --phase execute
```

Use `--resume /path/to/checkpoint.ssp` with the execute phase to restore model,
optimizer, schedule and data progress. `--model llama3-tiny` and smaller sequence
counts are available for pipeline checks; the 8B configuration is not a claimed
performance or memory-fit result on this refactor branch.

</details>

<details>
<summary><strong>GLM-5.3-Flash: eight-GPU LoRA at 32K</strong></summary>

**Migration status:** this practical recipe still uses the private preparation
path. The family quickstart uses the shared phases; migrating this training loop
and its distributed checkpoint/evaluation behavior remains work on this branch.

The [training recipe](workloads/recipes/text/train_glm53_flash_lora.py) loads a
local Hugging Face checkpoint and trains rank-32 LoRA adapters on real text with
held-out validation. The [eight-GPU launcher](workloads/recipes/text/glm53_flash_lora_32k_8gpu.sh)
uses one 32,768-token sequence per rank: 262,144 target tokens per update.
Weights, gradients and optimizer state are sharded across all eight DP ranks.
Frozen routed experts retain FP8 weights and quantization scales and use the
SonicMoE FP8 LoRA path. Frozen BF16/FP32 tensors keep checkpoint storage precision;
compressed dense projections dequantize to BF16 for GEMMs. Adapters use BF16
compute with FP32 masters, gradients and AdamW moments.

Prepare text with this checkpoint's tokenizer. The preparation script records
hashes of `tokenizer.json` and `tokenizer_config.json`; training verifies both
before loading weights. For example, this dataset revision supplies real text
and the first whole-document batches are held out for validation:

```bash
HF_HOME=/path/to/large-cache python -m workloads.recipes.text.data.prepare \
  --tokenizer /path/to/GLM-5.3-Flash \
  --dataset HuggingFaceFW/fineweb-edu \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --files 'sample/10BT/000_00000.parquet' \
  --out /path/to/glm-tokens --val-tokens 1048576

bash workloads/recipes/text/glm53_flash_lora_32k_8gpu.sh \
  --data /path/to/glm-tokens --checkpoint /path/to/GLM-5.3-Flash \
  --frozen-state-id glm53-flash-exact-checkpoint-revision \
  --run-dir /path/to/glm32k-run
```

The launcher defaults to 100 updates, with evaluation and adapter checkpoints
every 10 updates. Supply at least 100 complete global training updates and one
complete validation update. `--steps`, `--schedule-steps`, `--execution-gib` and
`--spill-gib` can be overridden on the command line. Keep the checkpoint on
storage with room for the full base; initialization writes into the sharded
spill pools with bounded staging. The starting budgets are 76 GiB execution and
96 GiB pinned host spill per rank (768 GiB pinned host memory across the node,
plus process and staging overhead). The planner also reserves 2 GiB external GPU
headroom. This configuration is being qualified on eight 80 GB H100s; it is not
yet a completed 32K checkpoint-training performance result.

`metrics.jsonl` records train/validation loss and timing, and the run directory
also contains the configuration, selected plan and partial checkpoints.
Supply an immutable `--frozen-state-id` for the exact base checkpoint. Resume by
adding `--resume /path/to/glm32k-run/checkpoints/step_00000010`, keeping the same
corpus, base ID and total schedule; `--steps` is the final total step count.
Frozen base weights are reloaded from the original checkpoint on resume.

For a learning sanity check, add `--repeat-training-batch
--minimum-relative-loss-drop 0.01`; normal training advances through the corpus.
For five-step Nsight capture, add `--nsys-start-step 2 --nsys-steps 5` and launch
under `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.
Inductor/Triton compiler intermediates use node-local temporary storage by
default and are separate from the saved run artifacts.

</details>

<details>
<summary><strong>Qwen3.5-35B-A3B: LoRA at 32K</strong></summary>

Start from a local Hugging Face checkpoint and train rank-32 adapters on the
attention projections and routed experts. The base, embeddings, norms and output
head stay frozen. Each update contains eight 32K-token sequences. The checkpoint
initializer loads base weights, rebuilds rotary buffers and initializes the
adapters directly in the training pools. The
[complete script](workloads/recipes/text/train_qwen35_lora.py) runs with:

```bash
python -m workloads.recipes.text.train_qwen35_lora \
  --data /path/to/qwen35-tokens --checkpoint /path/to/Qwen3.5-35B-A3B \
  --frozen-state-id qwen35-base-v1
```

The corpus also needs `val.bin`, containing at least two complete validation
updates: 16 sequences, or 524,288 target tokens. Evaluation reuses the same
held-out windows every two updates and at the final update. Training and
evaluation use separate saved Plans sharing current model tensors.

The recipe's [phase modules](workloads/recipes/text/qwen35_lora/) build each
candidate, measure it, search batch sizes/orderings on CPU, and execute in a
fresh worker. Saved settings let phases run independently:

```bash
python -m workloads.recipes.text.train_qwen35_lora \
  --data /path/to/qwen35-tokens --checkpoint /path/to/Qwen3.5-35B-A3B \
  --frozen-state-id qwen35-base-v1 --run-dir runs/qwen35-lora --phase build
python -m workloads.recipes.text.train_qwen35_lora \
  --run-dir runs/qwen35-lora --phase profile
CUDA_VISIBLE_DEVICES= python -m workloads.recipes.text.train_qwen35_lora \
  --run-dir runs/qwen35-lora --phase plan --execution-gib 28 --spill-gib 144
python -m workloads.recipes.text.train_qwen35_lora \
  --run-dir runs/qwen35-lora --phase execute
```

`programs/`, `profiles/` and `profiled/` retain reusable phase artifacts.
`search.json` records the candidates; `selected.json`, `plans/` and `reports/`
hold the selection and simulated timelines. `training/` contains metrics,
evaluation logs and checkpoints. Add `--wandb-project my-project` to the initial
build command for optional W&B logging.

Supply an immutable `--frozen-state-id` for the exact base and configuration.
Partial checkpoints save trainable tensors, optimizer data and progress while
omitting immutable base weights. The bounded loader reloads the base from HF
on resume. For example:

```bash
python -m workloads.recipes.text.train_qwen35_lora \
  --run-dir runs/qwen35-lora --phase execute \
  --resume runs/qwen35-lora/training/checkpoints/step_00000004.ssp
```

The configured `steps=5` is the total target, so this performs the remaining
update. Keep the same corpus, base and schedule. A tiny checkpoint fixture
qualifies exact cold resume on this branch; the full 35B/32K configuration is
not a claimed performance or memory-fit result.

</details>

<details>
<summary><strong>Qwen3-30B-A3B: full training on eight GPUs</strong></summary>

This recipe starts from random initialization and consumes 512 sequences per
update: 64 distinct 4K-token sequences per rank. `Parallelism.dp(ranks_per_replica=8)`
shards weights, gradients, FP32 masters and AdamW moments across the eight GPUs.
The compute weights are BF16 and accumulated gradients are FP32.

The [recipe](workloads/recipes/text/qwen3/README.md) builds and profiles local
compute on a representative GPU, measures communication on all ranks, searches sizes/orderings
on the CPU, and launches fresh Trainer workers with the selected saved Plan.
Its defaults request 77 GiB of total GPU memory and 96 GiB of pinned host spill
per rank. Batch candidates are 1, 2, 4, 8, 16 and 32 sequences per rank.

Run the orchestrator once; it launches torchrun only for distributed phases:

```bash
python -m workloads.recipes.text.train_qwen3_distributed \
  --data /path/to/qwen3-tokens --run-dir runs/qwen3-dp8
```

Each phase is also independently runnable. For example, reuse measurements to
search another budget with GPUs hidden:

```bash
CUDA_VISIBLE_DEVICES= python -m workloads.recipes.text.train_qwen3_distributed \
  --run-dir runs/qwen3-dp8 --phase plan --execution-gib 60
```

Initialization fills bounded Executor-provided CPU chunks. Per-rank losses use
the global target-token count, because gradient contributions are summed.
`search.json` records every candidate, `plan.ssp` holds the selected executable,
`report/` contains simulated timelines, and `training/rank-NNNNN/` holds metrics
and checkpoints. No model construction, profiling or planning occurs in the
execution worker. See [distributed ownership](docs/python/api/distributed.md)
for the general parallelism API.

</details>

## More control

`Program`, `Profiles`, `ProfiledProgram` and `Plan` each support explicit save/load.
Reuse a saved `ProfiledProgram` with `plan_program(...)` for different memory
budgets or communication assumptions without rebuilding or measuring again.
`plan.summary`, `plan.timeline` and `plan.memory_layout` expose the result;
`write_report(plan, path)` writes CPU-only diagnostics. Trainer consumes the
selected Plan; direct Executor use gives callers control over individual invocations.

The [training API](docs/python/api/training.md) covers schedules, evaluation and
checkpointing. [Reusable planning](docs/examples/reusable-planning.md) shows the
phase artifacts, and the [low-level training loop](docs/examples/training-lifecycle.md)
shows explicit Executor ownership. [Examples](docs/examples/README.md) includes
small models for learning the API. The [benchmark quickstart](benchmarking/quickstart.md)
measures experiments and plots plans. [SSD spill pools](docs/python/api/ssd.md)
provide an alternative to pinned host storage for single-rank training.

## Project structure

| Path | Purpose |
|---|---|
| `src/shadowspill/` | Installed Python package and PyTorch frontend |
| `csrc/` | The C library — planner, simulator, runtime — plus backends and the PyTorch adapter |
| `tests/` | Tests mirroring Python, C, integration, and tooling boundaries |
| `workloads/` | Optional models and recipes passed into generic APIs |
| `benchmarking/` | The quickstart tour, program collection, and planning evaluation |
| `qualification/` | Release gates: suite, numerical, performance, and remote |
| `training/` | Example text configs, launch scripts and run comparisons |
| `src/tools/` | Repository naming checks and sanitizer support |
| `reference/` | Readable reference implementations of the planner |
| `scripts/` | One-command environment setup |
| `docs/` | Architecture, Python, C, examples, and development guides |

## Documentation

[**docs/README.md**](docs/README.md) explains what the system does and links
every page once. The shortest routes from here:

| Topic | Start here |
|---|---|
| System architecture | [Architecture overview](docs/architecture/overview.md) |
| Python usage and API | [Python documentation](docs/python/README.md) |
| C components and APIs | [C documentation](docs/c/README.md) |
| Plan search | [Plan search](docs/architecture/search.md), [PressureFit](docs/architecture/pressurefit.md) |
| Physical admission | [Physical admission and offset handling](docs/architecture/physical-admission.md) |
| Plan and step diagnostics | [Diagnostics guides](docs/python/plan-report.md) |
| Serialized planning artifacts | [program and annotated-plan JSON](docs/python/planning-json.md) |
| Example architectures and configuration | [Model catalog](workloads/MODELS.md) |
| Practical workflows | [Examples](docs/examples/README.md) |
| Errors and cleanup | [Errors, failures, and cleanup](docs/python/failures.md) |
| Repository development | [Development guide](docs/development/README.md) |
