# ShadowSpill

**Refactor branch:** the public API now separates build, measurement, CPU planning
and saved-plan execution. Start with the [phase API guide](docs/architecture/offline-pipeline.md)
or [quickstart](benchmarking/quickstart.md). `Trainer(plan, ...)` consumes an already
prepared Plan. The older model/prepare recipes below are explicitly routed through
a private migration implementation until their evaluation/distributed paths are migrated.

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

## Model recipes awaiting migration

These recipes currently use the private model/prepare Trainer. The new public
`Trainer(plan, ...)` is documented in the phase guide above. The migration
implementation takes a model, loss function and iterable of complete updates.
`prepare()` opens the backend and admits a plan that fits. Supply named
`microbatch_candidates` to search sizes and orderings, or fix both explicitly. `fit()` runs forward, backward
and AdamW updates with the learning-rate schedule; `close()` releases owned
resources. Use `step()` for a custom loop.

| Example | Sequence length | Sequences per update | Execution | Microbatch selection |
|---|---:|---:|---|---|
| Llama 3 8B full training | 4,096 | 16 | 1 GPU | Fixed: four sequences, 1×4 ordering |
| GLM-5.3-Flash LoRA | 32,768 | 8 globally; 1 per GPU | Eight GPUs | Fixed: one sequence per rank |
| Qwen3.5-35B-A3B LoRA | 32,768 | 8 | 1 GPU | Search sizes and orderings |
| Qwen3-30B-A3B full training | 4,096 | 512 globally; 64 per GPU | Eight GPUs | Search sizes and orderings |

Run these scripts from the checkout after installation. Replace each example
path with a corpus tokenized using the matching model's tokenizer. `train.bin` contains
uint32 token IDs and `meta.json` records `vocab_size`. The
[text data preparation recipe](workloads/recipes/text/prepare_data.py) writes
this format. Supply at least five complete updates of data.

The small [TokenUpdates helper](workloads/recipes/text/fixed.py) reads text
windows, shifts targets by one token, and supplies every microbatch size that
divides the local sequence count. It gives each microbatch independent storage
and normalizes summed losses by the **global** token count. The Qwen examples
search sizes and orderings with `microbatch_candidates=data.candidates`; Llama
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

```python
import torch
from mlops.optim import AdamW
from shadowspill.planner import StepDataOrdering
from shadowspill.training import reset_parameters
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import WarmupCosine
from workloads.models.catalog import build_on_meta
from workloads.recipes.text.fixed import TokenUpdates
from workloads.recipes.text.objectives import next_token_prediction

data_directory = "/path/to/llama3-tokens"
steps = 5
# One optimizer update contains 16 sequences, split into four microbatches.
data = TokenUpdates(data_directory, sequence_length=4096, sequences_per_step=16)
model = build_on_meta("llama3-8b", dtype="bfloat16")
trainer = Trainer(
    model,
    objective=next_token_prediction,
    optimizer=AdamW,
    optimizer_args={"opt_state_dtype": torch.float32},
    schedules={
        "lr": WarmupCosine(lr=3e-4, min_lr=3e-5, warmup_steps=1, total_steps=steps)
    },
    master_dtype=torch.float32,
    grad_dtype=torch.float32,
    microbatches=data.splitter(sequences_per_microbatch=4),
    backend=ShadowSpill(
        execution_gib=28,
        spill_gib=144,
        # One pass visits all four microbatches at each layer (breadth-first).
        orderings=[StepDataOrdering(depth=1, breadth=4)],
    ),
)
# Open pools and plan storage/recomputation with the fixed size and ordering.
trainer.prepare(data.example(), initialize=reset_parameters)
# Perform forward/backward/AdamW updates, advancing the LR schedule each step.
trainer.fit(data, steps=steps, run_dir="runs/llama3-8b")
trainer.close()  # Release the callable and the pools this trainer opened.
```

</details>

<details>
<summary><strong>GLM-5.3-Flash: eight-GPU LoRA at 32K</strong></summary>

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
HF_HOME=/path/to/large-cache python -m workloads.recipes.text.prepare_data \
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
items: 16 sequences, or 524,288 target tokens. Evaluation uses the same held-out windows every two updates and at the
final update. `metrics.jsonl` records training/evaluation loss, learning rate and
timing; `search.json` and `plan.json` record the selected plan. Add
`--wandb-project my-project` to also send metrics and system telemetry to W&B
(using your W&B login).

The Trainer's `frozen_state_id` enables partial checkpoints: this LoRA run saves
each adapter parameter once (its FP32 master), adapter optimizer state, and training progress,
including RNG and data position. Frozen base weights and derived rotary tables
are omitted. The generic mechanism also retains persistent buffers or frozen
state that a training graph mutates. Supply an immutable ID for your exact base
checkpoint and configuration; resume checks that ID and reloads the base from HF.

Checkpoint precision and optimizer state are independent options:
`trainer.save(path, weights="compute", include_optimizer=False)` saves only
compute-precision parameters, plus buffers and progress. A new Trainer can
resume that progress with a fresh optimizer. Periodic saves expose the same
options as `checkpoint_weights` and `checkpoint_include_optimizer` in `fit`.

The script keeps the latest two checkpoints under `run_dir/checkpoints`. Resume
from a saved step, optionally using a new output directory:

```bash
python -m workloads.recipes.text.train_qwen35_lora \
  --data /path/to/qwen35-tokens --checkpoint /path/to/Qwen3.5-35B-A3B \
  --frozen-state-id qwen35-base-v1 \
  --resume runs/qwen35-lora/checkpoints/step_00000004 \
  --run-dir runs/qwen35-resumed
```

`steps=5` is the target total, so that command performs the remaining update.
Keep the same corpus, frozen base and schedule when resuming. W&B logging is
optional and starts a new run for each invocation; the
[logger API](docs/python/api/training.md) also accepts W&B run IDs and resume
options when you want one continuous history.

```python
import torch
from mlops.optim import AdamW
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.logging import Wandb
from shadowspill.training.schedules import WarmupCosine
from workloads.models.construction import build_on_meta
from workloads.models.qwen35moe.checkpoint import Qwen35MoECheckpoint
from workloads.models.qwen35moe.implementations.mlops import Model
from workloads.recipes.text.fixed import TokenUpdates
from workloads.recipes.text.objectives import next_token_prediction

data_directory = "/path/to/qwen35-tokens"
checkpoint_directory = "/path/to/Qwen3.5-35B-A3B"
frozen_state_id = "qwen35-base-v1"  # An immutable ID for your base and configuration.
run_dir = "runs/qwen35-lora"
resume = None  # Or a Trainer checkpoint directory from an earlier run.
wandb_project = None  # Set a project name to enable W&B.
steps = 5
# Read next-token examples and supply candidate microbatch sizes.
data = TokenUpdates(data_directory, sequence_length=32768, sequences_per_step=8)

# Each validation item contains eight 32K sequences, just like a training item.
# A fresh iterator evaluates the same held-out windows each time.
def validation_data():
    return TokenUpdates(
        data_directory, split="val", sequence_length=32768, sequences_per_step=8
    )

checkpoint = Qwen35MoECheckpoint(checkpoint_directory)
model = build_on_meta(
    Model,
    config=checkpoint.config,
    dtype="bfloat16",
    lora={"rank": 32, "alpha": 32},
)
trainer = Trainer(
    model,
    objective=next_token_prediction,
    optimizer=AdamW,
    optimizer_args={"opt_state_dtype": torch.float32},
    schedules={
        "lr": WarmupCosine(lr=2e-4, min_lr=2e-5, warmup_steps=1, total_steps=steps)
    },
    master_dtype=torch.float32,
    grad_dtype=torch.float32,
    microbatch_candidates=data.candidates,
    backend=ShadowSpill(execution_gib=28, spill_gib=144),
    # Save trainable state only, referencing this immutable frozen base ID.
    frozen_state_id=frozen_state_id,
)
# Load the frozen base; on resume, restore saved adapters/optimizer/progress.
trainer.prepare(data.example(), initialize=checkpoint.initialize, checkpoint=resume)
# Local JSONL logging is always enabled; W&B is optional.
logger = None
if wandb_project is not None:
    logger = Wandb(
        project=wandb_project,
        run_dir=run_dir,
        config={
            "model": "Qwen3.5-35B-A3B",
            "lora_rank": 32,
            "sequence_length": 32768,
            "sequences_per_step": 8,
        },
    )
trainer.fit(
    data,
    steps=steps,  # Target total updates, including any restored progress.
    run_dir=run_dir,
    log_every=1,  # Loss, LR and step timing go to metrics.jsonl and the logger.
    logger=logger,
    eval_data=validation_data,
    eval_every=2,  # Every two training updates, and also at the final update.
    eval_updates=2,  # Two validation updates: 16 sequences, 524,288 target tokens.
    checkpoint_every=2,  # Adapter masters once, optimizer, RNG and data position.
    keep_last=2,  # Retain the two newest checkpoints under run_dir/checkpoints.
)
trainer.close()  # Release the callable and its memory pools.
if logger is not None:
    logger.close()  # Finish and flush this W&B run.
```

</details>

<details>
<summary><strong>Qwen3-30B-A3B: full training on eight GPUs</strong></summary>

This example starts from random initialization and consumes 512 sequences per
update: 64 distinct 4K-token sequences on each rank. Both kinds of sharding are
explicit: `DataParallel(shard_weights=True, shard_gradients=True)` shards compute
weights and gradients in the spill pool; `shard_optimizer=True` independently shards FP32 masters and AdamW moments
across all eight ranks. Weight fetches all-gather the layer's execution tensors;
gradient evictions reduce-scatter their contributions. The plan decides when
those transfers occur and which tensors remain resident between uses.

`prepare()` creates Gloo for coordination before opening the pools, then NCCL
for tensor collectives. `close()` releases the pools before those groups.
`DataParallel()` also supports ordinary replicated compute weights and gradient
reductions; optimizer/master sharding remains enabled by default in either mode.
`device="auto"` selects the rank's GPU. The
[complete script](workloads/recipes/text/train_qwen3_distributed.py) requests
77 GiB of GPU memory and 96 GiB of pinned host memory **per rank** (768 GiB of
host pools across the node), suitable as a starting configuration for an
8×80-GB H100 node with sufficient host RAM. Microbatch candidates range from
one to 32 sequences per rank (4K–128K tokens):

```bash
torchrun --standalone --nproc-per-node=8 \
  -m workloads.recipes.text.train_qwen3_distributed --data /path/to/qwen3-tokens
```

```python
import torch
from mlops.optim import AdamW
from shadowspill.training import DataParallel, reset_parameters
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import WarmupCosine
from workloads.models.catalog import build_on_meta
from workloads.recipes.text.fixed import TokenUpdates
from workloads.recipes.text.objectives import next_token_prediction

data_directory = "/path/to/qwen3-tokens"
steps = 5
# Trainer creates Gloo for coordination, then NCCL for tensor collectives.
# Weight fetches all-gather; gradient evictions reduce-scatter.
parallel = DataParallel(shard_weights=True, shard_gradients=True, timeout=7200)
if parallel.world_size != 8:
    raise ValueError("launch this script with eight torchrun processes")
# Each rank reads 64 distinct sequences; losses use global token normalization.
# Search sizes from 1 to 32 sequences per microbatch, and all factor orderings.
data = TokenUpdates(
    data_directory,
    sequence_length=4096,
    sequences_per_step=64,
    rank=parallel.rank,
    world_size=parallel.world_size,
    max_sequences_per_microbatch=32,
)
model = build_on_meta("qwen3moe-30b-a3b", dtype="bfloat16")
trainer = Trainer(
    model,
    objective=next_token_prediction,
    optimizer=AdamW,
    optimizer_args={"opt_state_dtype": torch.float32},
    schedules={
        "lr": WarmupCosine(lr=3e-4, min_lr=3e-5, warmup_steps=1, total_steps=steps)
    },
    master_dtype=torch.float32,
    grad_dtype=torch.float32,
    microbatch_candidates=data.candidates,
    backend=ShadowSpill(
        device="auto", execution_gib=77, spill_gib=96, preparation_timeout=7200
    ),
    distributed=parallel,
    shard_optimizer=True,  # Separately shard FP32 masters and AdamW moments.
)
torch.manual_seed(0)
# Allocate sharded storage, initialize randomly, search sizes/orderings, admit.
trainer.prepare(data.example(), initialize=reset_parameters)
trainer.fit(data, steps=steps, run_dir="runs/qwen3-dp8")
trainer.close()  # Release callables, pools, then the owned process groups.
```

Each rank reads different corpus windows. Its losses are normalized by all
512 × 4,096 target tokens, because distributed gradients are summed. Search
coordinates a compatible collective schedule across ranks. The Trainer writes
per-rank metrics and plans under the run directory. See
[distributed ownership](docs/python/api/distributed.md) for subgroup layouts.

</details>

## More control

`trainer.planning` exposes the microbatch search and each budget's selected plan;
`plan.summary` reports estimated step time, memory peaks, recomputation, idle
time and transfer utilization. For repeated budgets, use
`trainer.prepare_artifacts(..., budgets=...)`, save its `PreparedTraining`, and
pass it to another trainer's `prepare(..., prepared=...)`. Persistent preparation
currently supports one rank; distributed training uses coordinated preparation.

The [training API](docs/python/api/training.md) covers schedules, evaluation and
checkpointing. [Reusable planning](docs/examples/reusable-planning.md) shows
prepare/search/admit separately, and the [low-level training loop](docs/examples/training-lifecycle.md)
shows explicit runtime ownership. [Examples](docs/examples/README.md) includes
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
