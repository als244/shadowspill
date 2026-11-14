# Training from Python

Use `train(model, config)` for the supported models below. It builds and profiles
requested candidates, searches within the memory budgets, then trains with the
selected Plan. You do not need to construct a Plan or a Trainer.

```python
from shadowspill_workloads.recipes import Model, train

model = Model.from_config("llama3-1b")  # Full training from random weights.
train(model, {
    "run_dir": "runs/llama-pretrain",   # Fresh directory; saves every phase.
    "data": "/data/fineweb-llama",      # Prepared with this model's tokenizer.
    "sequence_length": 2048,
    "sequences_per_step": 4,           # Per rank, per optimizer update.
    "microbatch_sizes": [1, 2, 4],     # Build/profile each, then choose a Plan.
    "execution_gib": 24,               # Total GPU budget per rank.
    "spill_gib": 64,                   # Maximum pinned host pool per rank.
    "steps": 100,
    "eval_every": 20,
    "checkpoint_every": 20,
})
```

Run the script once from a [configured checkout](../../../../../docs/development/repository.md#setup),
without an outer `torchrun`. Imports and model descriptions are CPU-only;
`train` launches fresh workers and returns the run directory after completion.
Choose budgets that fit your hardware and leave host RAM for the process and I/O.
The actual spill reservation is the selected Plan's requirement plus 5%, capped
by `spill_gib`.

## Supported model sources

| Source | Supported families | Training method |
| --- | --- | --- |
| `Model.from_config(name)` | Llama 3, OLMoE, Qwen3, Qwen3 MoE, Qwen3.5, Qwen3.5 MoE | Full training from random weights; BF16 compute, FP32 gradients, masters and AdamW moments |
| `Model.from_checkpoint(path, lora=LoRA(...))` | Qwen3.5 MoE and GLM-5.3-Flash | LoRA from a local checkpoint; rank/alpha default to 32 |

Catalog examples include `llama3-1b`, `olmoe-7b-a1b`, `qwen3-1.7b`,
`qwen3moe-30b-a3b`, `qwen35-2b` and `qwen35moe-35b-a3b`.
See the [model catalog](../MODELS.md) for sizes and implementations. Catalog
support does not imply that every size fits your hardware. Unsupported
checkpoint families or training methods fail before launching workers.

For checkpoint fine-tuning, replace the model source and supply a stable
identity for the exact frozen base:

```python
from shadowspill_workloads.recipes import LoRA, Model

model = Model.from_checkpoint("/checkpoints/Qwen3.5-35B-A3B", lora=LoRA())
# In the training config:
# "frozen_state_id": "exact-base-checkpoint-revision"
```

Commented scripts: [catalog full training](../../../../../examples/train_text.py),
[Qwen3 distributed full training](../../../../../examples/qwen3.py),
[Qwen3.5 LoRA](../../../../../examples/qwen35_lora.py),
[GLM Flash LoRA](../../../../../examples/glm53_flash_lora.py).
The checkpoint recipes document their storage/precision requirements.
For a custom model or loss, follow the [custom-model guide](../../../../../docs/python/custom-models.md).

## Data, defaults and search

Prepare `train.bin`, `val.bin` and `meta.json` with the model's tokenizer using
the [data guide](../inputs/text/README.md). The text runner shifts targets and
normalizes next-token loss over the global update. `sequences_per_step` is
**per rank**; global tokens/update are `ranks * sequences_per_step * sequence_length`.
Every `microbatch_sizes` entry must divide `sequences_per_step`.

| Setting | Catalog full-training default | What to change |
| --- | --- | --- |
| `ranks`, `parallelism` | `1`, `"dp"` | Number of GPUs; `"ep"` for supported MoE families |
| `sequence_length`, `sequences_per_step` | `2048`, `1` | Context and update size |
| `microbatch_sizes` | `[1]` | Exact sizes to build/profile, e.g. `[1, 2, 4]` |
| `chunk_sizes`, `moe_chunk_sizes` | `[0]`, `[0]` | Projection/expert chunk candidates; zero means unchunked/inherit |
| `graph_pair_policies` | `["cost-input"]` | SAVE/RECOMPUTE retention candidates |
| `execution_gib`, `spill_gib` | `28`, `144` | Per-rank GPU and pinned-host budgets |
| `external_headroom_mib` | `768` | GPU allowance outside the managed pool |
| `steps`, `schedule_steps` | `5`, initial `steps` (at least 2) | Total updates and fixed learning-rate horizon |
| `learning_rate`, `seed` | `2e-4`, `0` | Peak learning rate and bounded random initialization |
| `eval_every`, `checkpoint_every`, `keep_last` | `2`, `2`, `2` | Cadences (zero disables) and retained checkpoints |

Checkpoint recipes have their own geometry and resource defaults. For comparable
experiments, specify the settings you want in the script. The saved `config.json`
records every resolved choice. Unknown options and invalid geometry fail before
GPU workers launch.

Search crosses the requested microbatch, chunk and retention choices, then tries
factor orderings and PressureFit save/recompute/placement choices. It builds only
the listed alternatives; it does not silently expand the candidate set. EP
currently requires unchunked execution. See [search controls](configuration.md#search-and-saved-phases)
for policy meanings, provider limits and [tail-save choices](configuration.md#resolution-and-tail-save-search).

## Reuse or resume

```python
saved = {"run_dir": "runs/llama-pretrain"}
train(model, {**saved, "execution_gib": 20}, phase="plan")  # CPU-only replan.
train(model, {**saved, "resume": "runs/llama-pretrain/training/checkpoints/step_00000020.ssp"},
      phase="execute")
```

`steps` counts total updates including restored ones. If you intend to train in
segments, set `schedule_steps` to the final horizon in the initial configuration;
continuations can increase `steps` within that horizon. Changing model, data
geometry or build candidates requires a fresh run. Saved phases are `build`,
`profile`, `communicate`, `plan` and `execute`; the default runs all required phases.

See the [configuration reference](configuration.md) for profiling controls,
planning concurrency, budget sweeps, lifecycle logs, Nsight and checkpoints.
The [Trainer reference](../../../../../docs/python/api/training.md) is for custom loops over
an already selected Plan; [phase APIs](../../../../../docs/python/phases.md) expose each
preparation step directly.
