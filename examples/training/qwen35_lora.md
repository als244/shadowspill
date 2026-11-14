# Qwen3.5 MoE LoRA through saved phases

The [Python recipe](../qwen35_lora.py) defaults to 32K-token
sequences, eight sequences per update, rank/alpha 32, BF16 compute weights and
FP32 adapter masters, gradients and Adam moments. It loads a local Hugging Face
base and uses prepared `train.bin`, `val.bin` and `meta.json` token files.

```python
from shadowspill_workloads.recipes import LoRA, Model, train

model = Model.from_checkpoint("/checkpoints/Qwen3.5-35B-A3B", lora=LoRA())
train(model, {
    "data": "/data/tokenized-corpus",
    "frozen_state_id": "qwen-base-revision-and-config",
    "run_dir": "runs/qwen35-lora",
    "microbatch_sizes": [1, 2, 4, 8],
})
```

Each phase runs in a fresh process. Build constructs one training Program and
one evaluation Program for every explicitly requested microbatch size, crossed
with any chunk and retention candidates. Each size must divide the sequence
count. See [search controls](../../packages/workloads/src/shadowspill_workloads/recipes/configuration.md#search-and-saved-phases).
Profiling uses
the ordinary PyTorch allocator. CPU planning compares batch sizes and depth /
breadth orderings, including joint memory admission for training and evaluation.
Execution loads only the chosen Plans and initializes Executor-provided buffers.

The default execution allowance is 28 GiB, including overhead; spill capacity is
144 GiB. These are recipe settings, not a guarantee that every checkpoint or
sequence geometry fits. Use `--execution-gib` and `--spill-gib` to change them.

Phases can also run separately. Later phases read the saved configuration:

```python
# Start with the configuration above and phase="build", then:
saved = {"run_dir": "runs/qwen35-lora"}
train(model, saved, phase="profile")
train(model, {**saved, "execution_gib": 40}, phase="plan")
train(model, saved, phase="execute")
```

| Output | Contents |
| --- | --- |
| `programs/`, `profiles/`, `profiled/`, `communication/` | Separate archives for each batch size and training/evaluation |
| `search.json`, `selected.json` | Candidate outcomes and chosen geometry/ordering |
| `plans/` | Chosen Plans and each successful candidate pair |
| `reports/` | Selected simulated timelines and memory reports |
| `logs/` | Build, profile, plan and execute process logs |
| `training/` | Training/evaluation JSONL and periodic checkpoint archives |

Validation uses two held-out updates every two training updates and at the final
update. Each evaluation starts from the same held-out windows. Training preserves
the resumable corpus position, warmup/cosine schedule and optional `--wandb-project`.
The default run has five updates; use `--steps` for a longer run.

Checkpoint archives omit unchanged frozen parameters and reference the supplied
base identity. They retain buffers, trainable compute weights, FP32 masters,
optimizer tensors, data position and RNG state. Resume supplies the same base
through the bounded loader; no full-model snapshot is created:

```python
train(model, {
    **saved,
    "resume": "runs/qwen35-lora/training/checkpoints/step_00000002.ssp",
}, phase="execute")
```

The loader preserves checkpoint values, copies transformed projections directly
into requested chunks, and initializes only fresh adapters and derived RoPE
tables. Adapter initialization uses fixed-size RNG blocks, so chunk size and
callback order do not change the values; its distributions match the modules,
but its RNG stream differs from whole-module `reset_parameters`.
