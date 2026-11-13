# Language-model training with the phase APIs

This example trains the repository's Llama-3.2-1B decoder from random weights on
next-token prediction. It uses real tokenized text, a summed cross-entropy loss
normalized over the update, and AdamW with FP32 gradients/master weights.
For an end-to-end call that also searches candidates, use the
[commented Python runner](../../workloads/recipes/text/example.py). The separate
phases below show where a custom model, objective and data source enter.

Prepare a corpus with the model's tokenizer using the [data guide](../../workloads/recipes/text/data/README.md).
It supplies `train.bin`, `val.bin` and `meta.json`. Use paths and memory budgets
appropriate to your machine. A smaller catalog preset is useful for pipeline
checks, but needs a corpus whose token IDs fit its vocabulary.

## Build and profile

Run this in a preparation process with a representative GPU:

```python
import json
from pathlib import Path

import torch
from mlops.dispatch import set_weight_gradient_dtype
from mlops.optim import AdamW
from shadowspill import build_step, measure_program, apply_profiles, measure_communication
from workloads.models.catalog import build_on_meta
from workloads.recipes.text.data.calls import Inputs

settings = {
    "model": "llama3-1b",
    "data": "/data/fineweb-llama",
    "sequence_length": 2048,
    "sequences_per_step": 4,
    "sequences_per_example": 1,  # Four complete microbatch calls per update.
}
source = Inputs(settings)
example = source.example()       # Does not consume the first training update.
model = build_on_meta(
    settings["model"], dtype="bfloat16", max_seq_len=settings["sequence_length"],
)
set_weight_gradient_dtype(torch.float32)
optimizer = AdamW(
    model.parameters(), lr=torch.tensor(3e-4),
    gradient_dtype=torch.float32, opt_state_dtype=torch.float32,
)

def objective(model, tokens, targets, *, seq_lens, normalizer):
    # targets are the next tokens; seq_lens keeps independent contexts separate.
    # Here normalizer = 1 / (4 * 2048), not a separate mean per microbatch.
    return model.loss(tokens, targets, seq_lens=seq_lens, reduction="sum") * normalizer

program = build_step(
    model, objective=objective, optimizer=optimizer,
    example_args=example["args"][0], example_kwargs=example["kwargs"][0],
    num_examples=len(example["args"]),
    grad_dtype=torch.float32, master_dtype=torch.float32,
)
program.save("program.ssp")
profiles = measure_program(program)
profiles.save("profiles.ssp")
apply_profiles(program, profiles).save("profiled.ssp")
measure_communication(program=program).save("communication.ssp")
Path("inputs.json").write_text(json.dumps(settings))
```

`num_examples` counts complete calls, not tokens or individual samples. The
Program fixes call shapes and static metadata; planning chooses their traversal.
For different shapes, use `example_calls=[(args0, kwargs0), ...]` instead of a
single signature and count. This example uses the model's fused loss operation;
it does not materialize an extra full-vocabulary logits tensor in Python.

## Plan on CPU

This process needs neither PyTorch nor a GPU:

```python
from shadowspill import ProfiledProgram, CommunicationInfo, MemoryBudgets, plan_program

plan = plan_program(
    ProfiledProgram.load("profiled.ssp"),
    memory_budgets=MemoryBudgets(execution_bytes=24 << 30, spill_bytes=64 << 30),
    communication_info=CommunicationInfo.load("communication.ssp"),
)
plan.save("training.ssp")
```

## Execute in a fresh worker

The catalog initializer generates parameters and RoPE buffers in bounded CPU
chunks. It never constructs a second full model on the GPU. Use the same model
configuration and data settings saved during preparation:

```python
import json
from pathlib import Path

from shadowspill import Plan, Trainer
from workloads.recipes.text.data.calls import Inputs
from workloads.recipes.text.initialization import from_catalog

settings = json.loads(Path("inputs.json").read_text())
source = Inputs(settings)
fill_tensor = from_catalog(
    settings["model"], seed=0, max_seq_len=settings["sequence_length"],
)

with Trainer(Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)
    trainer.fit(source, steps=100, run_dir="runs/llama")
    trainer.save("runs/llama/checkpoint.ssp")
```

Construct Trainer before initializing another device allocator. To resume,
replace `initialize` with `load("runs/llama/checkpoint.ssp")` and increase the total
`steps` target. The checkpoint restores model/optimizer state, progress and the
input source's position. For a custom loop, use `trainer.step(**next(source))`.

Build and plan `build_forward` separately for validation, then pass its Plan as
`evaluation_plan` to share current model weights. See [Trainer](../python/api/training.md)
for schedules, evaluation and checkpoints, [distributed setup](../python/api/distributed.md)
for multiple workers, and [custom-model integration](../python/custom-models.md) for
replacing the supplied model and data helpers.
