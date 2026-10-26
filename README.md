# ShadowSpill

Run PyTorch models within a memory budget by planning what to keep on the GPU,
spill to host memory, and recompute.

[Install](docs/development/repository.md#setup) ·
[Quickstart](benchmarking/quickstart.md) ·
[API guide](docs/architecture/offline-pipeline.md) ·
[Docs](docs/README.md)

## Phases

Pseudocode; application-specific model, optimizer and data setup is omitted.

```python
import shadowspill as ss

# 1. Build: export, partition and compile on a representative GPU.
program = ss.build_step(model, optimizer=optimizer, objective=loss,
                        example_args=(inputs, targets))

# 2. Profile: measure GPU task timing/memory and communication separately.
profiles = ss.measure_program(program)
profiled = ss.apply_profiles(program, profiles)
communication = ss.measure_communication(program=program)

# 3. Plan on CPU: reuse measurements with different GPU/spill budgets.
plan = ss.plan_program(profiled, memory_budgets=memory_budgets,
                       communication_info=communication)
ss.write_report(plan, "report")  # Simulated timeline and memory use.
plan.save("training.ssp")
```

**Execute in a fresh worker.** `Trainer` manages the Executor and training loop.
Create it before any other device allocation in that worker:

```python
import shadowspill as ss

# 4. Execute: only this phase creates ShadowSpill runtime pools.
with ss.Trainer(ss.Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)  # Bounded loading or random initialization.
    trainer.fit(training_batches, steps=100)
    trainer.save("checkpoint.ssp")
```

Each phase artifact supports save/load. See the [Trainer example](docs/examples/generic-training.md)
for runnable code, evaluation and checkpoint loading.

## Real model training

Use these model/input sketches with the phases above. Linked recipes include
real text preparation and loading; optimizer setup is abbreviated.

<details>
<summary><strong>Llama 3 8B — full training</strong></summary>

```python
from workloads.models.catalog import build_on_meta

model = build_on_meta("llama3-8b", dtype="bfloat16", max_seq_len=4096)
# Four examples of 4 × 4K tokens = 16 sequences per update; random initialization.
program = ss.build_step(model, optimizer=optimizer, objective=next_token_loss,
                        example_args=(tokens_4x4096, targets_4x4096), num_examples=4)
```

[Complete recipe](workloads/recipes/text/train_llama3.py)

</details>

<details>
<summary><strong>GLM-5.3-Flash — LoRA at 32K on eight GPUs</strong></summary>

```python
import torch
from workloads.models.glm53_flash.checkpoint import GLMCheckpoint

with GLMCheckpoint("/models/GLM-5.3-Flash") as checkpoint:
    model = checkpoint.build_model(lora_rank=32, lora_factor_dtype=torch.bfloat16)
    # One 32K sequence per GPU; frozen base keeps checkpoint precision, including FP8.
    program = ss.build_step(model, optimizer=optimizer, objective=next_token_loss,
                            example_args=(tokens_1x32768, targets_1x32768),
                            parallelism=ss.Parallelism.dp(ranks_per_replica=8))
```

[Complete recipe](workloads/recipes/text/glm53_flash_lora/README.md)

</details>

<details>
<summary><strong>Qwen3.5-35B-A3B — LoRA at 32K</strong></summary>

```python
from workloads.models.catalog import build_on_meta

model = build_on_meta("qwen35moe-35b-a3b", dtype="bfloat16", max_seq_len=32768,
                      lora={"rank": 32, "alpha": 32})
# Eight examples of 1 × 32K tokens; load the frozen checkpoint and initialize adapters.
program = ss.build_step(model, optimizer=optimizer, objective=next_token_loss,
                        example_args=(tokens_1x32768, targets_1x32768), num_examples=8)
```

[Complete recipe](workloads/recipes/text/train_qwen35_lora.py)

</details>

<details>
<summary><strong>Qwen3-30B-A3B — full training on eight GPUs</strong></summary>

```python
from workloads.models.catalog import build_on_meta

model = build_on_meta("qwen3moe-30b-a3b", dtype="bfloat16", max_seq_len=4096)
# 16 examples of 4 × 4K tokens per GPU = 512 sequences globally; random initialization.
program = ss.build_step(model, optimizer=optimizer, objective=next_token_loss,
                        example_args=(tokens_4x4096, targets_4x4096), num_examples=16,
                        parallelism=ss.Parallelism.dp(ranks_per_replica=8))
```

[Complete recipe](workloads/recipes/text/qwen3/README.md)

</details>

`Parallelism.dp(...)` shards weights, gradients and optimizer data by default.
See the [model catalog](workloads/MODELS.md) for other families and recipe details.
