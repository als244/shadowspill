# ShadowSpill

Run PyTorch models within a memory budget by planning which tensors to keep on
the device, spill to host or external storage, and recompute during backward.

[Install](docs/development/repository.md#setup) ·
[Training example](docs/examples/generic-training.md) ·
[Python guide](docs/python/README.md) · [All docs](docs/README.md)

## Build, profile and plan

The application supplies its model, optimizer, objective and example inputs.
Build and profile on a representative GPU; reuse those artifacts to plan on CPU.

```python
import shadowspill as ss

# Capture, partition and compile the computation.
program = ss.build_step(model, optimizer=optimizer, objective=objective,
                        example_args=(inputs, targets))
profiles = ss.measure_program(program)
profiled = ss.apply_profiles(program, profiles)
communication = ss.measure_communication(program=program)

# Choose a schedule that fits the declared budgets.
plan = ss.plan_program(profiled, memory_budgets=memory_budgets,
                       communication_info=communication)
plan.save("training.ssp")
ss.write_report(plan, "report")
```

Each phase result can be saved and reused. See the [phase guide](docs/python/phases.md)
for process boundaries and links to the detailed controls.

## Execute in a fresh worker

Create Trainer before any other device allocation in this worker. Only execution
opens ShadowSpill's runtime pools; the saved Plan contains its task implementations.

```python
import shadowspill as ss

with ss.Trainer(ss.Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)  # Fill bounded buffers from a checkpoint or RNG.
    trainer.fit(training_batches, steps=100)
    trainer.save("checkpoint.ssp")
```

The [complete example](docs/examples/generic-training.md) defines the model,
objective, inputs and initializer used by these APIs.

## Model recipes

For model checkpoints and datasets, the [shared Python runner](workloads/recipes/README.md)
composes candidate construction, profiling, search and training. Start from a
commented script and edit its configuration:

| Workload | Example |
| --- | --- |
| Qwen3.5 LoRA | [Script](workloads/recipes/text/qwen35_lora/example.py), [guide](workloads/recipes/text/qwen35_lora/README.md) |
| GLM-5.3-Flash LoRA | [Script](workloads/recipes/text/glm53_flash_lora/example.py), [guide](workloads/recipes/text/glm53_flash_lora/README.md) |
| Qwen3 full training and expert parallelism | [Guide](workloads/recipes/text/qwen3/README.md) |
| Llama full training | [Script](workloads/recipes/text/train_llama3.py) |

The separate [config-based training harness](training/README.md) supports eager
comparisons and reference checks. Use [quickstart](benchmarking/quickstart.md)
for generic model-factory experiments. These source-tree tools call the same
library APIs; model-specific policies stay outside the core engine.
