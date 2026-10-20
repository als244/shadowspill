# Workloads

Example models and data recipes passed into the public ShadowSpill APIs.
These modules live in the repository and are not installed as part of the
`shadowspill` package. Core planning, runtime, lowering, and the generic trainer
must never import `workloads`.

## Choose a model

The [example model catalog](MODELS.md) is the central reference for:

- Llama 3, dense Qwen 3.5, OLMoE, Qwen 3 MoE, and Qwen 3.5 MoE.
- Catalog names, preset dimensions and parameter counts.
- Shape configuration, weights/masters/gradients/optimizer dtypes, and rounding.
- Frozen/trainable parameters, LoRA availability, and expert parallelism.
- Inputs, losses, packing, and runnable Python/JSON/CLI examples.

The [catalog](models/catalog.py) names every supplied model `<family>-<size>`:
`llama3-8b`, `qwen35-9b`, `qwen35-27b`, `olmoe-7b-a1b`, `qwen3moe-30b-a3b`,
`qwen35moe-35b-a3b`, the small `llama3-1b` and `qwen35-2b`, each family's
`numerical` and `tiny` shapes, and the GLM entries. The implementation,
`mlops` or `pytorch`, is chosen beside the name.

All MLOps MoE models expose optional expert parallelism through the same
constructor arguments. Choose the architecture first, then supply `ep_group`
and `token_capacity` (or an existing `buffer`). No separate EP model import is
needed; see [expert parallelism](MODELS.md#expert-parallelism).

## Layout

| Directory / file | Purpose |
|---|---|
| [models/](models/) | One package per model family: its configuration, checkpoint reader, README and implementations (`pytorch`, `mlops`); `models/shared/` holds the code families share |
| [lora/](lora/) | LoRA targeting for the supplied models over MLOps's reusable projections |
| [recipes/](recipes/) | Caller-side model/data/objective composition |
| [full_model.py](full_model.py) | Reproducible full-model specifications and construction |
| [numerical.py](numerical.py) | Exact-scale cases shared by the numerical qualification's two arms |
| [precision.py](precision.py) | Explicit precision settings used by supplied workloads |
| [providers.py](providers.py) | The implementation names |

Model definitions describe computation. Recipes choose data, objectives,
initialization, and run policy. Reusable training and forward execution live in
`shadowspill.training` and accept ordinary user models without this directory.

## Run an example

- [Training recipes](../training/README.md): complete configs, data preparation,
  dtype overrides, schedules, logging, evaluation and checkpoints.
- [Benchmark quickstart](../benchmarking/quickstart.md): throughput presets or
  a user experiment factory, budget/geometry search, plots and resolution plans.
- [Generic training example](../docs/examples/generic-training.md): a non-text
  model and iterable data.
- [Distributed API](../docs/python/api/distributed.md): devices, groups,
  parameter replicas, sharding, and verified symmetric planning.
