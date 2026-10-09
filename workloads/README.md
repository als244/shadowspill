# Workloads

Example models and data recipes passed into the public ShadowSpill APIs.
These modules live in the repository and are not installed as part of the
`shadowspill` package. Core planning, runtime, lowering, and the generic trainer
must never import `workloads`.

## Choose a model

The [example model catalog](MODELS.md) is the central reference for:

- Llama 3, dense Qwen 3.5, OLMoE, Qwen 3 MoE, and Qwen 3.5 MoE.
- Python constructors, quickstart names, preset dimensions and parameter counts.
- Shape configuration, weights/masters/gradients/optimizer dtypes, and rounding.
- Frozen/trainable parameters, LoRA availability, and expert parallelism.
- Inputs, losses, packing, and runnable Python/JSON/CLI examples.

`mlops_qwen35` is dense; `mlops_qwen3moe` and `mlops_qwen35moe` select
Qwen 3 and Qwen 3.5 MoE. Their default dimensions match the 30B-A3B and
35B-A3B models, respectively.

All MLOps MoE models expose optional expert parallelism through the same
constructor arguments. Choose the architecture first, then supply `ep_group`
and `token_capacity` (or an existing `buffer`). No separate EP model import is
needed; see [expert parallelism](MODELS.md#expert-parallelism).

## Layout

| Directory / file | Purpose |
|---|---|
| [pytorch/](pytorch/) | Readable PyTorch model references and shared configuration types |
| [mlops/](mlops/) | Model definitions using the separately installed MLOps operation library |
| [common/](common/) | Shared math, rotary tables, packing metadata and loss helpers |
| [recipes/](recipes/) | Caller-side model/data/objective composition |
| [full_model.py](full_model.py) | Reproducible full-model specifications and construction |
| [precision.py](precision.py) | Explicit precision settings used by supplied workloads |

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
