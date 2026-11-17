# Workloads

`shadowspill-workloads` is the optional library of supplied model architectures,
checkpoint readers, input adapters and training definitions. Custom packages use
the same public extension API and do not need this library.

Start with the [training client](../../../../docs/python/client.md) and
[built-in models and optimizers](../../../../docs/python/catalog.md), or browse
the detailed [architecture catalog](MODELS.md).

## Organization

| Owner | Responsibility |
| --- | --- |
| [models/](models) | One package per family: architecture, configuration, checkpoint mapping, LoRA targets and registration |
| [models/shared/](models/shared) | Reusable model layers, math and checkpoint mapping components |
| [definitions/](definitions) | Shared public registration, preparation and objective hooks |
| [inputs/](inputs) | Input schemas, processors, packing adapters and corpus readers |
| [objectives/](objectives) | Reusable losses and metrics |
| [initialization/](initialization) | Bounded checkpoint loading and parameter initialization |
| [lora/](lora) | Shared adapter selection and application |
| [policies/](policies) | Execution chunking and retention choices |

Each family declares its presets, checkpoint reader, pretrained identifiers and
training hook in `models/<family>/registration.py`. The catalog derives its
lookups from those declarations. See
[adding a family](../../../../docs/development/workloads.md) for the source
pattern and validation steps.

The engine owns build/profile/plan/execute, worker startup, shared state and
checkpoint orchestration. Model integrations supply computation and declarations;
they do not add another trainer. Shared process infrastructure lives in
[`shadowspill.orchestration`](../../../engine/src/shadowspill/orchestration/README.md).

## Examples and extension

- [Training examples](../../../../docs/examples/README.md)
- [Custom definitions](../../../../docs/python/advanced/definitions.md)
- [Non-text training](../../../../docs/examples/generic-training.md)
- [Benchmark quickstart](../../../../benchmarks/quickstart.md)
- [Distributed API](../../../../docs/python/api/distributed.md)
