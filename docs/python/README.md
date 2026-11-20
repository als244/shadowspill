# Python guides and API

## Guides

Start with the [training client](client.md) and the
[model, optimizer and objective catalog](catalog.md).

| Goal | Guide |
| --- | --- |
| Train through the public service | [Complete client example](../examples/client-training.md) |
| Run an editable training example | [Standalone demos](../../examples/demos/README.md) |
| Use a command-line training loop | [Training applications](../../examples/training/README.md) |
| Bring a custom model or control each phase | [Writing extensions](advanced/writing-extensions.md), [phase guide](phases.md) |
| Tune, inspect or extend the system | [Advanced guides](advanced/README.md) |
| Register a custom model, objective or optimizer | [Component hooks](advanced/components.md) |

Setup and runtime are separate: `ServiceClient` describes and creates a run;
`TrainingClient` submits batches to ready workers. The service owns preparation,
initialization and execution. Direct phase entry points are available from
`shadowspill`, with PyTorch policies in `shadowspill.pytorch`.

## API reference

Generated from the docstrings, these pages give every public object's purpose,
arguments, results and errors. Start with a guide above for the order of calls
and a complete example.

| Package | Reference |
| --- | --- |
| `shadowspill_client`, the training client | [Client reference](../reference/client/index.rst) |
| `shadowspill_client.extensions`, the records an extension declares | [Writing extensions](../reference/client/extensions.rst) |
| `shadowspill`, the engine and service | [Engine reference](../reference/engine/index.rst) |
