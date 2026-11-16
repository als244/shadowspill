# Add a model, objective or optimizer

An extension is an installed server package. It describes the computation;
the engine handles preparation, packing, workers, updates and checkpoints.
The client selects your definitions by name, just like built-ins.

This guide builds a small regression example with a custom model, weighted loss
and SGD optimizer. The complete runnable package is in
[`examples/extensions/custom_training`](../../../examples/extensions/custom_training).
It uses the public extension API and does not depend on `shadowspill-workloads`.
Use the [hook reference](definitions.md) for the full contracts.

## 1. Create the package

Use this layout for your server implementation:

```text
custom_training/
├── pyproject.toml
├── src/example_training/
│   ├── __init__.py    # Registration and source identity
│   ├── model.py       # Architecture, settings, inputs and initial weights
│   ├── objective.py   # Loss and output declarations
│   └── optimizer.py   # Optimizer factory and runtime controls
├── server.py
└── train.py           # Client application; imports only shadowspill_client
```

The example's `pyproject.toml` publishes one registration hook:

```toml
[project.entry-points."shadowspill.definitions"]
training = "example_training:register"
```

Registration and configuration must work without constructing a model or importing
Torch. Keep framework imports inside the model, objective, initialization and
optimizer functions that execute in workers.

## 2. Define the model and its inputs

In [`model.py`](../../../examples/extensions/custom_training/src/example_training/model.py),
`Config` validates architecture settings and `create(config)` returns an ordinary
PyTorch module. No required model base class or trainer adapter is involved:

```python
def create(config):
    import torch

    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(config.width, 1)

        def forward(self, features):
            return self.projection(features).squeeze(-1)

    return Regressor()
```

The input schema describes one Datum's `features` tensor as FP32 with shape
`(config.width,)`. A scheduled `max_examples` capacity lets the standard adapter
stack records and pad unused rows. You do not implement batching or a worker loop.

The model advertises `scalar_prediction`, which is the capability the example
objective requires. Give capabilities specific meanings: here, one floating-point
prediction per example. A different model with that contract can reuse the loss.

For a language model, expose the token outputs required by the selected objective
and reuse suitable text input and preparation helpers. Custom packing or operator
resources are optional hooks; add them when your architecture requires them.
See [input adapters](definitions.md#input-adapters) and
[build descriptions](definitions.md#geometry-aware-model-builds).

## 3. Describe initial weights

Model construction also runs on storage-free tensors during preparation. Initial
training weights therefore come from the `initialization(config, context)` hook,
not constructor side effects.

The example uses the engine's `fill_generated` helper to fill requested tensor
slices reproducibly from a seed. It declares its maximum temporary memory and
chunk size in `ModelInitialization`. This avoids allocating another complete
model just to initialize its weights, and gives the same values across different
loader chunk sizes.

For a pretrained architecture, provide the mapping from checkpoint names/layouts
to model parameters and fill the requested slices through the existing bounded
I/O APIs. Declare any temporary buffers that must be regenerated. Native training
checkpoint save/restore is handled by the engine; it does not need a second
model-specific checkpoint loop. See [bounded initialization](definitions.md#bounded-model-initialization).

## 4. Define the objective

[`objective.py`](../../../examples/extensions/custom_training/src/example_training/objective.py)
contains an ordinary differentiable function:

```python
def compute(model, inputs, loss_inputs, loss_params):
    import torch
    from shadowspill.pytorch.contracts import ObjectiveResult

    prediction = model(**inputs)
    valid = loss_inputs["valid"]
    error = prediction - torch.where(valid, loss_inputs["target"], 0)
    weight = torch.where(valid, loss_inputs["weight"], 0)
    return ObjectiveResult(
        (error.square() * weight).sum(),
        {"valid_examples": valid.sum()},
    )
```

Its loss-input schema declares a target, an optional weight and an optional
validity flag. Real records default to valid; artificial padding is invalid.
`Participation` identifies that flag, and `padding_safe=True` promises that
padding contributes neither loss nor gradients. The `metrics` output set lists
exactly which inputs and metrics this computation uses.

Choose reduction explicitly. This example returns a weighted sum. To train on
the mean of four examples, give each example weight `0.25`; keep those weights
when splitting that update across `forward_backward` calls. The engine adds no
implicit averaging.

You only need a new objective when the existing ones do not express your loss.
Compatible objectives can be combined with `Objective.sum(...)`; they need not
be built into the model. [Objective contracts](definitions.md#model-and-objective-hooks)
describe controls, output selection and model-aware binding.

## 5. Define the optimizer

[`optimizer.py`](../../../examples/extensions/custom_training/src/example_training/optimizer.py)
wraps PyTorch SGD. Its factory receives the model, static configuration and
tensor-valued runtime controls:

```python
def create(model, config, controls):
    import torch

    return torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=controls["lr"],
        foreach=False,
    )
```

Declare `lr` in `ControlSchema` and bind it to `optimizer_group.0.lr`. The engine
checks that the captured update actually reads it. The declared default supports
preparation and omitted controls; the example client supplies the learning rate
on each `step` or `optim_step` call.

Keep the update capturable and select only trainable parameters. An existing
optimizer such as `adamw` needs no custom factory. See
[optimizer definitions](definitions.md#optimizer-definitions) for precision and
control details.

## 6. Register and install

[`register(registry)`](../../../examples/extensions/custom_training/src/example_training/__init__.py)
adds the model, objective and optimizer with their corresponding definition
records. Each has an exact identity, such as `example/linear_regressor@1`.
Clients use the unique short name, `linear_regressor`; saved Plans retain the
exact identity and source digest.

The example hashes all Python implementation files in its package and records
the installed distribution version. If you add an external implementation
dependency, include it in that source closure too. Source changes invalidate
incompatible saved Plans, including changes made in editable installations.

On the server, after [installing the engine](../../development/repository.md#setup),
install the example from the repository root:

```bash
python -m pip install --no-deps ./examples/extensions/custom_training
```

The `--no-deps` command uses that existing engine installation. For your own
published package, declare its dependencies and install it normally. Install the
same package on every worker. The client needs only `shadowspill-client`.

Enable the installed distribution when starting the service:

```python
from shadowspill.server import Allocation, Server

server = Server(
    allocation=Allocation.from_environment(),
    storage_root="./custom-training-data",
    workload_packages=["example-training-definitions"],
)
server.serve(host="127.0.0.1", port=8000)
```

Set the same `SHADOWSPILL_TOKEN` on the server and client. Start
[`server.py`](../../../examples/extensions/custom_training/server.py) inside your
GPU server. Adding `"shadowspill"` to `workload_packages` enables
the built-in AdamW definition; adding `"shadowspill-workloads"` enables the
optional model catalog. Neither is needed by this example.

## 7. Select the definitions and train

The ordinary client settings select your installed implementation:

```python
training = service.create_training_client(
    model=Model("linear_regressor", config={"width": 4},
                initialize=Weights.random(seed=0)),
    objective=Objective("weighted_mse"),
    optimizer=Optimizer("sgd"),
    batch=BatchConfig(limits={"max_examples": 4}),
    memory=Memory(device_gib=3, spill_gib=1),
    operations={"step": ["metrics"], "forward_backward": ["metrics"]},
).result()
```

Prepare both operations because this example uses whole and split updates.
Preparing `forward_backward` also prepares `optim_step` to finish accumulation.

The complete [`train.py`](../../../examples/extensions/custom_training/train.py)
provides imports, four Datums, a whole update, a split update and checkpoint
resume. Copy that one file to the client and run:

```bash
python train.py --ssh-host your-server
```

Omit `--ssh-host` when the service is on local port 8000. Setup happens once;
subsequent calls submit data and runtime controls:

```python
training.step(data, optim={"lr": 0.05}).result()
training.forward_backward(data[:2]).result()
training.forward_backward(data[2:]).result()
training.optim_step(optim={"lr": 0.05}).result()
```

Use `Model(..., trainable=["projection.weight"])` to freeze the example's bias.
LoRA needs a model configuration hook that installs adapters and declares their
targets; the generic engine then honors the resulting trainable parameters.
[The catalog](../catalog.md#lora-and-trainable-parameters) shows the client settings
for models that provide this capability.

## 8. Check your extension

Before a large training run, compare a small update against eager framework
execution: loss, gradients, changed weights and frozen parameters. Check padding,
split accumulation, per-update LR and continuation from a saved checkpoint.
Test bounded initialization with different chunk sizes. Registration and setup
inspection should work without importing Torch or the optional catalog.

Adding another family should primarily change its architecture/configuration,
weight mapping and capability declaration. Reuse objectives and optimizers;
keep preparation and execution in the engine. Specialized kernels may also need
[operator integration](operator-integration.md) and allocation declarations.

For a contribution to the built-in catalog, follow the
[family organization guide](../../development/workloads.md). The same public
definition hooks apply; each built-in family keeps its integration metadata in
one declaration beside its architecture.
