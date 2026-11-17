# Split updates, evaluation and resume

This complete example separates setup from training calls. It uses the supplied
small regression model so the client needs only Python 3.12+ and NumPy, with no
tokenizer, model weights or training framework installed locally. The same
`ServiceClient` / `TrainingClient` interface serves installed text and custom
models. For a first training run, start with the [text example](../../examples/client-training.md).

## Start the server

[Start the service](../../../README.md#start-the-service) on the training host.
Use the same `SHADOWSPILL_TOKEN` on the server and client. The
[connection reference](service-client.md#connection-and-setup) covers allocation
and extension-package settings.

## Define client settings and inputs

Install the client from `packages/client` or its built wheel on the client host.
The commands use an existing OpenSSH alias `training-host`; a local client can instead
use `ServiceClient(base_url="http://127.0.0.1:8000")`.

Put the following two client blocks in one Python script. Setup owns model,
objective, optimizer, batch and memory choices. `prepare_run` returns a ready
client; the runtime section below has no build or worker initialization code.

```python
import numpy as np

from shadowspill_client import (
    SSH, BatchConfig, BuildCandidates, Datum, Memory, Model, Objective,
    Optimizer, PlanningOptions, Preparation, ServiceClient, Weights,
)

def prepare_run(service):
    preview = service.describe_training(
        model=Model("regression", initialize=Weights.random(seed=0)),
        objective=Objective("weighted_squared_error"),
        optimizer=Optimizer(
            "adamw",
            precision={"gradient_dtype": "float32", "opt_state_dtype": "float32"},
        ),
        batch=BatchConfig(limits={"max_examples": 8}),
        memory=Memory(4, 1),  # Device and spill budgets in GiB per worker.
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["predictions"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[4, 8]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
        ),
    ).result()
    print(preview.summary())
    return service.create_training_client(config=preview.config).result()


def make_batch(seed):
    values = np.random.default_rng(seed).normal(size=(8, 16)).astype(np.float32)
    return [
        Datum(inputs={"value": row}, loss_inputs={"targets": row[:8] * 0.5})
        for row in values
    ]
```

The model schema requires 16 input features and 8 targets per record. The objective
sums squared error over the 8 targets and then over records. Scaling by `1 / 8`
below therefore gives the mean of the per-record squared-error sums. No batch or
token normalization happens implicitly.

The batch limit includes scheduled padding. A candidate scheduling two physical
microbatches of 8 exceeds `max_examples=8` and is excluded; specifying counts does
not relax the cap. Preparation measures supported candidates and selects fitting
Plans. Omit `prepare` to use installed defaults instead.

For text packing, chunking and parallel planning choices, see the
[preparation reference](service-client.md#preparation-and-local-planning).
`training.validate_batch(batch)` checks actual packing against the prepared layouts.

## Submit updates and resume

```python
service = ServiceClient(ssh=SSH("training-host", 8000))
training = prepare_run(service)

# First update: one complete call with a global batch.
first_batch = make_batch(0)
result = training.step(
    first_batch, optim={"lr": 0.001}, grad_scale=1 / 8,
).result()
print(result.completed_updates, result.loss / 8)

# Second update: unequal contributions, followed by one optimizer call.
second_batch = make_batch(1)
first = training.forward_backward(second_batch[:3])
second = training.forward_backward(second_batch[3:])
update = training.optim_step(optim={"lr": 0.001}, grad_scale=1 / 8)
print(update.result().completed_updates, (first.result().loss + second.result().loss) / 8)

evaluation = training.forward(
    [Datum(inputs=row.inputs) for row in second_batch], outputs="predictions",
).result()
print(evaluation.outputs["prediction"].reference)
# Tensor downloads are explicit; metadata inspection does not fetch them.

checkpoint = training.save_checkpoint(
    name="after-two", metadata={"next_data_seed": 2},
).result()
plans = training.plans
training.close().result()

# Reuse saved Plans and restore weights, optimizer state and update count.
resumed = service.create_training_client(plans=plans, checkpoint=checkpoint).result()
result = resumed.step(make_batch(2), optim={"lr": 0.001}, grad_scale=1 / 8).result()
print(result.completed_updates)  # 3
resumed.close().result()
service.close()
```

Input values can change within the admitted schema and limits. Every submission
returns a future; `.result()` waits for completion. A wait timeout does not cancel
or repeat a training operation. Save the creation/request IDs for reconnect and
query the original outcome after an uncertain connection failure. See the
[client reference](service-client.md) for recovery, downloads and saved Plans.

`training.close().result()` releases that run's workers. Closing `ServiceClient`
only closes the connection; it does not stop an accepted training run.

For editable text-training examples, use the [standalone demos](../../../examples/demos/README.md).
The [CLI applications](../../../examples/training/README.md) add schedules,
evaluation and checkpoint continuation. Their setup/runtime helpers and corpus
code sit outside the model implementations.
For a new architecture, implement the public
[model and objective definitions](definitions.md); the engine does not
require the supplied catalog.
