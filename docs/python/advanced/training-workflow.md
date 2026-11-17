# Split updates, evaluation and resume

This complete example separates setup from training calls. It uses the supplied
small regression model so the client needs only Python 3.12+ and NumPy, with no
tokenizer, model weights or training framework installed locally. The same
`ServiceClient` / `TrainingClient` interface serves installed text and custom
models. For a first training run, start with the [text example](../../examples/client-training.md).

## Start the server

Use the configured GPU environment on a training host, with the
engine, workloads and their MLOps provider installed. Set `SHADOWSPILL_TOKEN` to the
same credential on server and client. Start this once in the assigned allocation:

```python
from shadowspill.server import Allocation, Server

server = Server(
    allocation=Allocation.from_environment(),
    service_root="/path/to/service-data",
    workload_packages=["shadowspill", "shadowspill-workloads"],
)
server.serve(host="127.0.0.1", port=8000)
```

The server uses the existing build/profile/plan/execute engine. It discovers the
assigned local devices and owns worker launch, initialization and cleanup. It
does not request a scheduler allocation. This example uses the default local worker.
The [connection reference](service-client.md#connection-and-setup) describes
deployment settings.

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

setup = dict(
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
)


def prepare_run(service):
    preview = service.describe_training(**setup).result()
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

For the supplied packed text models, a CPU-only helper creates the build policies:

```python
from shadowspill_client.text import build_candidates

prepare = Preparation(candidates=build_candidates(
    microbatch_tokens=[8192, 16384, 32768],
    projection_chunks=[0, 8192],
))
```

Microbatch tokens are per input partition (per rank with data parallelism).
Zero means unchunked. `moe_chunks` defaults to `[0]`, keeping expert computation
at the full microbatch; supply its own size list to search expert chunks too.
Accumulation counts are derived from the selected microbatch sizes and global
token limit unless explicitly supplied. The helper keeps the ordinary retention,
ordering and saved-head defaults; sequence attention boundaries remain intact.

## Submit updates and resume

```python
with ServiceClient(ssh=SSH("training-host", 8000)) as service:
    training = prepare_run(service)
    try:
        # First update: one complete call with a global batch.
        first_batch = make_batch(0)
        result = training.step(
            first_batch, optim={"lr": 0.001}, grad_scale=1 / 8,
        ).result()
        print(result.completed_updates, result.loss)

        # Second update: unequal backward contributions, then one optimizer call.
        second_batch = make_batch(1)
        training.forward_backward(second_batch[:3]).result()
        training.forward_backward(second_batch[3:]).result()
        result = training.optim_step(optim={"lr": 0.001}, grad_scale=1 / 8).result()
        print(result.completed_updates, result.loss)

        evaluation = training.forward(
            [Datum(inputs=row.inputs) for row in second_batch],
            outputs="predictions",
        ).result()
        print(evaluation.outputs["prediction"].reference)
        # Tensor downloads are explicit; metadata inspection does not fetch them.

        checkpoint = training.save_checkpoint(
            name="after-two", metadata={"next_data_seed": 2},
        ).result()
        plans = training.plans
    finally:
        training.close().result()

    # Reuse saved Plans and restore weights, optimizer state and update count.
    resumed = service.create_training_client(
        plans=plans, checkpoint=checkpoint,
    ).result()
    try:
        result = resumed.step(
            make_batch(2), optim={"lr": 0.001}, grad_scale=1 / 8,
        ).result()
        print(result.completed_updates)  # 3
    finally:
        resumed.close().result()
```

Input values can change within the admitted schema and limits. Every submission
returns a future; `.result()` waits for completion. A wait timeout does not cancel
or repeat a training operation. Save the creation/request IDs for reconnect and
query the original outcome after an uncertain connection failure. See the
[client reference](service-client.md) for recovery, downloads and saved Plans.

`training.close().result()` releases that run's workers. Closing `ServiceClient`
only closes the connection; it does not stop an accepted training run.

For text training, use the [Llama application](../../../examples/training/llama3/README.md),
[Qwen3 application](../../../examples/training/qwen3.md) or
[Qwen3.5 LoRA application](../../../examples/training/qwen35_lora.md). Their shared
setup/runtime helpers and corpus code sit outside the model implementations.
For a new architecture, implement the public
[model and objective definitions](definitions.md); the engine does not
require the supplied catalog.
