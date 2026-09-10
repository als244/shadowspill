# Python quickstart

Build a runtime, import the model's state into its spill pool, plan a step,
then call it. The sections below follow that order.

## Create the runtime

Construct `Runtime` before constructing or loading model state and before
PyTorch performs any accelerator allocation. The runtime installs the process
allocator, creates and registers the execution and spill pools, starts its C
worker, and calibrates transfers directly between those real pool addresses.
Registering a large pinned spill arena after anonymous model state has claimed
host pages can change its physical DMA mapping and measured bandwidth.

```python
from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import Runtime

runtime = Runtime(
    pools={
        "device": device(physical_capacity=24 << 30),
        "spill": pinned_host(capacity=64 << 30),
    },
    routes={
        "fetch": transfer_route(source="spill", destination="device"),
        "evict": transfer_route(source="device", destination="spill"),
    },
)
```

Pool names are user-defined. `execution="device"` and `spill="spill"` in the
examples below select those names; they are not reserved strings.

Only after `Runtime(...)` returns should the process construct or load the
model and import its registered state into the spill pool.

## Import model state

Planning requires registered model state to reside in the runtime spill pool.
Assign the returned model so Python can release the original CPU module when
no other references remain.

```python
from shadowspill.pytorch import import_model_state

model = import_model_state(
    model,
    runtime=runtime,
    pool="spill",
    release_source=True,
)
```

The returned module has distinct Python object identities, preserves parameter
ties and views, and points its registered storages directly at runtime-owned
spill leases.

## Plan accumulated training

```python
from functools import partial

import torch

from shadowspill.pytorch import plan_step


def objective(model, tokens, targets):
    return model(tokens, labels=targets).loss


def zero_state(
    name: str, tensor: torch.Tensor, parameter: torch.nn.Parameter
) -> None:
    """Moment-based optimizers start at zero; ShadowSpill never assumes it."""

    with torch.no_grad():
        tensor.zero_()


train_step = plan_step(
    model,
    objective=objective,
    optimizer=partial(torch.optim.AdamW, lr=3e-4, weight_decay=0.1),
    optimizer_state_init=zero_state,
    example_inputs=[
        [tokens_example_0, targets_example_0],
        [tokens_example_1, targets_example_1],
    ],
    runtime=runtime,
    execution="device",
    spill="spill",
    execution_budget=20 << 30,
    spill_budget=60 << 30,
    artifact_store="/local-fast-storage/shadowspill-planning",
    profiling_metadata=[
        {"sequence_lengths": [4096]},
        {"sequence_lengths": [512] * 8},
    ],
)
```

The outer example-input sequence fixes the accumulation-round count. Each
runtime call must have the same outer structure and matching tensor geometry.
ShadowSpill performs one forward/objective/backward contribution per round and
one optimizer update per call. It does not divide accumulated gradients.

`profiling_metadata` is JSON-compatible artifact identity for data-dependent
measurement effects. It is not passed to the model. Concrete examples still
supply the values used for capture and isolated profiling.

`artifact_store` roots the exports, compiled graphs, profiles and plans this
call may reuse and contribute to. Point it at fast local storage. Sweeps that
share their build work but keep their own plans root the two trees apart with
`build_store` and `plan_store`, and `build_store_mode`/`plan_store_mode` say
what this run does with each; see the [artifact store](artifact-store.md).

## Execute and inspect

```python
result = train_step(
    [
        [tokens_0, targets_0],
        [tokens_1, targets_1],
    ]
)

print(result.step_number)
losses = result.objectives
```

`losses` contains one detached scalar tensor per accumulation round. Ordinary
calls do not collect a detailed runtime trace; tracing is an explicit
operation below. The [frontend API](api/frontend.md#inputs-objectives-and-partitioning)
documents optional nondifferentiated objective metrics.

Tracing and profiler annotations are independent and disabled by default:

```python
debug_result = train_step(
    [[tokens_0, targets_0], [tokens_1, targets_1]],
    runtime_trace=True,
    profiler_annotations=True,
)
diagnostics = debug_result.diagnostics.result()
```

Resolve a traced step's diagnostics before launching another traced step.
Resolving may wait for the recorded events; an ordinary `runtime_trace=False`
call does not perform this diagnostic synchronization.

Use [Interpreting a PlanReport](plan-report.md) to inspect the selected plan and
[Interpreting StepResult diagnostics](step-diagnostics.md) to reconcile one
real call with its profiles and simulator prediction.

## Checkpoint and restore

```python
checkpoint = train_step.state_dict()
torch.save(checkpoint, "checkpoint.pt")

train_step.load_state_dict(checkpoint)
```

The checkpoint has exactly `model`, `optimizer`, and `step`, and its model
mapping loads into an ordinary `nn.Module`. `state_dict()` copies synchronously
into CPU memory outside the runtime pools, so serializing it can run on another
thread while training continues -- at the cost of one further copy of model and
optimizer state to budget for beside the pool itself.

Take the checkpoint before closing: optimizer state belongs to the plan, so
`close()` releases it and `state_dict()` afterwards raises. See [checkpoints and
closing](api/frontend.md#checkpoints-and-closing).

## Plan forward only

`plan_forward()` plans inference: one flat example-input sequence, no optimizer,
and no accumulation.

```python
from shadowspill.pytorch import plan_forward

run_forward = plan_forward(
    model,
    example_inputs=[token_example, conditioning_example],
    runtime=runtime,
    execution="device",
    spill="spill",
)

outputs = run_forward([tokens, conditioning])
```

## Close

Close the callable, export persistent state if ordinary CPU tensors are
needed, then close the Python runtime handle.

```python
from shadowspill.pytorch import export_model_state

train_step.close()  # or run_forward.close()
model = export_model_state(
    model,
    runtime=runtime,
    release_runtime=True,
)
runtime.close()
```

Closing copies nothing and moves no weights: the model's parameters keep the
spill-pool storage `import_model_state()` gave them, which already holds every
step's updates, and `export_model_state()` is the call that copies the values
into ordinary CPU tensors.

Both `Runtime` and planned callables are context managers. Explicit lifecycle
calls make ownership and failure handling easiest to audit. [Errors, failures,
and cleanup](failures.md) gives the close order and what a failure rolls
back.
