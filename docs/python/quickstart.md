# Python quickstart

## Create the runtime

Construct `Runtime` before PyTorch performs any accelerator allocation. The
runtime installs the process allocator, creates the execution and spill pools,
starts its C worker, and calibrates transfer capabilities.

```python
from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import Runtime

runtime = Runtime(
    pools={
        "device": device(physical_capacity=24 << 30),
        "spill": pinned_host(capacity=64 << 30),
    }
)
```

Pool names are user-defined. `execution="device"` and `spill="spill"` in the
examples below select those names; they are not reserved strings.

## Relocate model state

Planning requires registered model state to reside in the runtime spill pool.
Assign the returned model so Python can release the original CPU module when
no other references remain.

```python
from shadowspill.pytorch import relocate_model_state

model = relocate_model_state(
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

from shadowspill.pytorch import ObjectiveResult, plan_step


def objective(model, tokens, targets):
    loss = model(tokens, labels=targets).loss
    return ObjectiveResult(loss=loss, metrics={"loss": loss.detach()})


train_step = plan_step(
    model,
    objective=objective,
    opt=partial(torch.optim.AdamW, lr=3e-4, weight_decay=0.1),
    example_inputs=[
        [tokens_example_0, targets_example_0],
        [tokens_example_1, targets_example_1],
    ],
    runtime=runtime,
    execution="device",
    spill="spill",
    execution_budget=20 << 30,
    spill_budget=60 << 30,
    planning_cachedir="/local-fast-storage/shadowspill-planning",
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

`profiling_metadata` is JSON-compatible cache identity for data-dependent
measurement effects. It is not passed to the model. Concrete examples still
supply the values used for capture and isolated profiling.

## Execute and inspect

```python
result = train_step(
    [
        [tokens_0, targets_0],
        [tokens_1, targets_1],
    ]
)

print(result.objectives)
print(result.metrics)
print(result.step_number)
```

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

The checkpoint has exactly `model`, `optimizer`, and `step`. Its model mapping
is compatible with an ordinary `nn.Module.load_state_dict()` call.
`state_dict()` synchronously copies into ordinary CPU memory outside runtime
pools. After it returns, filesystem serialization can run on another thread or
process while training continues because the checkpoint no longer aliases
runtime-owned state.

## Plan forward only

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

Close the callable, externalize persistent state if ordinary CPU tensors are
needed, then close the Python runtime handle.

```python
from shadowspill.pytorch import externalize_model_state

train_step.close()  # or run_forward.close()
model = externalize_model_state(
    model,
    runtime=runtime,
    release_runtime=True,
)
runtime.close()
```

Both `Runtime` and planned callables are context managers. Explicit lifecycle
calls make ownership and failure handling easiest to audit.
