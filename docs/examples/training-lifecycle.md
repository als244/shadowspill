# Training loop

This complete example creates a runtime, imports model state, trains, and
writes a checkpoint.

```python
import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import (
    Runtime,
    plan_step,
    import_model_state,
)


def objective(model, features, targets):
    error = model(features) - targets
    return error.square().mean()


def batch(rows):
    return [torch.randn(rows, 128), torch.randn(rows, 128)]


runtime = Runtime(
    pools={
        "execution": device(physical_capacity=4 << 30),
        "spill": pinned_host(capacity=2 << 30),
    },
    routes={
        "fetch": transfer_route(source="spill", destination="execution"),
        "evict": transfer_route(source="execution", destination="spill"),
    },
)

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 128),
)
model = import_model_state(model, runtime=runtime, pool="spill")

train_step = plan_step(
    model,
    objective=objective,
    optimizer=torch.optim.AdamW,
    hyperparams=("lr",),
    example_inputs=[batch(4)],
    runtime=runtime,
    execution="execution",
    spill="spill",
)

for _ in range(10):
    result = train_step([batch(4)], hyperparams={"lr": 3e-4})
    loss = result.objectives[0]
    print("optimizer step", result.step_number, "loss", loss)

torch.save(train_step.state_dict(), "checkpoint.pt")
train_step.close()
```

Each call runs the objective and backward pass followed by one optimizer
update. Runtime inputs must match the shapes, strides, dtypes, static values,
and structure supplied to `plan_step()`.

The optimizer is built plainly, at its defaults. Anything that varies between
steps is named at planning instead: `hyperparams=("lr",)` declares that the
learning rate is a value the caller supplies, and each call sets it. A value
named this way is captured once, by geometry, so a schedule that changes it
every step never recaptures the update. The optimizer's state asks nothing of
the caller: the optimizer declares what it keeps by running on meta parameters,
each entry starts where the optimizer's own first step would start it, and the
import that adopts the state for the plan puts it in the spill pool.

The scalar loss is `result.objectives[0]`. ShadowSpill validates and records
this explicit objective return during capture; it does not guess which model
output is a loss.

An ordinary `train_step()` returns `StepResult` without collecting or resolving
a runtime trace. `DiagnosticsHandle.result()`, checkpoint operations, and
lifecycle close are explicit synchronous boundaries.

## Returned metrics

The objective may return `ObjectiveResult(loss, metrics)`. Each call returns
`result.objectives[i]` and `result.metrics[i]` for microbatch `i`, in input
order. These are raw detached device tensors; ShadowSpill neither sums nor
averages them. Model outputs only become public step results if the objective
returns them. `plan_forward`, in contrast, returns its model's output for one
invocation.

For observations of final accumulated gradients and pre-update parameters,
pass `parameter_metrics=callback` at planning. The callback takes
`(compute_weight, accumulated_gradient)` and returns a tensor pytree. Its
results are available once per step in `result.parameter_metrics`, keyed by
parameter name. See [optimizer observations](../architecture/optimizer.md#observing-parameters-and-gradients).

Keep tensor reductions inside the callback and host conversion outside it.
For example, a callback returns `torch.linalg.vector_norm(weight,
dtype=torch.float32)` directly; a logger may copy and read that scalar after
`train_step(...)` has returned.

`state_dict()` creates an ordinary CPU checkpoint before `torch.save()` begins.
The example closes the callable and then exits; long-lived embedding processes
should follow the complete ownership order in [Errors, failures, and
cleanup](../python/failures.md).
