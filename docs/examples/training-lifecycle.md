# Training loop

This complete example creates a runtime, imports model state, trains, and
writes a checkpoint.

```python
from functools import partial

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
    opt=partial(torch.optim.AdamW, lr=3e-4, foreach=False),
    example_inputs=[batch(4)],
    runtime=runtime,
    execution="execution",
    spill="spill",
)

for _ in range(10):
    result = train_step([batch(4)])
    loss = result.objectives[0]
    print("optimizer step", result.step_number, "loss", loss)

torch.save(train_step.state_dict(), "checkpoint.pt")
train_step.close()
```

Each call runs the objective and backward pass followed by one optimizer
update. Runtime inputs must match the shapes, strides, dtypes, static values,
and structure supplied to `plan_step()`.

The scalar loss is `result.objectives[0]`. ShadowSpill validates and records
this explicit objective return during capture; it does not guess which model
output is a loss.

An ordinary `train_step()` returns `StepResult` without collecting or resolving
a runtime trace. `DiagnosticsHandle.result()`, checkpoint operations, and
lifecycle close are explicit synchronous boundaries.

`state_dict()` creates an ordinary CPU checkpoint before `torch.save()` begins.
The example closes the callable and then exits; long-lived embedding processes
should follow the complete ownership order in [Errors, failures, and
cleanup](../python/failures.md).
