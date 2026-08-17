# Training loop

This complete example creates a runtime, relocates model state, plans two
accumulation rounds, trains, and writes a checkpoint.

```python
from functools import partial

import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    plan_step,
    relocate_model_state,
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
    }
)

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 128),
)
model = relocate_model_state(model, runtime=runtime, pool="spill")

train_step = plan_step(
    model,
    objective=objective,
    opt=partial(torch.optim.AdamW, lr=3e-4, foreach=False),
    example_inputs=[batch(4), batch(7)],
    runtime=runtime,
    execution="execution",
    spill="spill",
    planning_cachedir="artifacts/planning-cache",
)

for _ in range(10):
    result = train_step([batch(4), batch(7)])
    losses = result.objectives
    print("completed optimizer step", result.step_number)

torch.save(train_step.state_dict(), "checkpoint.pt")
train_step.close()
```

The outer sequence passed to `plan_step()` and `train_step()` is the
accumulation-round dimension. One call runs both objective/backward rounds and
then performs exactly one optimizer update. Shapes, strides, dtypes, static
values, and outer structure must match the examples used during planning.

The objective's scalar loss for each accumulation round is returned in
`result.objectives`. `losses` above holds the most recent result and is
replaced on the next loop iteration. ShadowSpill validates and records this
explicit objective return during capture; it does not guess which model output
is a loss. Optional nondifferentiated objective metrics are documented in the
[frontend API](../python/api/frontend.md#inputs-objectives-and-partitioning).

An ordinary `train_step()` returns `StepResult` without collecting or resolving
a runtime trace. `DiagnosticsHandle.result()`, checkpoint operations, and
lifecycle close are explicit synchronous boundaries.

Use fast local storage for the reusable planning cache. `state_dict()` creates
an ordinary CPU checkpoint before `torch.save()` begins. The example closes
the callable and then exits; long-lived embedding processes should follow the
complete ownership order in [Errors, failures, and cleanup](../python/failures.md).
