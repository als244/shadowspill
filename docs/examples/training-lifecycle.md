# Training loop

This example is a complete fixed-shape accumulated training program. It
creates the runtime before any accelerator allocation, relocates persistent
model state into the spill pool, plans two accumulation rounds, runs a normal
training loop, and writes a checkpoint.

```python
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    plan_step,
    relocate_model_state,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def objective(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    error = model(features) - targets
    return error.square().mean()


def batch(rows: int) -> list[torch.Tensor]:
    return [torch.randn(rows, 128), torch.randn(rows, 128)]


def training_batches(steps: int):
    for _ in range(steps):
        yield [batch(4), batch(7)]


def main() -> None:
    torch.manual_seed(17)
    model = TinyModel()
    examples = [batch(4), batch(7)]

    runtime = Runtime(
        pools={
            "execution": device(physical_capacity=4 << 30),
            "spill": pinned_host(capacity=2 << 30),
        }
    )
    model = relocate_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )

    planning_cache = Path("artifacts/planning-cache")
    checkpoint_directory = Path("artifacts/checkpoints")

    train_step = plan_step(
        model,
        objective=objective,
        opt=partial(torch.optim.AdamW, lr=3e-4, foreach=False),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        planning_cachedir=planning_cache,
        profiling_metadata=(
            {"batch_rows": 4},
            {"batch_rows": 7},
        ),
    )

    for step_inputs in training_batches(10):
        result = train_step(step_inputs)
        round_losses = result.objectives
        print("completed optimizer step", result.step_number)

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = train_step.state_dict()
    checkpoint_path = checkpoint_directory / "step-10.pt"
    torch.save(checkpoint, checkpoint_path)
    train_step.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    train_step.close()


if __name__ == "__main__":
    main()
```

The outer sequence passed to `plan_step()` and `train_step()` is the
accumulation-round dimension. One call runs both objective/backward rounds and
then performs exactly one optimizer update. Shapes, strides, dtypes, static
values, and outer structure must match the examples used during planning.

The objective's scalar loss for each accumulation round is returned in
`result.objectives`. `round_losses` above holds the most recent result and is
replaced on the next loop iteration. ShadowSpill validates and records this
explicit objective return during capture; it does not guess which model output
is a loss. Optional nondifferentiated objective metrics are documented in the
[frontend API](../python/api/frontend.md#inputs-objectives-and-partitioning).

An ordinary `train_step()` returns `StepResult` without collecting or resolving
a runtime trace. `DiagnosticsHandle.result()`, checkpoint operations, and
lifecycle close are explicit synchronous boundaries.

The planning cache is persistent and reusable across matching planning calls.
Place it on fast local storage for a real workload; it is separate from the
checkpoint directory and is not temporary scratch.

`state_dict()` synchronously creates an ordinary CPU copy. Once it returns,
writing that copy to storage can proceed independently while later training
steps use the runtime-owned state.

The ownership hierarchy is:

```text
Runtime
└── relocated persistent model state
    └── planned callable
```

The example closes the callable explicitly and then ends the process. Native
process-exit cleanup stops the worker and closes the pool backends. An
embedding application that keeps the process alive and explicitly closes its
`Runtime` must first release every relocated-state owner; that complete close
contract is documented in [Errors, failures, and cleanup](../python/failures.md).
