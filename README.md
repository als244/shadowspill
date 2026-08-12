# ShadowSpill

ShadowSpill plans transparent spilling, fetching, and recomputation around
ordinary PyTorch execution while enforcing an explicit physical-memory cap.

> **Status:** forward and accumulated-training callables run end to end through
> the slab runtime. Model-scale correctness, planning-latency, and retained
> throughput qualification remain in progress.

## Install

```bash
python -m pip install -e ".[dev]"
```

Development and qualification use the PyTorch build installed in the
`shadowspill` Conda environment. The release installer remains a later release
gate.

## Training

```python
from functools import partial

import torch

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import Runtime, plan_step

runtime = Runtime(
    pools={
        "device": device(physical_capacity=24 << 30),
        "spill": pinned_host(capacity=64 << 30),
    }
)

train_step = plan_step(
    model,
    objective=lambda model, tokens, targets: model.loss(tokens, targets),
    opt=partial(torch.optim.AdamW, lr=3e-4, weight_decay=0.1),
    example_inputs=[
        [token_spec_0, target_spec_0],
        [token_spec_1, target_spec_1],
    ],
    runtime=runtime,
    execution="device",
    spill="spill",
)

result = train_step(
    [
        [tokens_0, targets_0],
        [tokens_1, targets_1],
    ]
)

train_step.close()
runtime.close()
```

The outer input sequence is the fixed gradient-accumulation count. ShadowSpill
runs one forward/objective/backward contribution per inner sequence and one
optimizer update per call. It does not divide accumulated gradients.

## Forward

```python
from shadowspill.pytorch import plan_forward

run_forward = plan_forward(
    model,
    example_inputs=[token_spec, conditioning_spec, metadata],
    runtime=runtime,
    execution="device",
    spill="spill",
)

outputs = run_forward([tokens, conditioning, metadata])
```

`execution_device=None` uses PyTorch's current accelerator device. Passing an
explicit ordinal or `torch.device` selects it and must match the chosen
execution pool.

## Components

```text
IR ──────────────► Simulator
│                       ▲
└────► Planner ─────────┘
          │
          ▼
    ExecutionPlan ─────► Runtime
          ▲                ▲
          └──── PyTorch ───┘
```

- The simulator evaluates explicit schedules without invoking the planner.
- PressureFit uses the simulator to select a schedule.
- The neutral C runtime owns memory pools, leases, residency, routes,
  transfers, readiness, and failure propagation.
- PyTorch captures and launches numerical tasks; ShadowSpill is not a second
  model executor.
- Provider-specific pools, routes, events, and profiling live behind backend
  interfaces. The initial provider uses an accelerator execution pool and a
  pinned-memory spill pool.
- Pure-PyTorch and optional `mlops` models are qualification clients, never
  dependencies of the core.

See [the architecture](docs/architecture.md),
[the PyTorch API](docs/pytorch-frontend.md),
[memory-budget semantics](docs/memory-budget-semantics.md), and the
[development plan](docs/development-plan.md).
