# ShadowSpill

ShadowSpill transparently plans recomputation, host offload, and prefetching
around ordinary PyTorch execution while enforcing an explicit physical memory
budget.

> **Status:** architecture extraction is in progress. The public surface below
> is the compatibility target; it is not yet available in the Phase 0 scaffold.

## Development installation

```bash
conda activate shadowspill
python -m pip install -e ".[dev]"
```

The release installer will select a qualified accelerator-specific PyTorch
wheel and build the version-pinned adapter with `python tools/install.py`.
Until that installer reaches its release gate, development uses the qualified
PyTorch already installed in the `shadowspill` environment.

## Planned training API

```python
from functools import partial

import torch

from shadowspill.pytorch import plan

train_step = plan(
    model,
    objective=lambda model, tokens, targets: model.loss(tokens, targets),
    opt=partial(torch.optim.AdamW, lr=3e-4, weight_decay=0.1),
    example_inputs=[
        [token_spec_0, target_spec_0],
        [token_spec_1, target_spec_1],
    ],
    device_budget=24 << 30,
    host_budget=64 << 30,
)

result = train_step(
    [
        [tokens_0, targets_0],
        [tokens_1, targets_1],
    ]
)
```

## Planned forward API

```python
from shadowspill.pytorch import forward_pass

run_forward = forward_pass(
    model,
    example_inputs=[token_spec, conditioning_spec, metadata],
    device_budget=24 << 30,
    host_budget=64 << 30,
)

outputs = run_forward([tokens, conditioning, metadata])
```

## Component map

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

- The simulator can run an explicit schedule without invoking the planner.
- The planner uses the simulator to evaluate schedules.
- The C runtime owns allocation, residency, transfers, and readiness without
  knowing about PyTorch.
- The PyTorch frontend captures and launches numerical work; it does not turn
  the runtime into a second model executor.
- Model implementations and optional `mlops` examples are qualification
  assets, not dependencies of the core package.

See [the architecture](docs/architecture.md),
[memory-budget semantics](docs/memory-budget-semantics.md), and
[the development plan](docs/development-plan.md).
