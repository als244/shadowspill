# ShadowSpill

ShadowSpill turns a fixed-shape PyTorch forward or training step into a
memory-budgeted callable. It coordinates tensor spilling, fetching, and
recomputation while PyTorch continues to execute the numerical kernels.

## Installation

From a fresh checkout:

```bash
./scripts/setup.sh
```

The script creates `.venv`, installs the supported PyTorch and device-backend
stack, builds the C planner, simulator, runtime, backend, and PyTorch adapter,
installs the mlops operation library with its implementation providers, and
verifies the installation. To use an existing virtual or Conda
environment:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

## Minimal example

Initialize the runtime before constructing or loading model state. This lets
the runtime register its physical pools and calibrate their real transfer
routes before workload allocations claim host memory.

```python
from functools import partial

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import (
    Runtime,
    plan_step,
    import_model_state,
)

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

model = import_model_state(model, runtime=runtime, pool="spill")

train_step = plan_step(
    model,
    objective=lambda model, tokens, targets: model(
        tokens, labels=targets
    ).loss,
    opt=partial(torch.optim.AdamW, lr=3e-4),
    example_inputs=[[tokens_example, targets_example]],
    runtime=runtime,
    execution="device",
    spill="spill",
)

result = train_step([[tokens, targets]])
print("loss", result.objectives[0])

train_step.close()
```

One call performs one optimizer update. See the
[Python quickstart](docs/python/quickstart.md) for accumulation, checkpoints,
tracing, forward-only planning, and complete state lifecycle handling.

## Project structure

| Path | Purpose |
|---|---|
| `src/shadowspill/` | Installed Python package and PyTorch frontend |
| `csrc/` | The C library — planner, simulator, runtime — plus backends and the PyTorch adapter |
| `tests/` | Tests mirroring Python, C, integration, and tooling boundaries |
| `workloads/` | Model and data clients used by benchmarks and qualification |
| `benchmarking/` | Reusable Program collection and planning evaluation |
| `qualification/` | Numerical and performance release gates |
| `src/tools/` | Source-tree diagnostics and acceptance tooling |
| `reference/` | Executable reference implementations of the planner |
| `scripts/` | One-command environment setup |
| `docs/` | Architecture, Python, C, development, and investigation guides |

## Documentation

| Topic | Start here |
|---|---|
| System architecture | [Architecture overview](docs/architecture/overview.md) |
| Python usage and API | [Python documentation](docs/python/README.md) |
| C components and APIs | [C documentation](docs/c/README.md) |
| PressureFit planner | [PressureFit](docs/architecture/pressurefit.md) |
| Graph-pair construction | [Graph-pair construction](docs/architecture/graph-pair-construction.md) |
| Recomputation selection | [Recomputation selection](docs/architecture/recomputation-selection.md) |
| Physical admission | [Physical admission and offset handling](docs/architecture/physical-admission.md) |
| Plan and step diagnostics | [Diagnostics guides](docs/python/plan-report.md) |
| Serialized planning artifacts | [Program and annotated-plan JSON](docs/python/planning-json.md) |
| Practical workflows | [Examples](docs/examples/README.md) |
| Errors and cleanup | [Errors, failures, and cleanup](docs/python/failures.md) |
| Repository development | [Development guide](docs/development/README.md) |
| Root-cause records | [Engineering investigations](docs/investigations/README.md) |
