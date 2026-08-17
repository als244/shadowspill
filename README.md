# ShadowSpill

ShadowSpill turns a fixed-shape PyTorch forward or training step into a
memory-budgeted callable. It coordinates tensor spilling, fetching, and
recomputation while PyTorch continues to execute the numerical kernels.

## Installation

From a fresh checkout:

```bash
./scripts/setup.sh
```

The script creates `.venv`, installs the supported PyTorch and CUDA stack,
builds the C planner, simulator, runtime, backend, and PyTorch adapter, and
verifies the installation. To use an existing virtual or Conda environment:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

## Minimal example

```python
from functools import partial

import torch

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    externalize_model_state,
    plan_step,
    relocate_model_state,
)

runtime = Runtime(
    pools={
        "device": device(physical_capacity=24 << 30),
        "spill": pinned_host(capacity=64 << 30),
    }
)

model = relocate_model_state(model, runtime=runtime, pool="spill")

train_step = plan_step(
    model,
    objective=lambda model, tokens, targets: model(
        tokens, labels=targets
    ).loss,
    opt=partial(torch.optim.AdamW, lr=3e-4),
    example_inputs=[
        [tokens_example_0, targets_example_0],
        [tokens_example_1, targets_example_1],
    ],
    runtime=runtime,
    execution="device",
    spill="spill",
    planning_cachedir="/local-fast-storage/shadowspill-planning",
)

result = train_step(
    [
        [tokens_0, targets_0],
        [tokens_1, targets_1],
    ]
)

train_step.close()
model = externalize_model_state(model, runtime=runtime, release_runtime=True)
runtime.close()
```

The outer input sequence defines the accumulation rounds; one call performs
one optimizer update. See the [Python quickstart](docs/python/quickstart.md)
for checkpoints, tracing, forward-only planning, and complete state lifecycle
handling.

## Project structure

| Path | Purpose |
|---|---|
| `src/shadowspill/` | Installed Python package and PyTorch frontend |
| `csrc/` | C planner, simulator, runtime, backends, and PyTorch adapter |
| `tests/` | Tests mirroring Python, C, integration, and tooling boundaries |
| `workloads/` | Model and data clients used by benchmarks and qualification |
| `benchmarking/` | Reusable Program collection and planning evaluation |
| `qualification/` | Numerical and performance release gates |
| `src/tools/` | Source-tree diagnostics and acceptance tooling |
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
| Repository development | [Development guide](docs/development/README.md) |
| Root-cause records | [Engineering investigations](docs/investigations/README.md) |
