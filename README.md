# ShadowSpill

ShadowSpill turns a fixed-shape PyTorch forward or training step into a
memory-budgeted callable: it decides what to keep on the device, what to spill
and fetch back, and what to recompute, then proves the answer fits the declared
pools before the step runs. PyTorch still executes every kernel.

## Installation

```bash
./scripts/setup.sh                                    # fresh checkout
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python" # existing environment
```

The script creates `.venv`, installs the supported PyTorch and device-backend
stack, builds the C library with its backends and the PyTorch adapter, installs
the mlops operation library and the training harness's dependencies, and
verifies the install.

## Minimal example

The trainer takes an ordinary model, objective and iterable. Enter the backend
before creating accelerator resources. Initialized CPU models retain their state.

```python
import torch
from torch import nn

from shadowspill.training import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import Constant

model = nn.Linear(16, 8)
data = [(torch.randn(64, 16), torch.randn(64, 8)) for _ in range(5)]

def objective(model, batch):
    inputs, targets = batch
    return (model(inputs) - targets).square().mean()

with ShadowSpill(execution_gib=2, spill_gib=2) as backend:
    with Trainer(
        model,
        objective=objective,
        optimizer=torch.optim.AdamW,
        schedules={"lr": Constant(3e-4)},
        backend=backend,
    ) as trainer:
        trainer.prepare(data[0])
        trainer.fit(data, steps=5, run_dir="runs/example")
```

Use `trainer.step(data)` in a custom loop, or `Forward` for inference.
The [training API](docs/python/api/training.md) covers microbatches, schedules,
evaluation, checkpointing and alternate PyTorch execution. The lower-level
[Python quickstart](docs/python/quickstart.md) exposes planning and runtime
ownership directly. The [benchmark quickstart](benchmarking/quickstart.md)
accepts a user experiment factory or a supplied text recipe, and the
[examples](docs/examples/README.md) show complete workflows. The
[distributed example](docs/examples/generic-training.md#run-a-distributed-example)
uses ordinary torchrun, explicit device/groups, per-rank startup traces, and
separate rank plus aggregate W&B runs.
For equivalent distributed planning problems, opt into
[verified symmetric planning](docs/python/api/distributed.md#verified-symmetric-planning)
with `Distributed(group, symmetric_planning=True)` to share CPU search work.

## Project structure

| Path | Purpose |
|---|---|
| `src/shadowspill/` | Installed Python package and PyTorch frontend |
| `csrc/` | The C library — planner, simulator, runtime — plus backends and the PyTorch adapter |
| `tests/` | Tests mirroring Python, C, integration, and tooling boundaries |
| `workloads/` | Optional models and recipes passed into generic APIs |
| `benchmarking/` | The quickstart tour, program collection, and planning evaluation |
| `qualification/` | Release gates: suite, numerical, performance, and remote |
| `training/` | Example text configs, launch scripts and run comparisons |
| `src/tools/` | Repository naming checks and sanitizer support |
| `reference/` | Readable reference implementations of the planner |
| `scripts/` | One-command environment setup |
| `docs/` | Architecture, Python, C, examples, and development guides |

## Documentation

[**docs/README.md**](docs/README.md) explains what the system does and links
every page once. The shortest routes from here:

| Topic | Start here |
|---|---|
| System architecture | [Architecture overview](docs/architecture/overview.md) |
| Python usage and API | [Python documentation](docs/python/README.md) |
| C components and APIs | [C documentation](docs/c/README.md) |
| Plan search | [Plan search](docs/architecture/search.md), [PressureFit](docs/architecture/pressurefit.md) |
| Physical admission | [Physical admission and offset handling](docs/architecture/physical-admission.md) |
| Plan and step diagnostics | [Diagnostics guides](docs/python/plan-report.md) |
| Serialized planning artifacts | [program and annotated-plan JSON](docs/python/planning-json.md) |
| Practical workflows | [Examples](docs/examples/README.md) |
| Errors and cleanup | [Errors, failures, and cleanup](docs/python/failures.md) |
| Repository development | [Development guide](docs/development/README.md) |
