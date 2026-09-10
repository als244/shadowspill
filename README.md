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
the mlops operation library, and verifies it.

## Minimal example

Initialize the runtime before model state exists, so its pools and routes are
ready first. Planning declares what exists: `optimizer_state_init` says what
optimizer state starts at, `hyperparams` names what a step may change.

```python
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
    optimizer=torch.optim.AdamW,
    hyperparams=("lr",),
    optimizer_state_init=lambda name, tensor, parameter: tensor.zero_(),
    example_inputs=[[tokens_example, targets_example]],
    runtime=runtime,
    execution="device",
    spill="spill",
)

result = train_step([[tokens, targets]], hyperparams={"lr": 3e-4})
print("loss", result.objectives[0])

train_step.close()
```

One call performs one optimizer update. The
[Python quickstart](docs/python/quickstart.md) covers accumulation, checkpoints,
tracing, forward-only planning, and state lifecycle; the [quickstart
script](benchmarking/quickstart.md) runs one model end to end, and the
[examples](docs/examples/README.md) are complete workflows.

## Project structure

| Path | Purpose |
|---|---|
| `src/shadowspill/` | Installed Python package and PyTorch frontend |
| `csrc/` | The C library — planner, simulator, runtime — plus backends and the PyTorch adapter |
| `tests/` | Tests mirroring Python, C, integration, and tooling boundaries |
| `workloads/` | Model and data clients used by benchmarks and qualification |
| `benchmarking/` | The quickstart tour, program collection, and planning evaluation |
| `qualification/` | Numerical and performance release gates |
| `src/tools/` | Source-tree diagnostics and acceptance tooling |
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
