# ShadowSpill

ShadowSpill plans transparent spilling, fetching, and recomputation around
ordinary PyTorch execution while enforcing explicit execution-device and spill
memory caps.

> **Status:** fixed-shape forward and accumulated-training callables run end to
> end through the CUDA-device and registered pinned-host runtime. Model-scale
> correctness, planning latency, and retained-throughput qualification remain
> active release gates.

## Install

```bash
./scripts/setup.sh
```

This command creates a local `.venv`, installs PyTorch 2.13 with the machine's
CUDA backend, builds ShadowSpill, installs development dependencies, and
verifies the GPU and every compiled component. It uses
[`uv`](https://docs.astral.sh/uv/) and bootstraps an isolated copy when needed.
To populate an existing virtual or Conda environment instead:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

The installed package owns the runtime, PyTorch adapter, simulator, and
planner libraries. No library-path environment variables or Python
monkeypatching are required.

## Minimal training example

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

model = relocate_model_state(
    model,
    runtime=runtime,
    pool="spill",
    release_source=True,
)

train_step = plan_step(
    model,
    objective=lambda model, tokens, targets: model(tokens, labels=targets).loss,
    opt=partial(torch.optim.AdamW, lr=3e-4),
    example_inputs=[
        [token_spec_0, target_spec_0],
        [token_spec_1, target_spec_1],
    ],
    runtime=runtime,
    execution="device",
    spill="spill",
    planning_cachedir="/local-fast-storage/shadowspill-cache",
)

result = train_step(
    [
        [tokens_0, targets_0],
        [tokens_1, targets_1],
    ]
)

train_step.close()
model = externalize_model_state(
    model,
    runtime=runtime,
    release_runtime=True,
)
runtime.close()
```

The outer input sequence fixes the accumulation-round count. ShadowSpill runs
one forward/objective/backward contribution per inner sequence and one
optimizer update per call; it does not divide accumulated gradients.

The [Python quickstart](docs/python/quickstart.md) covers forward execution,
profiling metadata, opt-in tracing, checkpoints, and complete lifecycle
handling. The [Python API reference](docs/python/README.md) documents every
exported frontend and framework-neutral value.

## Learn the architecture

The [architecture overview](docs/architecture/overview.md) provides the
ordered reading path from capture through execution. It defines the artifact
ladder, component ownership, correctness invariants, supported scope, and one
complete logical-object walkthrough.

- [Python documentation](docs/python/README.md)
- [C API documentation](docs/c/README.md)
- [Development guide](docs/development/README.md)
- [Engineering investigations](docs/investigations/README.md)

Investigations preserve non-normative root-cause evidence. Architecture and
API pages define current behavior.

## Repository layout

```text
src/shadowspill/   installed Python package
csrc/              compiled planner, simulator, runtime, backends, and adapter
tests/             tests mirroring Python, C, integration, and tooling boundaries
workloads/         model and data clients used by benchmarks and qualification
benchmarking/      reusable Program collection and planning-frontier evaluation
qualification/     thin numerical and performance release gates
src/tools/         reusable source-tree diagnostics and acceptance tooling
docs/              architecture, Python, C, development, and investigations
```

The root build file only orders compiled components. Each component owns its
sources, public headers, and build declaration under `csrc/`; see the
[compiled-component guide](csrc/README.md).
