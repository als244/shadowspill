# Forward-only execution

Build and profile a fixed-shape forward computation, save its Plan, then load it
in a fresh execution worker. No optimizer or training objective is needed. The
[training example](generic-training.md) shows the same phases for updates.

## Prepare and save

The current GPU builder finalizes kernels on a representative GPU. Build and
profiling use ordinary GPU allocations; planning opens no execution pools.

```python
import torch
from torch import nn
from shadowspill import (
    CommunicationInfo, MemoryBudgets, apply_profiles, build_forward,
    measure_communication, measure_program, plan_program,
)

model = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 8)).eval()
program = build_forward(model, example_args=(torch.randn(8, 16),))
profiled = apply_profiles(program, measure_program(program))
profiled.save("forward-profiled.ssp")
communication = measure_communication()
communication.save("communication.ssp")
plan = plan_program(
    profiled,
    memory_budgets=MemoryBudgets(execution_bytes=2 << 30, spill_bytes=64 << 20),
    communication_info=communication,
)
plan.save("forward.ssp")
torch.save(model.state_dict(), "initial.pt")
```

The saved ProfiledProgram can also be loaded and planned in a separate CPU-only
process. `CommunicationInfo` may contain explicit rates instead of calibration
results; the [budget sweep example](reusable-planning.md) covers that workflow.

## Run in a fresh worker

```python
import torch
from shadowspill import Executor, Plan

initial = torch.load("initial.pt", map_location="cpu", weights_only=True)

def fill_tensor(chunk, destination):
    source = initial[chunk.name].reshape(-1)
    destination.copy_(source[chunk.offset:chunk.offset + destination.numel()])

with Executor(Plan.load("forward.ssp")) as executor:
    executor.initialize(fill_tensor)
    pending = executor.run(args=[(torch.randn(8, 16),)])
    # The CPU may do other work here. result() waits for completion.
    output = pending.result()
    prediction = output.cpu()
    print(prediction.shape)
    del pending, output
```

`run` takes a sequence of per-example positional tuples and keyword mappings;
a forward Plan has one example. Its result has the model's ordinary output
structure. The next call waits for the preceding invocation automatically.
Returned tensors remain valid across calls; release device outputs before closing
Executor so its allocator can shut down. CPU copies can outlive the worker.

`forward_fn=lambda model, *args, **kwargs: model.loss(*args, **kwargs)` selects
another model method while preserving registered parameter names. Automatic
partitioning uses those module paths. `partition="whole"` compiles one complete
forward task; a [custom partition policy](custom-partitioning.md) can define
other boundaries.

For evaluation during training, save a forward Plan with compatible parameter
names/layouts and pass it as `evaluation_plan` to Trainer. Both plans share the
current model tensors and one admitted execution workspace; see
[shared evaluation](../python/api/training.md#evaluation).
`run(..., profiler_annotations=True)` optionally adds semantic task/transfer
ranges for an external profiler. See the
[saved-Plan capture tool](../../benchmarking/quickstart.md#nsight-captures-of-saved-winners).
