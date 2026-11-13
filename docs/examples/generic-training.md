# Generic training and inference

Build and measure an ordinary PyTorch model in a preparation process. Planning
then runs on CPU, and a fresh worker loads the resulting Plan. The worker does
not need the model class, objective callback, or compiler cache.

## Build and profile

```python
import torch
from torch import nn
from shadowspill import build_step, measure_program, apply_profiles, measure_communication

model = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 8))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)

def objective(model, features, *, targets):
    error = model(features) - targets
    # Four microbatches of 16 samples make one 64-sample update.
    return error.square().mean() / 4

program = build_step(
    model, objective=objective, optimizer=optimizer,
    example_args=(torch.randn(16, 16),),
    example_kwargs={"targets": torch.randn(16, 8)}, num_examples=4,
)
program.save("program.ssp")
profiles = measure_program(program)
profiles.save("profiles.ssp")
apply_profiles(program, profiles).save("profiled.ssp")
measure_communication(program=program).save("communication.ssp")
torch.save(model.state_dict(), "initial.pt")
```

`example_args` and `example_kwargs` describe the complete call, including nested
inputs and model-specific metadata. For differing shapes, use
`example_calls=[(args0, kwargs0), (args1, kwargs1)]` instead of one example and a
count. This fixes the microbatch calls in the Program; planning chooses their traversal.
`num_examples` counts calls, not individual samples.

## Plan on CPU

This can run in a separate process without importing PyTorch or opening pools.

```python
from shadowspill import ProfiledProgram, CommunicationInfo, MemoryBudgets, plan_program

plan = plan_program(
    ProfiledProgram.load("profiled.ssp"),
    memory_budgets=MemoryBudgets(execution_bytes=8 << 30, spill_bytes=16 << 30),
    communication_info=CommunicationInfo.load("communication.ssp"),
)
plan.save("training.ssp")
```

## Execute in a fresh worker

The example's CPU tensor dictionary initializes contiguous model tensors. Larger
applications can stream checkpoints into the same bounded destination buffers.

```python
import torch
from shadowspill import Plan, Trainer

initial = torch.load("initial.pt", map_location="cpu", weights_only=True)

def fill_tensor(chunk, destination):
    source = initial[chunk.name].reshape(-1)
    destination.copy_(source[chunk.offset:chunk.offset + destination.numel()])

def training_batches():
    while True:
        # One yielded item is one update containing four microbatch calls.
        yield {
            "args": [(torch.randn(16, 16),) for _ in range(4)],
            "kwargs": [{"targets": torch.randn(16, 8)} for _ in range(4)],
        }

with Trainer(Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)
    trainer.fit(training_batches(), steps=5, run_dir="runs/regression")
    trainer.save("runs/regression/checkpoint.ssp")
```

Construct Trainer before initializing a different device allocator in this worker.
Trainer creates the Executor, installs its allocator, initializes bounded pools,
and binds the saved task implementations. The example uses CPU inputs; the
Executor transfers them according to the plan.

For a custom loop, call `trainer.step(args=..., kwargs=...)` directly. To resume,
replace `initialize` with `load("runs/regression/checkpoint.ssp")` and set a larger
total `steps` target. This synthetic source generates new data; a real resumable
source can also save and restore its position.

For prediction or validation, build and plan a separate `build_forward` Program.
Pass that Plan as `evaluation_plan` to Trainer to share the current model tensors.
The [training reference](../python/api/training.md) covers evaluation, schedules,
checkpointing and the lower-level asynchronous Executor API. The
[distributed guide](../python/api/distributed.md) covers training on several
GPUs and the scope of its qualification.
