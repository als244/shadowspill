# Generic training and inference

This example uses ordinary PyTorch modules and nested data, without a workload
registry. The summed objective is normalized by all target elements in the
update. The same functions work with a PyTorch backend or planned execution.

```python
import torch
from torch import nn

from shadowspill.training import Forward, Trainer
from shadowspill.training.backends import PyTorch, ShadowSpill
from shadowspill.training.schedules import Constant

model = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 8))
source = [
    {"inputs": torch.randn(64, 16), "target": torch.randn(64, 8)}
    for _ in range(5)
]

def objective(model, data):
    error = model(data["inputs"]) - data["target"]
    return error.square().sum(), {"absolute_error": error.abs().sum().detach()}

def split(data):
    scale = 1.0 / data["target"].numel()
    for start in range(0, len(data["inputs"]), 16):
        yield {key: value[start:start + 16] for key, value in data.items()}, scale

with ShadowSpill(execution_gib=8, spill_gib=16) as backend:
    with Trainer(
        model, objective=objective, optimizer=torch.optim.AdamW,
        optimizer_args={"weight_decay": 0.01}, schedules={"lr": Constant(3e-4)},
        microbatches=split, backend=backend,
    ) as trainer:
        trainer.prepare(source[0])
        for data in source:
            result = trainer.step(data)
            print(result.step, result.loss)
        with Forward(trainer.model, backend=backend) as predict:
            predict.prepare(source[0]["inputs"][:16])
            prediction = predict(source[0]["inputs"][:16])
            predict.synchronize()
            print(prediction.shape)
            del prediction
        trainer.save("runs/regression/step_00000005")
```

Replace the backend context with `PyTorch(compile=False, device="cpu")` for
eager reference execution, or `PyTorch(compile=True)` for ordinary compiled
execution. Initialized weights are retained; rebuild or copy the starting
model when comparing backends. A prepared plan guards its input shapes.

For a managed loop, replace the custom loop with
`trainer.fit(source, steps=5, run_dir="runs/regression")`. Optional evaluation,
logging and checkpoint cadence are described in the
[training reference](../python/api/training.md).

To search alternative microbatch sizes, provide a named mapping of splitting
functions to the ShadowSpill backend. All candidates represent the same
normalized update. The generic [benchmark quickstart](../../benchmarking/quickstart.md)
also accepts an experiment factory and reports any caller-named unit, such as
images or samples per second.

## Trace startup without updating AdamW state

For a ShadowSpill run using `mlops.optim.AdamW`, pass
`startup_diagnostics=True` to `trainer.fit`. This executes a warmup and a traced
step at zero LR, then begins ordinary training at update 1 with the same data.
Diagnostics are separate from W&B and training history. Each rank saves its own
trace and occupancy pages. See [startup diagnostics](../python/api/training.md#startup-diagnostics)
for the optimizer contract, state handling, and artifact paths.


## Run a distributed example

From the repository root, with MLOps and the training extras installed:

```bash
python -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29500 \
  --log-dir runs/dp-example/processes --tee 3 \
  -m workloads.recipes.distributed_training --run-dir runs/dp-example \
  --steps 10 --wandb-project my-project
```

[The runnable example](../../workloads/recipes/distributed_training.py) supplies
an ordinary regression model and source to the same generic Trainer. It performs
CPU host-budget admission before creating NCCL, shards optimizer state, records
startup traces without updating state, logs each rank plus the aggregate, and
writes a collective checkpoint. Omit `--wandb-project` for console/JSONL only,
or use `--wandb-mode offline` for local W&B artifacts. Budgets are per rank.

`torchrun` already supplies supervision and per-rank stdout/stderr capture here;
no separate ShadowSpill process launcher is required. The reports live under
`runs/dp-example/rank-NNNNN/` and `runs/dp-example/aggregate/`.
