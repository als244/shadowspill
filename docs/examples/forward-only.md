# Forward-only execution

Given the imported `model` and open `runtime` from the [training
example](training-lifecycle.md), `plan_forward()` creates a fixed-shape
forward-only callable:

```python
import torch

from shadowspill.pytorch import plan_forward

example_tokens = torch.randn(8, 128)

run_forward = plan_forward(
    model,
    example_inputs=[example_tokens],
    runtime=runtime,
    execution="execution",
    spill="spill",
    planning_cachedir="artifacts/planning-cache",
)

output = run_forward([torch.randn(8, 128)])
print(output.shape)
run_forward.close()
```

Forward outputs use caller-owned dynamic leases because the caller may retain
them after another invocation. Release references when they are no longer
needed. `PlannedForward` also supports `profiler_annotations=True`.
