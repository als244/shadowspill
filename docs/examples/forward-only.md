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
    artifact_store="artifacts/store",
)

output = run_forward([torch.randn(8, 128)])
print(output.shape)
run_forward.close()
```

Forward outputs use caller-owned dynamic leases because the caller may retain
them after another invocation. Release references when they are no longer
needed. Each call and each `submit()` on a `PlannedForward` also accepts
`profiler_annotations=True` and `runtime_trace=True`.

A traced call is timed task by task on the device's own clock and resolves
against the plan that predicted it, the same way a training step's trace
does. A training step carries its handle on the `StepResult` it returns; a
forward call returns the model's own output, so its handle is on the
callable:

```python
output = run_forward([torch.randn(8, 128)], runtime_trace=True)
step = run_forward.diagnostics.result()
print(step.summary.real_selected_span_seconds)
```

Resolve the handle before the next call. `mark_cycle_end()` and
`invocation_timings()` read the always-on invocation timeline, as they do on
a planned training step.
