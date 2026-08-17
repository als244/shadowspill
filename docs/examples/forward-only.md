# Forward-only execution

`plan_forward()` uses the same runtime, relocation, partitioning, profiling,
PressureFit, and physical-admission pipeline without objective, backward, or
optimizer tasks.

```python
from __future__ import annotations

import torch
import torch.nn as nn

from shadowspill.pytorch import plan_forward


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(128, 128, bias=False)

    def forward(self, tokens: torch.Tensor, width: int) -> torch.Tensor:
        return torch.relu(self.projection(tokens))[:, :width]


example_tokens = torch.randn(8, 128)

run_forward = plan_forward(
    model,
    example_inputs=[example_tokens, 32],
    runtime=runtime,
    execution="execution",
    spill="spill",
    planning_cachedir=planning_cache,
    profiling_metadata={"batch_rows": 8, "output_width": 32},
)

output = run_forward([torch.randn(8, 128), 32])
print(output.shape)
print(run_forward.plan_report.predicted_makespan_ns)
run_forward.close()
```

Here `model` is already the value returned by `relocate_model_state()`, and
`runtime` is the open owner created as in the [training lifecycle
example](training-lifecycle.md). The integer `width` is static captured
metadata; changing it at execution time raises `InputGuardError`.

Forward outputs use caller-owned dynamic leases because the caller may retain
them after another invocation. Release references when they are no longer
needed so subsequent calls do not retain unnecessary execution-pool capacity.
`PlannedForward` supports `profiler_annotations=True`; detailed
`runtime_trace=True` step diagnostics are a training-callable facility.
