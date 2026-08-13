"""Fresh-process CUDA check for Export-functionalized state mutation."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import Runtime, plan_forward


class _StatefulForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running", torch.zeros(64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.running.add_(value.sum(0))
        return self.running[8:56] * 0.5


def main() -> int:
    adapter = Path(sys.argv[1]).resolve()
    os.environ["SHADOWSPILL_PYTORCH_LIBRARY"] = str(adapter)
    with tempfile.TemporaryDirectory() as cache:
        os.environ["SHADOWSPILL_PROFILE_CACHE"] = cache
        torch.manual_seed(317)
        model = _StatefulForward().eval()
        reference = _StatefulForward().eval()
        reference.load_state_dict(model.state_dict())
        value = torch.randn(4, 64)
        runtime = Runtime(
            pools={
                "execution": device(
                    physical_capacity=2 << 30,
                    provider_headroom=512 << 20,
                ),
                "spill": pinned_host(capacity=1 << 30),
            },
            library_path=adapter,
        )
        planned = plan_forward(
            model,
            example_inputs=[value],
            runtime=runtime,
            execution="execution",
            spill="spill",
            partition="whole",
        )
        task = planned.plan_report.execution_plan.program.tasks[0]
        if len(task.mutations) != 1:
            raise AssertionError("buffer update was not lowered as a task mutation")
        diagnostics = planned.plan_report.diagnostics
        if len(diagnostics.task_stage_map) != 1 or len(diagnostics.unique_stages) != 1:
            raise AssertionError("mutation task diagnostics are incomplete")
        profile = diagnostics.unique_stages[0].graph_pairs[0].forward
        if (
            len(profile.semantic_mutations) != 1
            or profile.semantic_mutations[0].replacement_output_leaf is None
            or profile.replacement_transition_bytes
            != model.running.untyped_storage().nbytes()
            or profile.task_workspace_bytes
            != profile.workspace_charged_bytes + profile.replacement_transition_bytes
        ):
            raise AssertionError("functional-mutation diagnostics are inconsistent")
        for _ in range(3):
            actual = planned([value])
            expected = reference(value)
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
            del actual, expected
        state = planned.state_dict()
        torch.testing.assert_close(state["running"], reference.running)
        planned.close()
        runtime.close()
        if model.running.device.type != "cpu":
            raise AssertionError("close did not restore mutated buffer to the CPU")
        torch.testing.assert_close(model.running, reference.running)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
