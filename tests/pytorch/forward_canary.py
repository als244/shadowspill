"""Fresh-process public forward lifecycle through the production runtime."""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.pytorch import InputGuardError, forward_pass
from shadowspill.pytorch._abi import AdapterStatistics
from shadowspill.pytorch._allocator import installed_allocator


class _ForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(3)])
        self.tied = self.layers[0].weight

    def forward(self, value: torch.Tensor, width: int) -> dict[str, torch.Tensor]:
        for layer in self.layers:
            value = torch.relu(layer(value))
        return {"slice": value[:, :width], "mean": value.mean()}


def _statistics() -> AdapterStatistics:
    installed = installed_allocator()
    if installed is None:
        raise AssertionError("public forward did not install the allocator")
    result = AdapterStatistics()
    status = int(
        installed.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result))
    )
    if status != 0:
        raise AssertionError(f"statistics failed with status {status}")
    return result


def main() -> int:
    adapter = Path(sys.argv[1]).resolve()
    os.environ["SHADOWSPILL_PYTORCH_LIBRARY"] = str(adapter)
    with tempfile.TemporaryDirectory() as cache:
        os.environ["SHADOWSPILL_PROFILE_CACHE"] = cache
        torch.manual_seed(31)
        model = _ForwardModel().eval()
        reference = _ForwardModel().eval()
        reference.load_state_dict(model.state_dict())
        parameter_ids = tuple(id(value) for value in model.parameters())
        inputs = torch.randn(4, 256)
        planned = forward_pass(
            model,
            example_inputs=[inputs, 16],
            device_budget=2 << 30,
            host_budget=1 << 30,
        )
        if len(planned.plan_report.execution_plan.program.tasks) != 3:
            raise AssertionError("automatic partition did not retain three stages")
        plan_diagnostics = planned.plan_report.diagnostics
        if (
            plan_diagnostics.measured_wall_time_ns
            + plan_diagnostics.unattributed_overhead_ns
            != plan_diagnostics.total_wall_time_ns
        ):
            raise AssertionError("plan diagnostic wall time does not reconcile")
        if tuple(id(value) for value in model.parameters()) != parameter_ids:
            raise AssertionError("planning replaced a Parameter object")
        if (
            model.tied.untyped_storage()._cdata
            != model.layers[0].weight.untyped_storage()._cdata
        ):
            raise AssertionError("planning broke tied parameter storage")
        if any(value.device.type != "cuda" for value in model.parameters()):
            raise AssertionError("active planned model does not retain CUDA identity")

        retained: dict[str, torch.Tensor] | None = None
        for _ in range(3):
            retained = planned([inputs, 16])
            expected = reference(inputs, 16)
            torch.testing.assert_close(
                retained["slice"].cpu(), expected["slice"], rtol=2e-5, atol=2e-6
            )
            torch.testing.assert_close(
                retained["mean"].cpu(), expected["mean"], rtol=2e-5, atol=2e-6
            )
        if retained is None:
            raise AssertionError("forward loop produced no output")

        saved = planned.state_dict()
        replacement = {name: torch.zeros_like(value) for name, value in saved.items()}
        planned.load_state_dict(replacement)
        zero = planned([inputs, 16])
        if torch.count_nonzero(zero["slice"]).item() != 0:
            raise AssertionError(
                "load_state_dict did not update host-authoritative state"
            )
        planned.load_state_dict(saved)
        replay = planned([inputs, 16])
        torch.testing.assert_close(
            replay["slice"].cpu(), reference(inputs, 16)["slice"], rtol=2e-5, atol=2e-6
        )

        try:
            planned([inputs, 15])
        except InputGuardError:
            pass
        else:
            raise AssertionError("static metadata guard accepted a changed value")

        before_close = _statistics()
        if before_close.runtime.transfers_to_device == 0:
            raise AssertionError("public forward performed no real H2D transfer")
        if before_close.cuda.device_allocations != 1:
            raise AssertionError("steady execution grew the conventional CUDA slab")
        if before_close.cuda.pinned_host_allocations != 1:
            raise AssertionError("steady execution grew pinned host memory")

        planned.close()
        planned.close()
        if tuple(id(value) for value in model.parameters()) != parameter_ids:
            raise AssertionError("close replaced a Parameter object")
        if any(value.device.type != "cpu" for value in model.parameters()):
            raise AssertionError("close did not restore CPU model state")
        if (
            model.tied.untyped_storage()._cdata
            != model.layers[0].weight.untyped_storage()._cdata
        ):
            raise AssertionError("close broke tied parameter storage")
        if model.training:
            raise AssertionError("forward planning changed the model mode")
        torch.testing.assert_close(
            retained["slice"].cpu(),
            reference(inputs, 16)["slice"],
            rtol=2e-5,
            atol=2e-6,
        )
        try:
            planned([inputs, 16])
        except RuntimeError as error:
            if "closed" not in str(error):
                raise
        else:
            raise AssertionError("closed callable accepted another invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
