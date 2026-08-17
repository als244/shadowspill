"""Fresh-process public forward lifecycle through the production runtime."""

from __future__ import annotations

import ctypes
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    InputGuardError,
    Runtime,
    export_model_state,
    import_model_state,
    plan_forward,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.allocator import installed_allocator


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
    with tempfile.TemporaryDirectory() as cache:
        torch.manual_seed(31)
        model = _ForwardModel().eval()
        reference = _ForwardModel().eval()
        reference.load_state_dict(model.state_dict())
        inputs = torch.randn(4, 256)
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
        model = import_model_state(
            model,
            runtime=runtime,
            pool="spill",
            release_source=True,
        )
        parameter_ids = tuple(id(value) for value in model.parameters())
        planned = plan_forward(
            model,
            example_inputs=[inputs, 16],
            runtime=runtime,
            execution="execution",
            spill="spill",
            planning_cachedir=cache,
            profiling_metadata={"batch_size": 4, "width": 16},
        )
        if len(planned.plan_report.execution_plan.program.tasks) != 3:
            raise AssertionError("automatic partition did not retain three stages")
        plan_diagnostics = planned.plan_report.diagnostics
        if not plan_diagnostics.cache_artifacts:
            raise AssertionError("plan diagnostics omitted cache artifacts")
        if len(plan_diagnostics.profiling_metadata) != 1:
            raise AssertionError("plan diagnostics omitted profiling metadata")
        if (
            plan_diagnostics.measured_wall_time_ns
            + plan_diagnostics.unattributed_overhead_ns
            != plan_diagnostics.total_wall_time_ns
        ):
            raise AssertionError("plan diagnostic wall time does not reconcile")
        selected_task_diagnostics = tuple(
            item for item in plan_diagnostics.task_stage_map if item.selected
        )
        if tuple(item.execution_ordinal for item in selected_task_diagnostics) != tuple(
            range(len(selected_task_diagnostics))
        ):
            raise AssertionError(
                "forward diagnostics are not chronologically contiguous"
            )
        for item in selected_task_diagnostics:
            if not item.semantic_contract_digest or not item.compiled_layout_digest:
                raise AssertionError("forward task omitted lowering diagnostics")
        for stage in plan_diagnostics.unique_stages:
            profile = stage.graph_pairs[0].forward
            if not profile.semantic_roots or not profile.compiled_roots:
                raise AssertionError(
                    "forward contract omitted semantic/physical layout"
                )
            if not profile.allocation_contract_digest:
                raise AssertionError("forward contract omitted its allocation contract")
            if profile.semantic_contract_capture_ns <= 0:
                raise AssertionError(
                    "forward contract omitted contract extraction time"
                )
            if profile.physical_profile_wall_time_ns <= 0:
                raise AssertionError("forward contract omitted physical profiling time")
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
        if before_close.runtime.fetch_transfers == 0:
            raise AssertionError("public forward performed no real FETCH transfer")
        if before_close.cuda.device_allocations != 1:
            raise AssertionError("steady execution grew the conventional CUDA slab")
        if before_close.cuda.pinned_host_allocations != 1:
            raise AssertionError("steady execution grew pinned host memory")

        planned.close()
        planned.close()
        export_model_state(model, runtime=runtime, release_runtime=True)
        runtime.close()
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
