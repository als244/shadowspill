"""Fresh-process public forward lifecycle through the production runtime."""

from __future__ import annotations

import ctypes
import gc
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.errors import (
    InputGuardError,
)
from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import (
    Runtime,
    RuntimeConfigurationError,
    TensorRef,
    export_model_state,
    import_model_state,
    plan_forward,
    shared_input,
    shared_output,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics, ObjectSnapshot
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


class _ConsumerModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2.0 + 1.0


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


def _object_exists(runtime: Runtime, object_id: int) -> bool:
    snapshot = ObjectSnapshot()
    return int(
        runtime._installed.library.shadowspill_pytorch_object_snapshot(
            object_id, ctypes.byref(snapshot)
        )
    ) == 0


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
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
            library_path=adapter,
        )
        model = import_model_state(
            model,
            runtime=runtime,
            pool="spill",
            release_source=True,
        )
        consumer_model = import_model_state(
            _ConsumerModel().eval(),
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

        shared = plan_forward(
            model,
            example_inputs=[inputs, 16],
            runtime=runtime,
            execution="execution",
            spill="spill",
            planning_cachedir=cache,
            profiling_metadata={"batch_size": 4, "width": 16},
            shared_outputs=(shared_output("mean", retain_in="execution"),),
        )
        first = shared([inputs, 16])
        first_reference = first["mean"]
        if not isinstance(first_reference, TensorRef):
            raise AssertionError("declared shared output did not return TensorRef")
        shared_object_id = first_reference.object.object_id
        first_generation = first_reference.generation
        try:
            shared([inputs, 16])
        except RuntimeError as error:
            if "shared output slots remain owned" not in str(error):
                raise
        else:
            raise AssertionError("live shared output did not guard slot reuse")
        first_reference.close()
        second = shared([inputs, 16])
        second_reference = second["mean"]
        if not isinstance(second_reference, TensorRef):
            raise AssertionError("reused shared output slot lost its reference")
        if second_reference.object.object_id != shared_object_id:
            raise AssertionError("shared output recurrence changed logical identity")
        if second_reference.generation == first_generation:
            raise AssertionError("shared output recurrence did not replace generation")

        consumer = plan_forward(
            consumer_model,
            example_inputs=[
                shared_input(second_reference, require_in="execution")
            ],
            runtime=runtime,
            execution="execution",
            spill="spill",
            planning_cachedir=cache,
            profiling_metadata={"shared_input": "scalar_mean"},
        )
        peer_consumer = plan_forward(
            consumer_model,
            example_inputs=[
                shared_input(second_reference, require_in="execution")
            ],
            runtime=runtime,
            execution="execution",
            spill="spill",
            planning_cachedir=cache,
            profiling_metadata={"shared_input": "scalar_mean"},
        )
        before_consumer = _statistics()
        consumed = consumer([second_reference])
        expected_consumed = reference(inputs, 16)["mean"] * 2.0 + 1.0
        torch.testing.assert_close(
            consumed.cpu(), expected_consumed, rtol=2e-5, atol=2e-6
        )
        after_consumer = _statistics()
        if (
            after_consumer.runtime.fetch_transfers
            != before_consumer.runtime.fetch_transfers
        ):
            raise AssertionError("execution-resident shared input was fetched")

        second_reference.close()
        third = shared([inputs, 16])
        third_reference = third["mean"]
        if not isinstance(third_reference, TensorRef):
            raise AssertionError("recurrent shared output lost its reference")
        if third_reference.object.object_id != shared_object_id:
            raise AssertionError("recurrent shared output changed logical identity")
        if third_reference.generation == second_reference.generation:
            raise AssertionError("recurrent shared output did not replace its value")
        submitted_consumer = consumer.submit([third_reference])
        submitted_peer = peer_consumer.submit([third_reference])
        if submitted_consumer.resolved or submitted_peer.resolved:
            raise AssertionError("concurrent forwards synchronized during dispatch")
        repeated = submitted_consumer.result()
        peer_repeated = submitted_peer.result()
        torch.testing.assert_close(
            repeated.cpu(), expected_consumed, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            peer_repeated.cpu(), expected_consumed, rtol=2e-5, atol=2e-6
        )

        consumer.close()
        peer_consumer.close()
        shared.close()
        if not _object_exists(runtime, shared_object_id):
            raise AssertionError("callable close destroyed a public shared output")
        third_reference.close()
        if _object_exists(runtime, shared_object_id):
            raise AssertionError("final shared-output owner did not reclaim object")

        export_model_state(model, runtime=runtime, release_runtime=True)
        export_model_state(
            consumer_model,
            runtime=runtime,
            release_runtime=True,
        )
        try:
            runtime.close()
        except RuntimeConfigurationError as error:
            if "caller-owned device outputs" not in str(error):
                raise
        else:
            raise AssertionError("runtime close invalidated live caller outputs")

        retained_slice = retained["slice"].cpu()
        del consumed
        del first
        del repeated
        del peer_repeated
        del submitted_consumer
        del submitted_peer
        del replay
        del retained
        del second
        del third
        del zero
        gc.collect()
        torch.cuda.synchronize()
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
            retained_slice,
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
