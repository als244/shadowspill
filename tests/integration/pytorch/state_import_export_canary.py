"""Fresh-process persistent-state ownership and plan-adoption canary."""

from __future__ import annotations

import ctypes
import gc
import sys
import tempfile
import weakref
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import (
    Runtime,
    export_model_state,
    export_optimizer_state,
    import_model_state,
    import_optimizer_state,
    plan_forward,
)
from shadowspill.pytorch.runtime_adapter.abi import (
    AdapterStatistics,
    ObjectSnapshot,
    runtime_library,
)
from shadowspill.pytorch.state.storage import persistent_state


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(32, 32, bias=False)
        self.tied = self.projection.weight
        self.register_buffer(
            "weight_view", self.projection.weight.detach().view(16, 64)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.projection(value))


def _statistics(runtime: Runtime) -> AdapterStatistics:
    result = AdapterStatistics()
    status = int(
        runtime._installed.library.shadowspill_pytorch_allocator_statistics(
            ctypes.byref(result)
        )
    )
    if status != 0:
        raise AssertionError(f"runtime statistics failed with status {status}")
    return result


def _snapshot(runtime: Runtime, object_id: int) -> ObjectSnapshot:
    result = ObjectSnapshot()
    status = int(
        runtime_library().shadowspill_object_snapshot(
            runtime._runtime_handle,
            object_id, ctypes.byref(result)
        )
    )
    if status != 0:
        raise AssertionError(f"object snapshot failed with status {status}")
    return result


def main() -> int:
    adapter = Path(sys.argv[1]).resolve()
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
    torch.manual_seed(811)
    model = _Model().eval()
    reference = _Model().eval()
    reference.load_state_dict(model.state_dict())
    source_parameter_id = id(model.projection.weight)
    source_storage_id = int(model.projection.weight.untyped_storage()._cdata)
    source_pointer = model.projection.weight.untyped_storage().data_ptr()
    before = _statistics(runtime)

    copied_model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=False,
    )
    copied = persistent_state(runtime, copied_model)
    if copied is None or len(copied.storages) != 1:
        raise AssertionError("tied model state did not become one persistent object")
    if copied_model is model:
        raise AssertionError("non-consuming import returned its input model")
    if copied.source_owner is not model:
        raise AssertionError("non-consuming import did not retain its source owner")
    if id(copied_model.projection.weight) == source_parameter_id:
        raise AssertionError("non-consuming import reused its input Parameter")
    if model.projection.weight.untyped_storage().data_ptr() != source_pointer:
        raise AssertionError("non-consuming import replaced its CPU source")
    if copied_model.projection.weight.untyped_storage().data_ptr() != (
        copied.storages[0].pool_pointer
    ):
        raise AssertionError("imported copy does not point into spill memory")
    if (
        copied_model.tied is not copied_model.projection.weight
        or copied_model.weight_view.untyped_storage()._cdata
        != copied_model.projection.weight.untyped_storage()._cdata
    ):
        raise AssertionError("imported copy did not preserve ties and views")
    torch.testing.assert_close(
        copied_model.projection.weight,
        reference.projection.weight,
        rtol=0,
        atol=0,
    )
    copied_snapshot = _snapshot(runtime, copied.storages[0].current_object_id)
    if int(copied_snapshot.spill_pointer or 0) == source_pointer:
        raise AssertionError("runtime spill lease aliases the external CPU source")
    after_copy = _statistics(runtime)
    if after_copy.runtime.registered_objects != before.runtime.registered_objects + 1:
        raise AssertionError("non-consuming import registered the wrong object count")

    export_model_state(copied_model, runtime=runtime, release_runtime=True)
    del copied
    source_reference = weakref.ref(model)
    imported_model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
    )
    imported = persistent_state(runtime, imported_model)
    if imported is None:
        raise AssertionError("source-releasing import has no persistent state")
    if imported.source_owner is not None:
        raise AssertionError("source-releasing import retained its input model")
    if (
        id(model.projection.weight) != source_parameter_id
        or int(model.projection.weight.untyped_storage()._cdata) != source_storage_id
        or model.projection.weight.untyped_storage().data_ptr() != source_pointer
    ):
        raise AssertionError("source-releasing import mutated its input model")
    del model
    gc.collect()
    if source_reference() is not None:
        raise AssertionError("source-releasing import retained the source model")
    record = imported.storages[0]
    if (
        imported_model.projection.weight.untyped_storage().data_ptr()
        != record.pool_pointer
    ):
        raise AssertionError("returned model does not point into its spill lease")
    if (
        imported_model.tied is not imported_model.projection.weight
        or imported_model.weight_view.untyped_storage()._cdata
        != imported_model.projection.weight.untyped_storage()._cdata
    ):
        raise AssertionError("source-releasing import did not preserve ties and views")

    value = torch.randn(3, 32)
    persistent_id = record.persistent_object_id
    spill_pointer = record.pool_pointer
    spill_bytes_before_plan = int(_statistics(runtime).runtime.spill_allocated_bytes)
    with tempfile.TemporaryDirectory() as cache:
        planned = plan_forward(
            imported_model,
            example_inputs=[value],
            runtime=runtime,
            execution="execution",
            spill="spill",
            artifact_store_dir=cache,
            verbose=False,
        )
        adopted = _snapshot(runtime, record.current_object_id)
        if int(adopted.spill_pointer or 0) != spill_pointer:
            raise AssertionError("planning copied instead of adopting the spill lease")
        actual = planned([value])
        torch.testing.assert_close(actual.cpu(), reference(value), rtol=2e-5, atol=2e-6)
        planned.close()
    if record.current_object_id != persistent_id:
        raise AssertionError("close did not restore the persistent object ID")
    if imported_model.projection.weight.untyped_storage().data_ptr() != spill_pointer:
        raise AssertionError("close replaced explicitly imported spill storage")
    spill_bytes_after_plan = int(_statistics(runtime).runtime.spill_allocated_bytes)
    if spill_bytes_after_plan != spill_bytes_before_plan:
        raise AssertionError("planning retained a duplicate model spill copy")

    export_model_state(imported_model, runtime=runtime, release_runtime=True)
    if imported_model.projection.weight.untyped_storage().data_ptr() == spill_pointer:
        raise AssertionError("export retained the spill pointer")
    torch.testing.assert_close(
        imported_model.projection.weight, reference.projection.weight
    )

    optimizer = torch.optim.AdamW(imported_model.parameters(), lr=1e-3, foreach=False)
    imported_model(torch.randn(2, 32)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state_tensors = tuple(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    expected = tuple(value.clone() for value in state_tensors)
    identities = tuple(id(value) for value in state_tensors)
    import_optimizer_state(
        optimizer,
        runtime=runtime,
        pool="spill",
    )
    export_optimizer_state(
        optimizer,
        runtime=runtime,
        release_runtime=True,
    )
    if tuple(id(value) for value in state_tensors) != identities:
        raise AssertionError("optimizer import replaced tensor identities")
    for actual, wanted in zip(state_tensors, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    failing = _Model().eval()
    failing_pointer = failing.projection.weight.untyped_storage().data_ptr()
    try:
        plan_forward(
            failing,
            example_inputs=[torch.randn(3, 32)],
            runtime=runtime,
            execution="execution",
            spill="spill",
            verbose=False,
        )
    except RuntimeError as error:
        if "import_model_state" not in str(error):
            raise AssertionError("unimported model error is not actionable") from error
    else:
        raise AssertionError("planning accepted unimported model state")
    if persistent_state(runtime, failing) is not None:
        raise AssertionError("failed planning created runtime model state")
    if failing.projection.weight.untyped_storage().data_ptr() != failing_pointer:
        raise AssertionError("failed planning changed model storage ownership")

    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
