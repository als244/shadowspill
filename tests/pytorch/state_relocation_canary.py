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

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    externalize_model_state,
    externalize_optimizer_state,
    plan_forward,
    relocate_model_state,
    relocate_optimizer_state,
)
from shadowspill.pytorch.runtime_adapter.abi import (
    AdapterStatistics,
    ObjectSnapshot,
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
        runtime._installed.library.shadowspill_pytorch_object_snapshot(
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

    copied_model = relocate_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=False,
    )
    copied = persistent_state(runtime, copied_model)
    if copied is None or len(copied.storages) != 1:
        raise AssertionError("tied model state did not become one persistent object")
    if copied_model is model:
        raise AssertionError("copy relocation returned its input model")
    if copied.source_owner is not model:
        raise AssertionError("copy relocation did not retain its source owner")
    if id(copied_model.projection.weight) == source_parameter_id:
        raise AssertionError("copy relocation reused its input Parameter")
    if model.projection.weight.untyped_storage().data_ptr() != source_pointer:
        raise AssertionError("copy relocation replaced its CPU source")
    if copied_model.projection.weight.untyped_storage().data_ptr() != (
        copied.storages[0].spill_pointer
    ):
        raise AssertionError("relocated copy does not point into spill memory")
    if (
        copied_model.tied is not copied_model.projection.weight
        or copied_model.weight_view.untyped_storage()._cdata
        != copied_model.projection.weight.untyped_storage()._cdata
    ):
        raise AssertionError("relocated copy did not preserve ties and views")
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
        raise AssertionError("copy relocation registered the wrong object count")

    externalize_model_state(copied_model, runtime=runtime, release_runtime=True)
    del copied
    source_reference = weakref.ref(model)
    relocated_model = relocate_model_state(
        model,
        runtime=runtime,
        pool="spill",
    )
    relocated = persistent_state(runtime, relocated_model)
    if relocated is None:
        raise AssertionError("source-releasing relocation has no persistent state")
    if relocated.source_owner is not None:
        raise AssertionError("source-releasing relocation retained its input model")
    if (
        id(model.projection.weight) != source_parameter_id
        or int(model.projection.weight.untyped_storage()._cdata) != source_storage_id
        or model.projection.weight.untyped_storage().data_ptr() != source_pointer
    ):
        raise AssertionError("source-releasing relocation mutated its input model")
    del model
    gc.collect()
    if source_reference() is not None:
        raise AssertionError("source-releasing relocation retained the source model")
    record = relocated.storages[0]
    if (
        relocated_model.projection.weight.untyped_storage().data_ptr()
        != record.spill_pointer
    ):
        raise AssertionError("returned model does not point into its spill lease")
    if (
        relocated_model.tied is not relocated_model.projection.weight
        or relocated_model.weight_view.untyped_storage()._cdata
        != relocated_model.projection.weight.untyped_storage()._cdata
    ):
        raise AssertionError("move relocation did not preserve ties and views")

    value = torch.randn(3, 32)
    persistent_id = record.persistent_object_id
    spill_pointer = record.spill_pointer
    spill_bytes_before_plan = int(_statistics(runtime).runtime.spill_allocated_bytes)
    with tempfile.TemporaryDirectory() as cache:
        planned = plan_forward(
            relocated_model,
            example_inputs=[value],
            runtime=runtime,
            execution="execution",
            spill="spill",
            planning_cachedir=cache,
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
    if relocated_model.projection.weight.untyped_storage().data_ptr() != spill_pointer:
        raise AssertionError("close replaced explicitly relocated spill storage")
    spill_bytes_after_plan = int(_statistics(runtime).runtime.spill_allocated_bytes)
    if spill_bytes_after_plan != spill_bytes_before_plan:
        raise AssertionError("planning retained a duplicate model spill copy")

    externalize_model_state(relocated_model, runtime=runtime, release_runtime=True)
    if relocated_model.projection.weight.untyped_storage().data_ptr() == spill_pointer:
        raise AssertionError("externalization retained the spill pointer")
    torch.testing.assert_close(
        relocated_model.projection.weight, reference.projection.weight
    )

    optimizer = torch.optim.AdamW(relocated_model.parameters(), lr=1e-3, foreach=False)
    relocated_model(torch.randn(2, 32)).sum().backward()
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
    relocate_optimizer_state(
        optimizer,
        runtime=runtime,
        pool="spill",
    )
    externalize_optimizer_state(
        optimizer,
        runtime=runtime,
        release_runtime=True,
    )
    if tuple(id(value) for value in state_tensors) != identities:
        raise AssertionError("optimizer relocation replaced tensor identities")
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
        if "relocate_model_state" not in str(error):
            raise AssertionError("unrelocated model error is not actionable") from error
    else:
        raise AssertionError("planning accepted unrelocated model state")
    if persistent_state(runtime, failing) is not None:
        raise AssertionError("failed planning created runtime model state")
    if failing.projection.weight.untyped_storage().data_ptr() != failing_pointer:
        raise AssertionError("failed planning changed model storage ownership")

    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
