from __future__ import annotations

import ctypes
from dataclasses import replace

import pytest
import torch

from shadowspill.ir import AliasGroupSpec, ObjectRole, ObjectSpec
from shadowspill.pytorch.runtime_bridge import RuntimeBridge, RuntimeExecutionError
from tests.ir._examples import representative_program


class _FailureLibrary:
    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True

    def shadowspill_pytorch_allocator_failure(self, output: object) -> int:
        failure = ctypes.cast(output, ctypes.POINTER(ctypes.c_uint8))
        del failure
        from shadowspill.pytorch._abi import AdapterFailure

        result = ctypes.cast(output, ctypes.POINTER(AdapterFailure))[0]
        result.status = 4
        result.device_ordinal = 0
        result.requested_bytes = 525_336_576
        result.runtime.status = 4
        result.runtime.object_id = (1 << 64) - 1
        result.runtime.allocation_id = (1 << 64) - 1
        result.runtime.requested_bytes = 525_336_576
        result.runtime.free_bytes = 600_000_000
        result.runtime.largest_free_range_bytes = 400_000_000
        return 4


def test_failed_task_translates_latched_allocator_diagnostics() -> None:
    library = _FailureLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("tensor data is not allocated")

    with pytest.raises(RuntimeExecutionError) as caught:
        bridge.abort_task_after_failure("execute task task_000007", original)

    assert library.aborted
    assert caught.value.__cause__ is original
    message = str(caught.value)
    assert "task_000007" in message
    assert "requested=525336576" in message
    assert "free=600000000" in message
    assert "largest_free_range=400000000" in message


class _HealthyLibrary:
    def shadowspill_pytorch_abort_task_range(self) -> None:
        self.aborted = True

    def shadowspill_pytorch_allocator_failure(self, output: object) -> int:
        del output
        return 0


def test_failed_task_preserves_original_error_without_allocator_failure() -> None:
    library = _HealthyLibrary()
    bridge = RuntimeBridge(library, representative_program())
    original = RuntimeError("ordinary operation failure")

    bridge.abort_task_after_failure("execute task task_000007", original)

    assert library.aborted


class _LabelLibrary:
    def __init__(self) -> None:
        self.labels: tuple[str, ...] = ()

    def shadowspill_pytorch_task_labels_configure(
        self, values: object, count: int
    ) -> int:
        labels = ctypes.cast(values, ctypes.POINTER(ctypes.c_char_p))
        self.labels = tuple(labels[index].decode() for index in range(count))
        return 0


def test_task_trace_labels_are_projected_by_dense_canonical_id() -> None:
    library = _LabelLibrary()
    bridge = RuntimeBridge(library, representative_program())

    bridge.configure_task_labels(
        {
            "task_000002": "execution_000001.backward.stage_0001",
            "task_000000": "execution_000000.forward.stage_0000",
        }
    )

    assert library.labels == (
        "execution_000000.forward.stage_0000",
        "",
        "execution_000001.backward.stage_0001",
    )


def test_zero_size_alias_uses_no_physical_runtime_operation() -> None:
    program = representative_program()
    program = replace(
        program,
        alias_groups=(
            *program.alias_groups,
            AliasGroupSpec("alias_000099", "cuda_0", 0),
        ),
        objects=(
            *program.objects,
            ObjectSpec(
                "object_000099",
                "alias_000099",
                0,
                0,
                ObjectRole.ACTIVATION,
            ),
        ),
    )
    bridge = RuntimeBridge(object(), program)
    tensor = torch.empty(0)

    bridge.register_placeholder("alias_000099")
    binding = bridge.bind_registered_tensor("alias_000099", tensor)
    generations = bridge.adopt_many(((tensor, "alias_000099"),))
    bridge.rebind_many(((tensor, "alias_000099", binding),))

    assert not bridge.requires_storage("alias_000099")
    assert binding.pointer is None
    assert binding.generation == 0
    assert generations == (0,)
