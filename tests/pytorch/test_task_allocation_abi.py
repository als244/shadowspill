from __future__ import annotations

import pytest

from shadowspill.pytorch.capture.storage import (
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)
from shadowspill.pytorch.profiling import (
    TaskAllocationABI,
    TaskAllocationABIStep,
    TaskAllocationEvent,
    TaskAllocationOperation,
)


def _mutation_contract() -> TaskStorageContract:
    return TaskStorageContract(
        roots=(
            StorageRoot(
                root_id=0,
                kind=StorageRootKind.FRESH,
                source_input=None,
                producer_node="replacement",
                producer_target="aten.add.Tensor",
                producer_result=0,
                minimum_span_bytes=16,
            ),
        ),
        output_views=(
            OutputView(
                leaf_index=0,
                root_id=0,
                offset_bytes=0,
                span_bytes=16,
                shape=(4,),
                stride=(1,),
                dtype="torch.float32",
                layout="torch.strided",
            ),
        ),
        mutations=(
            MutationBinding(
                input_position=3,
                replacement_output_leaf=0,
                producer_node="replacement",
                producer_target="aten.add.Tensor",
                argument_name="self",
            ),
        ),
        compatibility_digest="0" * 64,
    )


def test_task_allocation_abi_is_pointer_free_and_deterministic() -> None:
    trace = (
        TaskAllocationEvent(
            0,
            TaskAllocationOperation.ALLOCATE,
            64,
            64,
            alignment_bytes=512,
        ),
        TaskAllocationEvent(
            0,
            TaskAllocationOperation.FREE,
            64,
            64,
            alignment_bytes=512,
        ),
        TaskAllocationEvent(
            1,
            TaskAllocationOperation.ALLOCATE,
            16,
            64,
            output_leaf_indices=(0,),
            output_view_offsets=(32,),
            reuses_ordinal=0,
            alignment_bytes=512,
        ),
    )
    abi = TaskAllocationABI.capture(trace, _mutation_contract())
    assert abi.steps[-1].mutation_input_positions == (3,)
    assert abi.steps[-1].persistent_after_task
    assert "offset" not in str(abi.to_dict())
    assert TaskAllocationABI.from_dict(abi.to_dict()) == abi

    changed_physical_placement = (*trace[:-1], trace[-1].__class__(
        1,
        TaskAllocationOperation.ALLOCATE,
        16,
        64,
        output_leaf_indices=(0,),
        output_view_offsets=(48,),
        reuses_ordinal=None,
        alignment_bytes=512,
    ))
    assert (
        TaskAllocationABI.capture(changed_physical_placement, _mutation_contract())
        == abi
    )


def test_task_allocation_abi_rejects_geometry_changes_on_free() -> None:
    steps = (
        TaskAllocationABIStep(
            0,
            0,
            TaskAllocationOperation.ALLOCATE,
            32,
            64,
            256,
        ),
        TaskAllocationABIStep(
            1,
            0,
            TaskAllocationOperation.FREE,
            32,
            64,
            512,
        ),
    )
    with pytest.raises(ValueError, match="geometry on free"):
        TaskAllocationABI(steps, "0" * 64)


def test_task_allocation_abi_specializes_returned_output_ownership() -> None:
    profiled = TaskAllocationABI.capture(
        (
            TaskAllocationEvent(
                0,
                TaskAllocationOperation.ALLOCATE,
                64,
                64,
                output_leaf_indices=(0, 1),
                output_view_offsets=(0, 16),
            ),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 64, 64),
        )
    )
    assert not profiled.steps[0].persistent_after_task
    assert len(profiled.steps) == 2

    retained = profiled.for_retained_output_leaves((1,))

    assert len(retained.steps) == 1
    assert retained.steps[0].persistent_after_task
    assert retained.steps[0].output_leaf_indices == (0, 1)
    assert retained.compatibility_digest != profiled.compatibility_digest
