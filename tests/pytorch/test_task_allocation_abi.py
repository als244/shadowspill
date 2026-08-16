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
    compare_allocation_path,
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

    changed_physical_placement = (
        *trace[:-1],
        trace[-1].__class__(
            1,
            TaskAllocationOperation.ALLOCATE,
            16,
            64,
            output_leaf_indices=(0,),
            output_view_offsets=(48,),
            reuses_ordinal=None,
            alignment_bytes=512,
        ),
    )
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


def test_task_allocation_abi_rejects_sparse_allocation_ordinals() -> None:
    steps = (
        TaskAllocationABIStep(
            0,
            1,
            TaskAllocationOperation.ALLOCATE,
            32,
            64,
            256,
            output_leaf_indices=(0,),
            persistent_after_task=True,
        ),
    )

    with pytest.raises(ValueError, match="dense allocation ordinals"):
        TaskAllocationABI(steps, "0" * 64)


def test_task_allocation_abi_retains_bounded_provider_allocation() -> None:
    abi = TaskAllocationABI.capture(
        (
            TaskAllocationEvent(
                0,
                TaskAllocationOperation.ALLOCATE,
                32,
                32,
            ),
        )
    )

    assert abi.steps[0].persistent_after_task
    assert not abi.steps[0].output_leaf_indices
    assert abi.for_retained_output_leaves(()).steps[0].persistent_after_task


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


def test_allocation_path_reconciles_multiple_insertions_and_omissions() -> None:
    reference = TaskAllocationABI.capture(
        (
            TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 16, 16),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 16, 16),
            TaskAllocationEvent(1, TaskAllocationOperation.ALLOCATE, 32, 32),
            TaskAllocationEvent(1, TaskAllocationOperation.FREE, 32, 32),
            TaskAllocationEvent(2, TaskAllocationOperation.ALLOCATE, 64, 64),
            TaskAllocationEvent(2, TaskAllocationOperation.FREE, 64, 64),
            TaskAllocationEvent(
                3,
                TaskAllocationOperation.ALLOCATE,
                48,
                48,
                output_leaf_indices=(0,),
                output_view_offsets=(0,),
            ),
        )
    )
    observed = TaskAllocationABI.capture(
        (
            TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 8, 8),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 8, 8),
            TaskAllocationEvent(1, TaskAllocationOperation.ALLOCATE, 32, 32),
            TaskAllocationEvent(1, TaskAllocationOperation.FREE, 32, 32),
            TaskAllocationEvent(2, TaskAllocationOperation.ALLOCATE, 12, 12),
            TaskAllocationEvent(2, TaskAllocationOperation.FREE, 12, 12),
            TaskAllocationEvent(
                3,
                TaskAllocationOperation.ALLOCATE,
                48,
                48,
                output_leaf_indices=(0,),
                output_view_offsets=(0,),
            ),
        )
    )

    path = compare_allocation_path(
        reference,
        observed,
        probe_index=2,
        repetition=1,
    )

    assert path.scratch_allocation_count == 2
    assert path.scratch_maximum_charged_bytes == 12
    assert path.scratch_peak_charged_bytes == 12
    assert path.scratch_terminal_charged_bytes == 0


def test_allocation_path_rejects_unknown_framework_output() -> None:
    reference = TaskAllocationABI.capture(())
    observed = TaskAllocationABI.capture(
        (
            TaskAllocationEvent(
                0,
                TaskAllocationOperation.ALLOCATE,
                32,
                32,
                output_leaf_indices=(0,),
                output_view_offsets=(0,),
            ),
        )
    )

    with pytest.raises(ValueError, match="framework-visible allocation"):
        compare_allocation_path(reference, observed, probe_index=0, repetition=0)
