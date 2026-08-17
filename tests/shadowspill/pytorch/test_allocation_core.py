from __future__ import annotations

import pytest

from shadowspill.pytorch.profiling import (
    AllocationPathProbe,
    AmbiguousAllocationPathError,
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
    derive_core_allocation_path,
)


def _abi(*sizes: int) -> TaskAllocationContract:
    events: list[TaskAllocationEvent] = []
    for ordinal, size in enumerate(sizes):
        events.append(
            TaskAllocationEvent(
                ordinal,
                TaskAllocationOperation.ALLOCATE,
                size,
                size,
            )
        )
        events.append(
            TaskAllocationEvent(
                ordinal,
                TaskAllocationOperation.FREE,
                size,
                size,
            )
        )
    return TaskAllocationContract.capture(events)


def test_identical_probe_matrix_keeps_exact_core() -> None:
    reference = _abi(64, 128)
    derived = derive_core_allocation_path(
        reference,
        (
            AllocationPathProbe(0, 0, reference),
            AllocationPathProbe(0, 1, reference),
        ),
    )

    assert derived.allocation_contract == reference
    assert derived.weighted_edit_distance == 0
    assert all(item.scratch_allocation_count == 0 for item in derived.observations)


def test_one_time_insertion_is_scratch_around_warmed_core() -> None:
    reference = _abi(64, 128)
    cold = _abi(24, 64, 128)
    derived = derive_core_allocation_path(
        reference,
        (
            AllocationPathProbe(0, 0, cold),
            AllocationPathProbe(0, 1, reference),
        ),
    )

    assert derived.allocation_contract == reference
    cold_observation, warm_observation = derived.observations
    assert cold_observation.scratch_allocation_count == 1
    assert cold_observation.scratch_peak_charged_bytes == 24
    assert warm_observation.scratch_allocation_count == 0


def test_same_geometry_insertion_is_rejected_as_ambiguous() -> None:
    reference = _abi(64)
    ambiguous = _abi(64, 64)

    with pytest.raises(AmbiguousAllocationPathError, match="multiple minimum-edit"):
        derive_core_allocation_path(
            reference,
            (
                AllocationPathProbe(0, 0, ambiguous),
                AllocationPathProbe(0, 1, reference),
            ),
        )
