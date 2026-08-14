from __future__ import annotations

import pytest

from shadowspill.runtime import (
    AdmissionError,
    MemoryReplayLeaseState,
    MemoryReplayOperation,
    MemoryReplayOperationKind,
    replay_memory_pool,
)


def _operation(
    sequence: int,
    lease_id: int,
    kind: MemoryReplayOperationKind,
    *,
    bytes_: int = 0,
    dependency_id: int | None = None,
    dependency_expected: bool = False,
) -> MemoryReplayOperation:
    return MemoryReplayOperation(
        sequence,
        lease_id,
        kind,
        bytes=bytes_,
        alignment=1 if bytes_ else 0,
        dependency_id=dependency_id,
        dependency_expected=dependency_expected,
    )


def test_replay_uses_exact_causal_successor_policy() -> None:
    replay = replay_memory_pool(
        128,
        (
            _operation(0, 0, MemoryReplayOperationKind.ACQUIRE, bytes_=96),
            _operation(
                1,
                0,
                MemoryReplayOperationKind.BEGIN_RETIREMENT,
                dependency_id=0,
            ),
            _operation(2, 1, MemoryReplayOperationKind.RESERVE, bytes_=64),
            _operation(3, 1, MemoryReplayOperationKind.ACQUIRE_RESERVED),
            _operation(
                4,
                0,
                MemoryReplayOperationKind.COMPLETE_RETIREMENT,
                dependency_id=0,
            ),
            _operation(5, 1, MemoryReplayOperationKind.RELEASE),
        ),
        lease_count=2,
        dependency_count=1,
        minimum_alignment=1,
    )

    assert replay.peak_allocated_bytes == 96
    assert replay.final_allocated_bytes == 0
    assert replay.decisions[2].physical_bytes_delta == 0
    assert replay.decisions[2].resulting_state is (
        MemoryReplayLeaseState.SUCCESSOR_RESERVED
    )
    assert replay.dependencies[0].predecessor_lease_id == 0
    assert replay.dependencies[0].successor_lease_id == 1
    assert replay.dependencies[0].dependency_id == 0


def test_replay_carries_promised_dependency_without_timing() -> None:
    replay = replay_memory_pool(
        128,
        (
            _operation(0, 0, MemoryReplayOperationKind.ACQUIRE, bytes_=128),
            _operation(
                1,
                0,
                MemoryReplayOperationKind.BEGIN_RETIREMENT,
                dependency_id=0,
                dependency_expected=True,
            ),
            _operation(2, 1, MemoryReplayOperationKind.RESERVE, bytes_=32),
            _operation(3, 1, MemoryReplayOperationKind.ACQUIRE_RESERVED),
            _operation(
                4,
                0,
                MemoryReplayOperationKind.COMPLETE_RETIREMENT,
                dependency_id=0,
            ),
            _operation(5, 1, MemoryReplayOperationKind.RELEASE),
        ),
        lease_count=2,
        dependency_count=1,
        minimum_alignment=1,
    )

    assert replay.dependencies[0].dependency_id == 0
    assert replay.dependencies[0].predecessor_lease_id == 0
    assert replay.dependencies[0].successor_lease_id == 1
    assert replay.decisions[2].physical_bytes_delta == 0


def test_replay_reports_exact_infeasible_geometry() -> None:
    with pytest.raises(AdmissionError) as caught:
        replay_memory_pool(
            128,
            (
                _operation(0, 0, MemoryReplayOperationKind.ACQUIRE, bytes_=96),
                _operation(1, 1, MemoryReplayOperationKind.ACQUIRE, bytes_=64),
            ),
            lease_count=2,
            dependency_count=0,
            minimum_alignment=1,
        )

    assert caught.value.kind == "memory_pool_fragmentation"
    assert caught.value.position == 1
    assert caught.value.required_bytes == 64
    assert caught.value.free_bytes == 32
    assert caught.value.largest_free_range_bytes == 32
