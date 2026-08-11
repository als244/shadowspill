from __future__ import annotations

import ctypes

import pytest

from shadowspill.pytorch._abi import AllocationEvent as CAllocationEvent
from shadowspill.pytorch._telemetry import (
    AllocationCategory,
    AllocationEventKind,
    AllocationTelemetryError,
    CapturedAllocationEvent,
    read_allocation_telemetry,
    summarize_task_workspace,
)


def _event(
    sequence: int,
    allocation_id: int,
    kind: AllocationEventKind,
    bytes_: int,
    *,
    task_id: int = 7,
    category: AllocationCategory = AllocationCategory.ANONYMOUS,
) -> CapturedAllocationEvent:
    return CapturedAllocationEvent(
        sequence=sequence,
        task_id=task_id,
        allocation_id=allocation_id,
        generation=allocation_id,
        requested_bytes=bytes_,
        charged_bytes=max(bytes_, 1),
        slab_offset=0,
        kind=kind,
        category=category,
    )


def test_workspace_uses_live_peak_and_excludes_promoted_outputs() -> None:
    events = (
        _event(0, 1, AllocationEventKind.CREATED, 64),
        _event(1, 1, AllocationEventKind.RELEASED, 64),
        _event(2, 2, AllocationEventKind.CREATED, 96),
        _event(3, 3, AllocationEventKind.CREATED, 32),
        _event(4, 3, AllocationEventKind.PROMOTED, 32),
        _event(5, 2, AllocationEventKind.RELEASED, 96),
    )
    profile = summarize_task_workspace(events, task_id=7)
    assert profile.peak_requested_bytes == 96
    assert profile.peak_charged_bytes == 96
    assert profile.peak_extent_bytes == (96,)
    assert profile.promoted_allocation_ids == (3,)


def test_workspace_ignores_other_tasks_and_unknown_prior_release() -> None:
    events = (
        _event(0, 1, AllocationEventKind.RELEASED, 64),
        _event(1, 2, AllocationEventKind.CREATED, 48, task_id=8),
    )
    assert summarize_task_workspace(events, task_id=7).peak_charged_bytes == 0


class _ReadFunction:
    def __init__(self, events: tuple[CAllocationEvent, ...]) -> None:
        self.events = events

    def __call__(self, destination: object, capacity: int, count: object) -> int:
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint64))[0] = len(self.events)
        if destination is not None:
            assert capacity >= len(self.events)
            target = ctypes.cast(destination, ctypes.POINTER(CAllocationEvent))
            for index, event in enumerate(self.events):
                target[index] = event
        return 0


class _Library:
    def __init__(self, events: tuple[CAllocationEvent, ...]) -> None:
        self.shadowspill_pytorch_allocation_telemetry_read = _ReadFunction(events)


def test_read_decodes_no_task_and_validates_sequence() -> None:
    raw = CAllocationEvent(
        sequence=0,
        task_id=(1 << 64) - 1,
        allocation_id=3,
        generation=4,
        requested_bytes=0,
        charged_bytes=1,
        slab_offset=8,
        kind=AllocationEventKind.CREATED,
        category=AllocationCategory.ANONYMOUS,
    )
    decoded = read_allocation_telemetry(_Library((raw,)))
    assert decoded[0].task_id is None
    assert decoded[0].charged_bytes == 1

    raw.sequence = 2
    with pytest.raises(AllocationTelemetryError, match="sequence"):
        read_allocation_telemetry(_Library((raw,)))
