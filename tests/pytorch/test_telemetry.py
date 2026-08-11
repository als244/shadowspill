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
    slab_offset: int = 0,
    charged_bytes: int | None = None,
) -> CapturedAllocationEvent:
    return CapturedAllocationEvent(
        sequence=sequence,
        task_id=task_id,
        allocation_id=allocation_id,
        generation=allocation_id,
        requested_bytes=bytes_,
        charged_bytes=max(bytes_, 1) if charged_bytes is None else charged_bytes,
        slab_offset=slab_offset,
        kind=kind,
        category=category,
    )


def test_workspace_uses_live_peak_and_excludes_promoted_outputs() -> None:
    events = (
        _event(0, 1, AllocationEventKind.CREATED, 64),
        _event(1, 1, AllocationEventKind.LOGICAL_FREED, 64),
        _event(2, 2, AllocationEventKind.CREATED, 96),
        _event(3, 3, AllocationEventKind.CREATED, 32),
        _event(4, 3, AllocationEventKind.PROMOTED, 32),
        _event(5, 2, AllocationEventKind.LOGICAL_FREED, 96),
    )
    profile = summarize_task_workspace(events, task_id=7)
    assert profile.peak_requested_bytes == 96
    assert profile.peak_charged_bytes == 96
    assert profile.peak_extent_bytes == (96,)
    assert profile.promoted_allocation_ids == (3,)
    assert profile.output_allocation_ids == ()


def test_workspace_excludes_outputs_resolved_after_task_execution() -> None:
    events = (
        _event(0, 10, AllocationEventKind.CREATED, 128),
        _event(1, 11, AllocationEventKind.CREATED, 64),
        _event(2, 11, AllocationEventKind.LOGICAL_FREED, 64),
    )
    profile = summarize_task_workspace(
        events, task_id=7, output_allocation_leaves={10: (0,)}
    )
    assert profile.peak_charged_bytes == 64
    assert profile.output_allocation_ids == (10,)
    assert profile.allocation_trace[0].output_leaf_indices == (0,)


def test_workspace_classifies_unbound_live_allocation_as_persistent() -> None:
    events = (
        _event(0, 10, AllocationEventKind.CREATED, 128),
        _event(1, 11, AllocationEventKind.CREATED, 32),
        _event(2, 10, AllocationEventKind.LOGICAL_FREED, 128),
    )
    profile = summarize_task_workspace(events, task_id=7)
    assert profile.peak_requested_bytes == 128
    assert profile.peak_charged_bytes == 128
    assert profile.persistent_allocation_ids == (11,)
    assert profile.persistent_extent_bytes == (32,)
    assert all(event.allocation_ordinal == 0 for event in profile.allocation_trace)


def test_workspace_ignores_other_tasks_and_unknown_prior_release() -> None:
    events = (
        _event(0, 1, AllocationEventKind.LOGICAL_FREED, 64),
        _event(1, 2, AllocationEventKind.CREATED, 48, task_id=8),
    )
    assert summarize_task_workspace(events, task_id=7).peak_charged_bytes == 0


def test_trace_identifies_same_stream_cached_extent_reuse() -> None:
    events = (
        _event(0, 1, AllocationEventKind.CREATED, 64, slab_offset=128),
        _event(1, 1, AllocationEventKind.LOGICAL_FREED, 64, slab_offset=128),
        _event(2, 2, AllocationEventKind.CREATED, 64, slab_offset=128),
        _event(3, 2, AllocationEventKind.LOGICAL_FREED, 64, slab_offset=128),
    )
    profile = summarize_task_workspace(events, task_id=7)
    assert profile.allocation_trace[2].reuses_ordinal == 0
    assert profile.allocation_trace[2].charged_bytes == 64


def test_trace_treats_a_split_cached_extent_as_an_ordinary_suballocation() -> None:
    events = (
        _event(0, 1, AllocationEventKind.CREATED, 64, slab_offset=128),
        _event(1, 1, AllocationEventKind.LOGICAL_FREED, 64, slab_offset=128),
        _event(2, 2, AllocationEventKind.CREATED, 48, slab_offset=128),
        _event(
            3,
            1,
            AllocationEventKind.RELEASED,
            0,
            slab_offset=176,
            charged_bytes=16,
        ),
        _event(4, 2, AllocationEventKind.LOGICAL_FREED, 48, slab_offset=128),
    )
    profile = summarize_task_workspace(events, task_id=7)
    assert profile.allocation_trace[2].reuses_ordinal is None
    assert profile.allocation_trace[2].charged_bytes == 48


def test_physical_release_breaks_cached_extent_reuse() -> None:
    events = (
        _event(0, 1, AllocationEventKind.CREATED, 64, slab_offset=128),
        _event(1, 1, AllocationEventKind.LOGICAL_FREED, 64, slab_offset=128),
        _event(2, 1, AllocationEventKind.RELEASED, 64, slab_offset=128),
        _event(3, 2, AllocationEventKind.CREATED, 64, slab_offset=128),
    )
    profile = summarize_task_workspace(
        events,
        task_id=7,
        output_allocation_leaves={2: (0,)},
    )
    assert profile.allocation_trace[2].reuses_ordinal is None


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
