from __future__ import annotations

import pytest

from shadowspill.runtime import (
    AdmissionError,
    AllocationEvent,
    AllocationOperation,
    admit_physical_budget,
    plan_slab_layout,
    replay_slab_timeline,
    workspace_reserve_bytes,
)

MIB = 1 << 20
GIB = 1 << 30


def event(
    position: int,
    allocation_id: str,
    operation: AllocationOperation,
    bytes_: int,
    alignment: int = 1,
    *,
    planned: bool = False,
) -> AllocationEvent:
    return AllocationEvent(
        position,
        allocation_id,
        operation,
        bytes_,
        alignment,
        planned=planned,
    )


def test_sequential_temporaries_are_charged_by_net_live_peak() -> None:
    replay = replay_slab_timeline(
        128,
        (
            event(0, "first", AllocationOperation.ALLOCATE, 96),
            event(1, "first", AllocationOperation.FREE, 96),
            event(2, "second", AllocationOperation.ALLOCATE, 112),
            event(3, "second", AllocationOperation.FREE, 112),
        ),
    )
    assert replay.peak_allocated_bytes == 112
    assert replay.final_allocated_bytes == 0
    assert replay.final_largest_free_range_bytes == 128


def test_spatial_replay_reports_fragmentation_not_only_total_free() -> None:
    timeline = (
        event(0, "left", AllocationOperation.ALLOCATE, 32),
        event(1, "middle", AllocationOperation.ALLOCATE, 32),
        event(2, "right", AllocationOperation.ALLOCATE, 32),
        event(3, "left", AllocationOperation.FREE, 32),
        event(4, "right", AllocationOperation.FREE, 32),
        event(5, "large", AllocationOperation.ALLOCATE, 48),
    )
    with pytest.raises(AdmissionError, match="largest range") as captured:
        replay_slab_timeline(96, timeline)
    assert captured.value.kind == "slab_fragmentation"
    assert captured.value.free_bytes == 64
    assert captured.value.largest_free_range_bytes == 32
    assert captured.value.position == 5
    assert captured.value.free_range_evidence == (
        (0, 32, None, "middle"),
        (64, 32, "middle", None),
    )


def test_anonymous_replay_uses_smallest_compatible_range() -> None:
    timeline = (
        event(0, "a", AllocationOperation.ALLOCATE, 64),
        event(1, "b", AllocationOperation.ALLOCATE, 32),
        event(2, "c", AllocationOperation.ALLOCATE, 96),
        event(3, "d", AllocationOperation.ALLOCATE, 32),
        event(4, "b", AllocationOperation.FREE, 32),
        event(5, "a", AllocationOperation.FREE, 64),
        event(6, "d", AllocationOperation.FREE, 32),
        event(7, "small", AllocationOperation.ALLOCATE, 48),
    )
    replay = replay_slab_timeline(256, timeline)
    assert replay.final_allocated_bytes == 144
    assert replay.final_largest_free_range_bytes == 96


def test_planned_replay_preserves_large_holes_with_best_fit_low() -> None:
    timeline = (
        event(0, "a", AllocationOperation.ALLOCATE, 64, planned=True),
        event(1, "b", AllocationOperation.ALLOCATE, 32, planned=True),
        event(2, "c", AllocationOperation.ALLOCATE, 96, planned=True),
        event(3, "d", AllocationOperation.ALLOCATE, 32, planned=True),
        event(4, "b", AllocationOperation.FREE, 32),
        event(5, "a", AllocationOperation.FREE, 64),
        event(6, "d", AllocationOperation.FREE, 32),
        event(7, "small", AllocationOperation.ALLOCATE, 48, planned=True),
        event(8, "large", AllocationOperation.ALLOCATE, 80, planned=True),
    )
    replay = replay_slab_timeline(256, timeline)
    assert replay.final_allocated_bytes == 224
    assert replay.final_largest_free_range_bytes == 16


def test_pending_extent_reuse_preserves_its_physical_range() -> None:
    replay = replay_slab_timeline(
        128,
        (
            event(0, "cached", AllocationOperation.ALLOCATE, 64),
            AllocationEvent(
                1,
                "replacement",
                AllocationOperation.REUSE,
                64,
                alignment=1,
                source_allocation_id="cached",
            ),
            event(2, "replacement", AllocationOperation.FREE, 64),
        ),
    )
    assert replay.peak_allocated_bytes == 64
    assert replay.final_allocated_bytes == 0
    assert replay.final_largest_free_range_bytes == 128


def test_static_layout_avoids_online_fragmentation_and_preserves_reuse() -> None:
    timeline = (
        event(0, "long_middle", AllocationOperation.ALLOCATE, 32),
        event(1, "short_left", AllocationOperation.ALLOCATE, 32),
        event(2, "short_right", AllocationOperation.ALLOCATE, 32),
        event(3, "short_left", AllocationOperation.FREE, 32),
        event(4, "short_right", AllocationOperation.FREE, 32),
        AllocationEvent(
            5,
            "large",
            AllocationOperation.ALLOCATE,
            64,
            alignment=1,
        ),
        AllocationEvent(
            6,
            "large_reused",
            AllocationOperation.REUSE,
            64,
            alignment=1,
            source_allocation_id="large",
        ),
    )

    layout = plan_slab_layout(96, timeline)
    offsets = layout.offset_by_allocation()

    assert offsets["large"] == offsets["large_reused"]
    assert layout.layout_bytes == 96
    assert layout.replay.peak_allocated_bytes == 96


def test_dynamic_identity_reserves_its_complete_reuse_lifetime_in_suffix() -> None:
    timeline = (
        event(0, "static", AllocationOperation.ALLOCATE, 32),
        event(1, "dynamic_source", AllocationOperation.ALLOCATE, 16),
        AllocationEvent(
            2,
            "dynamic_result",
            AllocationOperation.REUSE,
            16,
            alignment=16,
            source_allocation_id="dynamic_source",
        ),
    )

    layout = plan_slab_layout(
        96,
        timeline,
        dynamic_allocation_ids=frozenset({"dynamic_result"}),
    )
    offsets = layout.offset_by_allocation()

    assert layout.static_layout_bytes == 32
    assert layout.layout_bytes == 48
    assert layout.dynamic_allocation_ids == (
        "dynamic_result",
        "dynamic_source",
    )
    assert offsets["static"] == 0
    assert offsets["dynamic_source"] == offsets["dynamic_result"] == 80


def test_pending_extent_reuse_requires_a_live_size_matched_source() -> None:
    with pytest.raises(ValueError, match="not live"):
        replay_slab_timeline(
            128,
            (
                AllocationEvent(
                    0,
                    "replacement",
                    AllocationOperation.REUSE,
                    64,
                    alignment=1,
                    source_allocation_id="missing",
                ),
            ),
        )


def test_physical_admission_exposes_every_subtraction() -> None:
    admission, replay = admit_physical_budget(
        device_budget_bytes=4 * GIB,
        spill_budget_bytes=3 * GIB,
        context_bytes=500 * MIB,
        observed_external_bytes=600 * MIB,
        maximum_task_workspace_bytes=600 * MIB,
        predicted_spill_peak_bytes=1 * GIB,
        allocation_timeline=(
            event(0, "parameter", AllocationOperation.ALLOCATE, 1 * GIB, 256),
            event(1, "parameter", AllocationOperation.FREE, 1 * GIB, 256),
        ),
    )
    assert admission.provider_headroom_bytes == 1280 * MIB
    assert admission.slab_bytes == 4 * GIB - 500 * MIB - 1280 * MIB
    assert admission.workspace_reserve_bytes == 750 * MIB
    assert admission.spill_reservation_bytes == 1280 * MIB
    assert replay.peak_allocated_bytes == 1 * GIB


def test_workspace_reserve_has_explicit_fragmentation_allowance() -> None:
    assert workspace_reserve_bytes(600 * MIB) == 750 * MIB
    assert workspace_reserve_bytes(1) == 512 * MIB


@pytest.mark.parametrize(
    ("overrides", "kind"),
    [
        ({"device_budget_bytes": 1000 * MIB}, "fixed_device_budget"),
        ({"maximum_task_workspace_bytes": 3 * GIB}, "workspace_budget"),
        ({"spill_budget_bytes": 1 * GIB}, "spill_budget"),
    ],
)
def test_admission_failures_identify_the_physical_category(
    overrides: dict[str, int], kind: str
) -> None:
    arguments = {
        "device_budget_bytes": 4 * GIB,
        "spill_budget_bytes": 3 * GIB,
        "context_bytes": 500 * MIB,
        "observed_external_bytes": 0,
        "maximum_task_workspace_bytes": 128 * MIB,
        "predicted_spill_peak_bytes": 1 * GIB,
    }
    arguments.update(overrides)
    with pytest.raises(AdmissionError) as captured:
        admit_physical_budget(**arguments)
    assert captured.value.kind == kind


def test_invalid_timeline_lifetimes_are_rejected() -> None:
    with pytest.raises(ValueError, match="not live"):
        replay_slab_timeline(
            64,
            (event(0, "missing", AllocationOperation.FREE, 16),),
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        replay_slab_timeline(
            64,
            (
                event(1, "allocation", AllocationOperation.ALLOCATE, 16),
                event(0, "allocation", AllocationOperation.FREE, 16),
            ),
        )
