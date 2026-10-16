"""Videos preserve exact addresses, timing, pending bytes, and plan identity."""

from __future__ import annotations

import json
import shutil
import subprocess
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from shadowspill.planner import plan_program
from tests.shadowspill.planner.test_annotated_plan import _pressurefit_program
from tools.pressurefit_trace.plan.data import EventSeries, PlanData
from tools.pressurefit_trace.plan.placement import PlacementReplay
from tools.pressurefit_trace.plan.storyboard import Timing, build_story, state_at
from tools.pressurefit_trace.plan.waits import TaskWaits


@pytest.fixture(scope="module")
def plan():
    return json.loads(plan_program(_pressurefit_program(), verbose=False).to_json())


@pytest.fixture
def data(plan):
    return PlanData(plan)


def test_pending_bytes_are_half_open_and_simultaneous_events_are_atomic():
    # First copy starts exactly when another is triggered. Active copying
    # does not stay in the queue. A zero-wait copy adds no backlog.
    series = EventSeries.build(
        [
            (0, 10, 64, "fetch"),
            (10, 20, 128, "fetch"),
            (5, 15, 32, "evict"),
            (7, 7, 1024, "fetch"),
        ],
        30,
    )
    assert series.at(9) == {"evict": 32, "fetch": 64}
    assert series.at(10) == {"evict": 32, "fetch": 128}
    assert series.at(15) == {"evict": 0, "fetch": 128}
    assert series.at(20) == {"evict": 0, "fetch": 0}
    assert series.peak == 160


def test_loader_validates_semantics_and_schedule_identity(plan):
    with pytest.raises(ValueError, match="semantic labels"):
        PlanData(plan, semantics={"program_digest": "wrong"})
    with pytest.raises(ValueError, match="metadata"):
        PlanData(plan, metadata={"schedule_digest": "wrong"})


def test_saved_plan_occupancy_and_physical_offsets_survive_loading(data):
    replay = PlacementReplay(data)
    assert replay.verify()["verified_offsets"] == len(data.leases)
    assert data.execution.peak == data.occupancy.execution.peak()[0]
    assert data.spill.peak == data.occupancy.spill.peak()[0]
    assert data.end_ns == data.simulation["makespan_ns"]


def test_replay_explains_conflict_alignment_reuse_and_resident_slice():
    def lease(index, start, end, size, offset, align=1, alias=None):
        return {
            "lease_id": index,
            "predicted_start_ns": start,
            "predicted_end_ns": end,
            "bytes": size,
            "offset": offset,
            "alignment": align,
            "alias_group_id": alias,
            "purpose": "task_output",
        }

    # A is largest; B conflicts and must align up. C reuses A at the exact
    # endpoint. The resident lease contributes no lifetime endpoint to the
    # main ranking and takes a static address after the reusable range.
    leases = [
        lease(0, 0, 10, 100, 0),
        lease(1, 5, 15, 80, 128, 64),
        lease(2, 10, 20, 64, 0),
        lease(3, 3, 17, 8, 208, 8, "small"),
    ]
    data = SimpleNamespace(
        leases=leases, resident_aliases={"small"}, layout={"fixed_slice_bytes": 216}
    )
    replay = PlacementReplay(data)
    assert replay.verify()["main_bytes"] == 208
    assert replay.order.tolist() == [0, 1, 2, 3]
    blocked = replay.detail(1)
    assert blocked["occupied_union"] == [[0, 100]]
    assert blocked["attempts"] == [{"offset": 0, "blocked_range": [0, 100]}]
    assert blocked["offset"] == 128
    reused = replay.detail(2)
    assert reused["reused_predecessor_indices"] == [0]
    assert reused["offset"] == 0
    assert replay.detail(3)["resident_slice"]


def test_placement_mismatch_is_rejected(data):
    replay = PlacementReplay(data)
    replay.offsets = replay.offsets.copy() + 1
    with pytest.raises(ValueError, match="differs from saved"):
        replay.verify()


def test_placement_closeup_labels_use_the_zoomed_address_scale(data):
    from tools.pressurefit_trace.plan.render import create_renderer

    replay = PlacementReplay(data)
    replay.verify()
    index = int(replay.order[0])
    data.pool_bytes *= 1000
    story = build_story(data, replay, "emission-placement", placements=[index])
    scene = next(s for s in story["scenes"] if s.get("reveal"))
    renderer = create_renderer("emission-placement", data, replay, width=1280)
    try:
        renderer.draw(state_at(scene, 0.5, data))
        axis = next(a for a in renderer.axes if a.get_ylabel() == "Pool address (GiB)")
        assert str(replay.ranks[index] + 1) in [t.get_text() for t in axis.texts]
    finally:
        renderer.close()


def test_ranking_uses_distinct_boundaries_not_elapsed_duration():
    # Both equal-size leases have equal elapsed time, but the second spans
    # more distinct boundaries and must be placed first.
    items = [(0, 10, 100), (20, 30, 100), (21, 22, 1), (23, 24, 1)]
    leases = [
        {
            "lease_id": i,
            "predicted_start_ns": start,
            "predicted_end_ns": end,
            "bytes": size,
            "offset": 0,
            "alignment": 1,
            "alias_group_id": None,
            "purpose": "task_workspace",
        }
        for i, (start, end, size) in enumerate(items)
    ]
    replay = PlacementReplay(SimpleNamespace(leases=leases, resident_aliases=set()))
    assert replay.order[0] == 1


@pytest.mark.parametrize("mode", ["emission-placement", "execution"])
def test_storyboard_reaches_the_exact_end_and_draws_a_generic_plan(
    data, mode, tmp_path
):
    from tools.pressurefit_trace.plan.render import create_renderer

    replay = PlacementReplay(data)
    replay.verify()
    story = build_story(data, replay, mode, timing=Timing(0.1, 0.1, 0.1, 0.1, 0.1))
    state = state_at(story["scenes"][-1], 1, data)
    assert state["cursor_ns"] == data.end_ns
    assert sum(s["seconds"] for s in story["scenes"]) == story["seconds"]
    renderer = create_renderer(mode, data, replay, width=1280)
    try:
        for scene in story["scenes"]:
            assert len(renderer.draw(state_at(scene, 0.5, data))) == 1280 * 720 * 4
        renderer.save(tmp_path / "frame.png")
        assert (tmp_path / "frame.png").stat().st_size > 1000
    finally:
        renderer.close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_duration_dimensions_and_saved_evidence(data, tmp_path):
    from tools.pressurefit_trace.plan.render import create_renderer, render_movie

    replay = PlacementReplay(data)
    replay.verify()
    story = build_story(
        data, replay, "execution", timing=Timing(0.2, 0.2, 0.2, 0.2, 0.2)
    )
    path = tmp_path / "execution.mp4"
    renderer = create_renderer("execution", data, replay, width=1280)
    try:
        result = render_movie(
            renderer, story, path, fps=10, motion_fps=5, quality="preview"
        )
    finally:
        renderer.close()
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (1280, 720)
    assert float(probe["format"]["duration"]) == pytest.approx(result["seconds"])
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["schedule_digest"] == data.schedule_digest
    assert saved["frames"] == result["frames"]
    assert saved["window_tasks"] == 8
    assert saved["sweep_camera"] == "fixed_time_window"
    assert len(list((tmp_path / "execution_frames").glob("*.png"))) == len(
        story["scenes"]
    )


def test_exact_occupancy_steps_are_not_downsampled():
    series = EventSeries.build([(10, 11, 2**40, "workspace")], 10**9)
    assert series.peak == 2**40
    assert np.count_nonzero(series.values) == 1


def test_sliding_camera_keeps_a_fixed_time_scale_across_task_durations():
    durations = [10, 10, 10, 100, 100, 10]
    ends = np.cumsum(durations)
    data = SimpleNamespace(
        tasks=[
            {"start_ns": int(end - duration), "end_ns": int(end)}
            for duration, end in zip(durations, ends, strict=True)
        ],
        end_ns=int(ends[-1]),
    )
    scene = {"pan": True, "cursor_start_ns": 0, "cursor_end_ns": data.end_ns}
    states = [
        state_at(scene, t / data.end_ns, data, window_tasks=2)
        for t in range(data.end_ns + 1)
    ]
    widths = [s["window_ns"][1] - s["window_ns"][0] for s in states]
    assert max(widths) - min(widths) <= 1  # integer timestamp rounding
    assert (
        max(abs(b["window_ns"][0] - a["window_ns"][0]) for a, b in pairwise(states))
        <= 1
    )
    assert states[-1]["window_ns"][1] == data.end_ns


def test_backlog_bars_are_proportional_to_bytes_not_objects(data):
    from matplotlib.colors import to_rgba

    from tools.pressurefit_trace.plan.render import create_renderer
    from tools.pressurefit_trace.plan.render_common import ROLE_COLOR

    pending = [
        {"bytes": 2**20, "role": "parameter"},
        {"bytes": 2 * 2**20, "role": "gradient"},
    ]
    data.queue = lambda direction, instant: pending if direction == "fetch" else []
    data.pending_bytes = EventSeries.build([(0, 10, 3 * 2**20, "fetch")], 10)
    replay = PlacementReplay(data)
    replay.verify()
    renderer = create_renderer("execution", data, replay, width=1280)
    try:
        card = renderer.card()
        renderer.queue_card(card, {"cursor_ns": 5})

        def width(role):
            color = to_rgba(ROLE_COLOR[role])
            return sum(
                p.get_width() for p in card.patches if p.get_facecolor() == color
            )

        assert width("gradient") == pytest.approx(2 * width("parameter"))
        assert any(t.get_text() == "3 MiB" for t in card.texts)
        assert not any("backlog:" in t.get_text() for t in card.texts)
    finally:
        renderer.close()


def test_incremental_layout_can_rewind_without_changing_geometry(data):
    import matplotlib.pyplot as plt

    from tools.pressurefit_trace.plan.render_common import LayoutRaster

    replay = PlacementReplay(data)
    replay.verify()
    raster = LayoutRaster(data, replay, 1280)
    try:
        first = raster.image(len(data.leases)).copy()
        raster.image(0)
        second = raster.image(len(data.leases)).copy()
        np.testing.assert_array_equal(first, second)
    finally:
        plt.close(raster.fig)


def waiting_example():
    tasks = [
        {
            "task_id": name,
            "phase": "backward",
            "inputs": inputs,
            "outputs": [],
            "mutations": [],
            "workspace_bytes": 32,
            "ready_ns": ready,
            "start_ns": start,
            "end_ns": end,
            "stall_reasons": ["input-residency"] if start > ready else [],
        }
        for name, inputs, ready, start, end in (
            ("setup", [], 0, 0, 5),
            ("consumer", ["needed"], 5, 20, 25),
            ("later", ["other"], 25, 25, 30),
        )
    ]
    actions = [
        {
            "index": i,
            "alias_group_id": alias,
            "bytes": size,
            "kind": "fetch",
            "boundary": 1,
            "trigger_ns": 0,
            "transfer": {
                "direction": "fetch",
                "start_ns": start,
                "end_ns": end,
                "stall_reasons": ["lane-busy"] if start else [],
            },
        }
        for i, (alias, size, start, end) in enumerate(
            [("other", 128, 0, 10), ("needed", 64, 10, 20)]
        )
    ]
    return SimpleNamespace(
        tasks=tasks,
        actions=actions,
        transfers=actions,
        leases=[],
        object_alias={"other": "other", "needed": "needed"},
        facts=SimpleNamespace(alias_size={"other": 128, "needed": 64}),
    )


def test_task_wait_separates_inputs_from_later_needed_copies_ahead():
    data = waiting_example()
    waits = TaskWaits(data, {})
    detail = waits.detail(1)
    assert detail["stall_ns"] == 15
    assert detail["blocking_transfers"] == [
        {"action": 1, "relation": "required_input", "purposes": []}
    ]
    assert detail["lane_predecessors"] == [0]
    assert detail["later_use_predecessor_bytes"] == 128
    assert detail["input_fetch_bytes"] == 64


def test_memory_wait_uses_admission_edges_and_actual_workspace_addresses():
    data = waiting_example()
    data.actions[1]["transfer"].update(start_ns=0, end_ns=4, stall_reasons=[])
    data.actions[0].update(kind="evict")
    data.actions[0]["transfer"].update(
        direction="evict", start_ns=10, end_ns=20, stall_reasons=["lane-busy"]
    )
    data.tasks[1]["stall_reasons"] = ["memory-reuse"]
    data.leases = [
        {
            "task_id": None,
            "alias_group_id": "other",
            "predicted_start_ns": 0,
            "predicted_end_ns": 20,
            "offset": 0,
            "bytes": 128,
            "purpose": "task_output",
        },
        {
            "task_id": "consumer",
            "alias_group_id": None,
            "predicted_start_ns": 20,
            "predicted_end_ns": 25,
            "offset": 64,
            "bytes": 32,
            "purpose": "task_workspace",
        },
    ]
    waits = TaskWaits(
        data,
        {
            "reuse_dependencies": [
                {"predecessor_action_index": 0, "successor_task_id": "consumer"}
            ]
        },
    )
    detail = waits.detail(1)
    assert detail["input_fetch_bytes"] == 0
    assert detail["critical_action"] == 0
    assert detail["blocking_transfers"] == [
        {"action": 0, "relation": "storage_reuse", "purposes": ["task_workspace"]}
    ]


def test_execution_examples_exclude_backlog_without_a_compute_stall():
    from tools.pressurefit_trace.plan.storyboard import execution_examples

    data = waiting_example()
    data.task_detail = TaskWaits(data, {}).detail
    assert execution_examples(data) == [1]
    data.tasks[1].update(ready_ns=20, stall_reasons=[])
    data.task_detail = TaskWaits(data, {}).detail
    assert execution_examples(data) == []


def test_eviction_can_block_a_required_prefetch_and_then_compute():
    from tools.pressurefit_trace.plan.storyboard import execution_examples

    data = waiting_example()
    data.actions[0]["transfer"].update(
        direction="evict", start_ns=5, end_ns=15, stall_reasons=["lane-busy"]
    )
    data.actions[0]["kind"] = "evict"
    data.actions[1]["transfer"].update(start_ns=15, stall_reasons=["memory-reuse"])
    waits = TaskWaits(
        data,
        {
            "reuse_dependencies": [
                {"predecessor_action_index": 0, "successor_action_index": 1}
            ]
        },
    )
    data.task_detail = waits.detail
    assert waits.detail(1)["upstream_storage_dependencies"] == [
        {"action": 0, "successor_action": 1}
    ]
    assert execution_examples(data) == [1]
