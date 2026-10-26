"""Trace drawings preserve planner state while filtering the visible rows."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shadowspill.diagnostics.pressurefit_trace import read_manifest
from shadowspill.ir import (
    MemoryLocation,
    ObjectRole,
    ObjectSpec,
    Persistence,
    ResidencySpec,
)
from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.planner.search.algorithms.pressurefit.tracing import record_plan_trace
from tests.shadowspill.planner._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)
from tools.pressurefit_trace.collect.budgets import execution_budget
from tools.pressurefit_trace.data import (
    Trajectory,
    analyze,
    annotate_recompute,
    short_task,
    task_memory,
)
from tools.pressurefit_trace.explain import decision
from tools.pressurefit_trace.pressure import PressureComposition
from tools.pressurefit_trace.render import (
    Overview,
    Renderer,
    compressed_targets,
    cut_intervals,
    explanation_events,
    paced_steps,
    recompute_endpoint,
    render_frame,
    render_movie,
    spaced_labels,
    subtract_intervals,
)


@pytest.mark.parametrize(
    ("saved_ns", "recompute_ns", "prefix_ns"),
    [(100, 160, 60), (200, 160, 0), (0, 160, 160), (0, 0, 0)],
)
def test_recompute_prefix_uses_matching_alternative_on_both_axes(
    saved_ns, recompute_ns, prefix_ns
):
    program = {
        "tasks": [
            {"task_id": "f", "phase": "forward", "profile_id": "f"},
            {"task_id": "saved", "phase": "backward", "profile_id": "save"},
            {"task_id": "chosen", "phase": "backward", "profile_id": "recompute"},
        ],
        "profiles": [
            {"profile_id": "f", "runtime_ns": 40},
            {"profile_id": "save", "runtime_ns": saved_ns},
            {"profile_id": "recompute", "runtime_ns": recompute_ns},
        ],
        "task_alternative_groups": [
            {
                "group_id": "stage_2_mb_3",
                "options": [
                    {"option_id": "save", "active_task_ids": ["f", "saved"]},
                    {"option_id": "recompute", "active_task_ids": ["f", "chosen"]},
                ],
            }
        ],
    }
    tasks = [
        {"task_id": "f", "phase": "forward", "profiled_duration_ns": 40},
        {
            "task_id": "chosen",
            "phase": "backward",
            "profiled_duration_ns": recompute_ns,
        },
    ]
    annotate_recompute(tasks, program)
    assert "recompute_prefix_ns" not in tasks[0]
    assert tasks[1]["save_backward_task_id"] == "saved"
    assert tasks[1]["save_backward_duration_ns"] == saved_ns
    assert tasks[1]["recompute_extra_ns"] == recompute_ns - saved_ns
    assert tasks[1]["recompute_prefix_ns"] == prefix_ns
    compute = SimpleNamespace(
        tasks=tasks, edges=np.array([0, 40, 40 + recompute_ns]) / 1e9
    )
    indexed = SimpleNamespace(tasks=tasks, edges=np.arange(3, dtype=float))
    assert recompute_endpoint(compute, 0) == 0
    assert recompute_endpoint(compute, 1) == pytest.approx((40 + prefix_ns) / 1e9)
    fraction = prefix_ns / recompute_ns if recompute_ns else 0
    assert recompute_endpoint(indexed, 1) == pytest.approx(1 + fraction)


def test_physical_budget_preserves_captured_runtime_carveout():
    gib = 2**30
    capture = {"physical_budget_bytes": 7 * gib, "execution_budget_bytes": 6 * gib}
    assert execution_budget(capture) == (6 * gib, 7 * gib)
    assert execution_budget(capture, physical_gib=16) == (15 * gib, 16 * gib)
    assert execution_budget(capture, execution_gib=16) == (16 * gib, None)
    with pytest.raises(ValueError, match="no execution capacity"):
        execution_budget(capture, physical_gib=1)
    with pytest.raises(ValueError, match="alternatives"):
        execution_budget(capture, physical_gib=16, execution_gib=15)


@pytest.mark.parametrize("axis", ["compute", "task"])
def test_recompute_prefix_is_drawn_and_recorded(data, axis):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    from tools.pressurefit_trace.render import RECOMPUTE

    position = next(
        i for i, task in enumerate(data.tasks) if task["profiled_duration_ns"] > 0
    )
    task = data.tasks[position]
    task["phase"] = "backward"
    task["stage_index"] = 16
    task["microbatch"] = 14
    duration = task["profiled_duration_ns"]
    task["save_backward_duration_ns"] = duration // 2
    task["recompute_prefix_ns"] = duration - duration // 2
    data.collection_context = {
        "workload": "example",
        "physical_budget_bytes": 7 * 2**30,
        "ordering": "4x2rp",
        "geometry": "4",
        "microbatches": 8,
    }
    data.select_axis(axis)
    renderer = Renderer(data, width=1280)
    state = data.trace.state(0, 0)
    renderer.draw(0, 0, state.spans, state.pressure, state.capacity, boundary=position)
    bands = [
        p for p in renderer.tasks.patches if p.get_facecolor()[:3] == to_rgb(RECOMPUTE)
    ]
    assert len(bands) == 1
    x = bands[0].get_path().transformed(bands[0].get_patch_transform()).vertices[:, 0]
    assert x.min() == data.edges[position]
    assert x.max() == pytest.approx(recompute_endpoint(data, position))
    record = renderer.last_record["recompute_prefixes"][0]
    assert record["label"] == "R16 · mb14"
    assert record["duration_ns"] == task["recompute_prefix_ns"]
    assert record["save_backward_duration_ns"] == task["save_backward_duration_ns"]
    labels = {text.get_text() for text in renderer.tasks.texts}
    assert "R16\nmb14" in labels
    name = "B16 · mb14"
    assert (f"{position}\n{name}" if axis == "task" else name) in labels
    assert any("ordering 4x2rp" in text.get_text() for text in renderer.dynamic_text)
    plt.close(renderer.fig)


@pytest.fixture
def data(tmp_path):
    return record_data(tmp_path)


def record_data(
    tmp_path,
    strategy="headroom-stall",
    final=(),
    *,
    threshold=32,
    capacity=500,
    workspace=False,
    program=None,
    initial=None,
):
    algorithm = PressureFit(
        PressureFitOptions(
            residency_strategies=(strategy,),
            fetch_rules=("packed-fifo",),
        )
    )
    options = SearchOptions(
        generic=GenericPlanningOptions(
            minimum_object_bytes_evict_eligible=threshold, deterministic=True
        ),
        algorithm=algorithm,
        workers=1,
    )
    program = program or training_chain_program(10)
    if workspace:
        program = replace(
            program,
            profiles=tuple(
                replace(p, workspace_bytes=16 * i)
                for i, p in enumerate(program.profiles)
            ),
        )
    result = algorithm(
        program,
        initial_residency=initial or training_chain_initial(10),
        final_residency=final,
        config=training_chain_config(capacity),
        generic=options.generic,
        workers=1,
    )
    path = record_plan_trace(
        result, options=replace(options, trace_directory=str(tmp_path))
    )
    return Trajectory.load(path)


def test_shared_buffer_keeps_its_identity_across_saved_activation_views(tmp_path):
    program = training_chain_program(10)
    program = replace(
        program,
        objects=(
            *(
                replace(obj, role=ObjectRole.ACTIVATION)
                if obj.object_id == "A_0"
                else obj
                for obj in program.objects
            ),
            ObjectSpec("rotary", "A_0", 0, 32, ObjectRole.BUFFER, Persistence.RUN),
        ),
        tasks=tuple(
            replace(task, inputs=(*task.inputs, "rotary"))
            if task.task_id == "f_0"
            else task
            for task in program.tasks
        ),
    )
    data = record_data(
        tmp_path,
        program=program,
        initial=(
            *training_chain_initial(10),
            ResidencySpec("A_0", MemoryLocation.SPILL),
        ),
    )
    alias = next(i for i, a in enumerate(data.aliases) if a["alias_id"] == "A_0")
    path = data.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    row = manifest["contexts"][0]["aliases"][alias]
    assert row["role"] == "buffer"  # New recordings choose the persistent storage.
    assert row["objects"][0]["object_id"] == "rotary"

    # Older recordings put the saved view first and classify the alias as an
    # activation. The reader must repair the display without changing its uses.
    row["objects"].reverse()
    row.update(role="activation", category="saved activations")
    path.write_text(json.dumps(manifest))
    (data.directory / "semantic_labels.json").write_text(
        json.dumps(
            {
                "program_digest": manifest["program_digest"],
                "objects": {"rotary": {"label": "buffer rotary.cosine"}},
            }
        )
    )
    loaded = Trajectory.load(data.directory)
    annotated = read_manifest(data.directory)["candidates"][0]["aliases"][alias]
    assert annotated["role"] == "buffer"
    assert annotated["label"] == "buffer rotary.cosine"
    assert loaded.aliases[alias]["role"] == "buffer"
    assert loaded.aliases[alias]["member_roles"] == ["activation", "buffer"]
    assert loaded.uses == data.uses
    assert loaded.productions == data.productions
    for position in loaded.uses[alias]:
        assert loaded.focus_label(alias, position) == "buffer rotary.cosine"
    parts = PressureComposition(loaded)
    assert parts.alias_roles[alias] == 5  # Other roles, not activations.
    for count, _, pressure, _ in loaded.trace._walk(loaded.trace.answer_reduction):
        np.testing.assert_array_equal(
            parts.at(loaded.trace.answer_reduction, count, pressure).sum(axis=0),
            pressure,
        )


def test_task_memory_unions_shared_views_and_in_place_mutations():
    task = {
        "inputs": ["a", "view", "b"],
        "outputs": ["view", "c"],
        "mutations": [{"object_id": "a"}, {"object_id": "c"}],
    }
    memory = task_memory(task, {"a": 0, "view": 0, "b": 1, "c": 2}, [8, 16, 32], 4)
    assert memory["inputs"] == {"object_count": 3, "alias_count": 2, "bytes": 24}
    assert memory["outputs"] == {"object_count": 2, "alias_count": 2, "bytes": 40}
    assert memory["mutated"] == {"object_count": 2, "alias_count": 2, "bytes": 40}
    assert memory["workspace_bytes"] == 4
    assert memory["total"] == {"object_count": 4, "alias_count": 3, "bytes": 60}
    assert memory["overlap_bytes"] == 48


def test_object_capacity_survives_repairs_but_not_unknown_refinements(data):
    kinds = (0, 1, 3, 2, 4, 3, 1)
    reductions = tuple(
        replace(data.trace.reductions[0], reason=np.array([kind] + [0] * 8))
        for kind in kinds
    )
    steps = np.zeros((3, 7), dtype=np.uint64)
    steps[:, 1] = [1024, 768, 768]
    trace = replace(
        data.trace,
        reductions=reductions,
        steps=steps,
        step_reductions=np.array([[0, 0], [1, 2], [2, 2]]),
    )
    assert [trace.object_capacity_bytes(i) for i in range(7)] == [
        1024,
        1024,
        768,
        768,
        768,
        None,
        None,
    ]
    legacy = replace(
        trace, reductions=tuple(replace(r, reason=None) for r in reductions)
    )
    assert legacy.object_capacity_bytes(0) == 1024
    assert legacy.object_capacity_bytes(1) is None
    assert legacy.object_capacity_bytes(2) == 768


def test_focus_orders_required_storage_by_size_then_role_and_pins_selected():
    data = SimpleNamespace(
        tasks=[{}] * 10,
        aliases=[
            {"role": role}
            for role in (
                "parameter",
                "activation",
                "gradient",
                "buffer",
                "activation",
                "parameter",
                "optimizer_state",
                "activation",
            )
        ],
        trace=SimpleNamespace(
            alias_sizes=np.array([64, 64, 64, 1024, 128, 32, 128, 8])
        ),
        order=np.arange(8),
        y_rank=np.arange(8),
        uses=[[1], [1], [1], [1], [2], [1], [1], [9]],
    )
    visible, local = Trajectory.focus_aliases(data, 1, 7, 1, 7)
    assert local == [7, 6, 0, 1, 2, 5, 4, 3]
    assert visible == local[:7]
    # Even the largest required buffer cannot displace a main-role neighbor.
    assert 3 not in visible
    fewer, _ = Trajectory.focus_aliases(data, 1, 7, 0, 7)
    assert fewer == [7, 6, 0, 1, 2, 5, 3]
    pinned, _ = Trajectory.focus_aliases(data, 1, 3, 1, 7)
    assert pinned[0] == 3
    np.testing.assert_array_equal(data.order, np.arange(8))


@pytest.mark.parametrize("strategy", ["headroom-stall", "tight-stall"])
@pytest.mark.parametrize("final", [(), (ResidencySpec("W_0", MemoryLocation.DEVICE),)])
def test_role_composition_reconciles_all_cuts_and_restarts(tmp_path, strategy, final):
    data = record_data(tmp_path, strategy, final)
    composition = PressureComposition(data)
    assert composition.attributed
    for rid, _ in enumerate(data.trace.reductions):
        for count, _, pressure, _ in data.trace._walk(rid):
            roles = composition.at(rid, count, pressure)
            np.testing.assert_array_equal(roles.sum(axis=0), pressure)
            assert np.all(roles >= 0)
        seed = data.trace.state(rid, 0)
        np.testing.assert_array_equal(
            composition.at(rid, 0, seed.pressure).sum(axis=0), seed.pressure
        )
    data.pressure_accounting = None
    legacy = PressureComposition(data)
    assert not legacy.attributed
    final = data.trace.state(0)
    np.testing.assert_array_equal(
        legacy.at(0, final.cut_count, final.pressure).sum(axis=0), final.pressure
    )


def test_pressure_axis_zooms_without_clipping_budget_or_pressure(data):
    seed, final = data.trace.state(0, 0), data.trace.state(0)
    renderer = Renderer(data, width=1280)
    ceilings = []
    for state in (seed, final):
        renderer.draw(0, state.cut_count, state.spans, state.pressure, state.capacity)
        floor, ceiling = renderer.pressure.get_ylim()
        assert floor == 0
        assert ceiling > max(state.pressure.max(), state.capacity.max()) / 2**30
        ceilings.append(ceiling)
    assert ceilings[1] < ceilings[0]
    renderer.pressure_scale = "fixed"
    renderer.draw(0, final.cut_count, final.spans, final.pressure, final.capacity)
    assert renderer.pressure.get_ylim()[1] == renderer.pressure_max
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_resident_pool_includes_hidden_mutated_state_without_changing_excess(tmp_path):
    data = record_data(tmp_path, threshold=65, capacity=5000)
    alias = next(i for i, a in enumerate(data.aliases) if a["alias_id"] == "dW_head")
    data.aliases[alias]["role"] = "optimizer_state"
    assert data.aliases[alias]["hidden_reason"] == "below eviction threshold"
    reservation = next(
        row
        for row in data.pressure_accounting["resident_reservations"]
        if row[0] == alias
    )
    assert reservation[1] == 2  # Old and new generations both need a home.
    assert reservation[2] >= 2 * int(data.trace.alias_sizes[alias])
    parts = PressureComposition(data, include_resident=True)
    state = data.trace.state(0)
    values = parts.at(0, state.cut_count, state.pressure)
    assert (
        parts.resident_roles.sum() == data.pressure_accounting["resident_slice_bytes"]
    )
    assert np.all(values[2] == reservation[2])  # Purple optimizer layer persists.
    total = values.sum(axis=0)
    np.testing.assert_array_equal(total, state.pressure + parts.offset)
    np.testing.assert_array_equal(
        total - (state.capacity.astype(np.int64) + parts.offset),
        state.pressure.astype(np.int64) - state.capacity.astype(np.int64),
    )


@pytest.mark.parametrize("include_resident", [False, True])
def test_flat_budget_and_workspace_preserve_every_boundary_excess(
    tmp_path, include_resident
):
    data = record_data(tmp_path, workspace=True)
    composition = PressureComposition(data, include_resident=include_resident)
    assert set(composition.workspace) == {0, 16, 32, 48}
    assert composition.workspace[-1] == 0
    expected_budget = 500 - (0 if include_resident else composition.resident_slice)
    assert composition.flat_budget == expected_budget
    for rid, _ in enumerate(data.trace.reductions):
        for count, _, pressure, capacity in data.trace._walk(rid):
            roles = composition.display(rid, count, pressure, capacity)
            np.testing.assert_array_equal(roles[-2], composition.workspace)
            np.testing.assert_array_equal(
                roles.sum(axis=0) - expected_budget,
                pressure.astype(np.int64) - capacity.astype(np.int64),
            )
            assert np.all(roles >= 0)
    renderer = Renderer(
        data, width=1280, pressure_view="total" if include_resident else "reducer"
    )
    state = data.trace.state(0)
    renderer.draw(0, state.cut_count, state.spans, state.pressure, state.capacity)
    budget_line = renderer.pressure.lines[2].get_ydata()
    assert np.all(budget_line == expected_budget / 2**30)
    assert renderer.last_record["display_budget_bytes"] == expected_budget
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


@pytest.mark.parametrize("axis", ["compute", "task"])
def test_next_use_reports_task_boundary_and_profiled_time_on_both_axes(data, axis):
    data.select_axis(axis)
    row = data.trace.reductions[0].cuts[0]
    assert data.aliases[int(row[4])]["alias_id"] == "W_0"
    assert int(row[0]) == 10  # Before f_9; W_0 is next required by r_0.
    assert data.next_use(row) == 30
    assert data.tasks[30]["task_id"] == "r_0"
    # Trigger to use: f_9 (10 us), head (2 us), nine backwards (20 us each).
    assert data.time_to_use_ns(row, 30) == 192_000
    assert data.departure_position(row) == 2
    assert data.gap_duration_ns(row, 30) == 272_000
    later = data.trace.reductions[0].cuts[1]
    assert data.aliases[int(later[4])]["alias_id"] == "W_1"
    assert int(later[0]) == 10
    assert data.departure_position(later) == 3  # After f_1, before the trigger.
    assert data.next_use(later) == 28
    assert data.gap_duration_ns(later, 28) == 242_000
    assert data.time_to_use_ns(later, 28) == 172_000
    state = data.trace.state(0, 1)
    renderer = Renderer(data, width=1280)
    renderer.draw(0, 1, state.spans, state.pressure, state.capacity, event=0)
    highlight = renderer.last_record["cut_highlight"]
    assert highlight["action"] == "release"
    assert highlight["departure_position"] == 2
    assert highlight["next_use_position"] == 30
    assert highlight["compute_gap_ns"] == 272_000
    assert highlight["compute_until_next_use_ns"] == 192_000
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_filters_and_chronological_order_preserve_pressure(data):
    assert data.minimum_bytes == 32
    reasons = data.row_filter_summary()["hidden"]
    assert reasons["below eviction threshold"] == 1
    assert reasons["required use only"] > 0
    first = [(data.uses[a][0], int(a)) for a in data.order]
    assert first == sorted(first)
    for rid, reduction in enumerate(data.trace.reductions):
        for state in data.trace.iter_states(rid):
            data.assert_required_covered(state.spans)
        assert set(int(a) for a in reduction.cuts[:, 4]) <= set(data.order)
    seed = data.trace.state(0, 0)
    before = seed.pressure.copy()
    raster = Overview(data, seed.spans, height=20, width=80)
    assert np.isfinite(raster.image()).all()
    np.testing.assert_array_equal(seed.pressure, before)


def test_required_only_optimizer_rows_remain_visible_above_threshold(data):
    optimizer = next(
        i
        for i, a in enumerate(data.aliases)
        if a["hidden_reason"] == "required use only"
    )
    small = next(
        i
        for i, a in enumerate(data.aliases)
        if a["hidden_reason"] == "below eviction threshold"
    )
    before = data.trace.initial.copy()
    data.aliases[optimizer]["role"] = "optimizer_state"
    data.aliases[small]["role"] = "optimizer_state"
    data.filter_rows()
    assert optimizer in data.order
    assert small not in data.order
    assert data.aliases[optimizer]["hidden_reason"] is None
    first = [(data.uses[a][0], int(a)) for a in data.order]
    assert first == sorted(first)
    np.testing.assert_array_equal(data.trace.initial, before)


def test_incremental_raster_matches_exact_state(data):
    walk = data.trace._walk(0)
    _, spans, _, _ = next(walk)
    drawing = Overview(data, spans, height=20, width=80)
    for index, (_, spans, _, _) in enumerate(walk):
        drawing.update(spans, [int(data.trace.reductions[0].cuts[index, 4])])
    independent = Overview(data, spans, height=20, width=80)
    np.testing.assert_allclose(drawing.image(), independent.image(), atol=1e-13)


@pytest.mark.parametrize("axis", ["compute", "task"])
def test_cut_highlight_matches_replay_and_keeps_required_intervals(data, axis):
    data.select_axis(axis)
    walk = data.trace.iter_states(0)
    before = next(walk)
    for row, after in zip(data.trace.reductions[0].cuts, walk, strict=True):
        alias = int(row[4])
        removed = cut_intervals(data, before.spans, row)
        assert removed == subtract_intervals(
            data.intervals(alias, before.spans[alias]),
            data.intervals(alias, after.spans[alias]),
        )
        if removed:
            assert removed[0][0] == data.edges[data.departure_position(row)]
        for a, b in removed:
            assert all(
                min(b, data.edges[task + 1]) <= max(a, data.edges[task])
                for task in data.uses[alias]
            )
        before = after
    state = data.trace.state(0, 1)
    seed = data.trace.state(0, 0)
    removed = cut_intervals(data, seed.spans, data.trace.reductions[0].cuts[0])
    renderer = Renderer(data, width=1280)
    renderer.draw(
        0,
        1,
        state.spans,
        state.pressure,
        state.capacity,
        event=0,
        removed=removed,
    )
    highlight = renderer.last_record["cut_highlight"].copy()
    assert highlight["removed_intervals"] == removed
    assert highlight["status"] == "applied"
    renderer.rgba(flash=1)
    renderer.rgba(flash=0)
    assert all(not patch.get_visible() for patch in renderer.flashes)
    assert renderer.last_record["cut_highlight"] == highlight
    np.testing.assert_array_equal(state.pressure, data.trace.state(0, 1).pressure)
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_analysis_retains_every_decision_and_pressure(data, tmp_path):
    report = analyze(data, tmp_path / "analysis")
    records = [
        json.loads(s)
        for s in (tmp_path / "analysis" / "decisions.jsonl").read_text().splitlines()
    ]
    assert len(records) == sum(len(r.cuts) for r in data.trace.reductions)
    assert report["row_filter"]["pressure_includes_all_aliases"]
    for record in records:
        row = data.trace.reductions[record["reduction"]].cuts[record["cut"] - 1]
        assert len(record["choices"]) == int(row[2]) + 1
        assert record["boundaries"][0]["position"] == int(row[0])
        assert record["choices"][0]["size_bytes"] == int(
            data.trace.alias_sizes[int(row[4])]
        )
    for rid, row in enumerate(report["reductions"]):
        assert row["final_peak_bytes"] == int(data.trace.state(rid).pressure.max())


def test_explanation_uses_the_planner_comparator(tmp_path):
    data = record_data(tmp_path)
    value = decision(data, 0, 0)
    assert value["ranking_criteria"] == [
        "exposed_ns",
        "writeback",
        "negative_slack_ns",
        "negative_size_bytes",
        "alias",
        "start",
        "cut_id",
    ]
    for cut in value["choices"]:
        component = cut["score_components"]
        assert cut["slack_ns"] == (
            component["evict_slack_ns"]
            + component["fetch_slack_ns"]
            - component["evict_ns"]
            - component["fetch_ns"]
        )
    renderer = Renderer(data, width=1280)
    state = data.trace.state(0, 0)
    renderer.draw(
        0, 0, state.spans, state.pressure, state.capacity, event=0, explain=True
    )
    renderer.rgba()
    text = "\n".join(t.get_text() for ax in renderer.fig.axes for t in ax.texts)
    assert "I ↑" not in text and "L ↑" not in text and "f ↑" not in text
    assert "Slack (ms) ↑" in text
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


@pytest.mark.parametrize("axis", ["compute", "task"])
def test_frame_and_explanation_render_without_changing_state(data, tmp_path, axis):
    order = data.order.copy()
    data.select_axis(axis)
    np.testing.assert_array_equal(data.order, order)
    if axis == "task":
        np.testing.assert_array_equal(np.diff(data.edges), np.ones(len(data.tasks)))
    data.assert_required_covered(data.trace.state(0, 0).spans)
    path = tmp_path / "explanation.png"
    render_frame(data, path, reduction=0, cut=0, width=1280, explain=True)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    meta = json.loads(path.with_suffix(".json").read_text())
    state = data.trace.state(0, 0)
    assert meta["peak_pressure_bytes"] == int(state.pressure.max())
    assert meta["explanation"]
    assert meta["axis"] == axis
    assert meta["event_index"] == 0
    chosen = decision(data, 0, 0)["choices"][0]["alias_index"]
    assert meta["focus_aliases"][0] == chosen
    assert chosen in meta["all_local_aliases"]
    renderer = Renderer(data, width=1280, focus_mode="boundary")
    renderer.draw(
        0, 0, state.spans, state.pressure, state.capacity, event=0, explain=True
    )
    assert renderer.explanation.get_ylim() == (0, 1)
    assert not renderer.focus.get_visible()
    assert not renderer.tasks.get_visible()
    # Ensure an explanation pause cannot obscure the bottom pressure panel.
    assert renderer.explanation.get_position().y0 > renderer.pressure.get_position().y1
    position = renderer.last_record["boundary_position"]
    task = renderer.last_record["current_task"]
    assert task["task_id"] == data.tasks[position]["task_id"]
    assert task["required_memory"] == data.tasks[position]["required_memory"]
    side_labels = {text.get_text() for text in renderer.side.texts}
    assert "Workspace" in side_labels
    assert (
        f"Inputs ({task['required_memory']['inputs']['object_count']:,})" in side_labels
    )
    first, last = max(0, position - 3), min(len(data.tasks), position + 4)
    labels = {text.get_text() for text in renderer.tasks.texts}
    assert {f"b{i}" for i in range(first, last + 1)} <= labels
    for i in range(first, last):
        name = short_task(data.tasks[i])
        assert (f"{i}\n{name}" if axis == "task" else name) in labels
    renderer.draw(0, 0, state.spans, state.pressure, state.capacity, event=0)
    assert renderer.focus.get_visible() and renderer.tasks.get_visible()
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_close_boundary_labels_remain_separate_and_inside_the_strip():
    anchors = np.array([0, 0, 0.1, 0.9, 0.90001, 1])
    widths = np.array([0.04, 0.04, 0.05, 0.05, 0.06, 0.06])
    centers = spaced_labels(anchors, widths, 0.01)
    assert np.all(centers - widths / 2 >= -1e-12)
    assert np.all(centers + widths / 2 <= 1 + 1e-12)
    assert np.all(np.diff(centers) >= (widths[:-1] + widths[1:]) / 2 + 0.01 - 1e-12)
    np.testing.assert_array_equal(anchors, [0, 0, 0.1, 0.9, 0.90001, 1])


def test_acceleration_never_skips_the_final_state_or_invents_cuts():
    for first, total, frames in [(6, 7587, 200), (6, 7, 200), (0, 1, 1), (6, 6, 10)]:
        targets = compressed_targets(first, total, frames)
        assert targets == sorted(set(targets))
        assert all(first < x <= total for x in targets)
        assert not targets if first == total else targets[-1] == total
    assert subtract_intervals([(0, 10)], [(0, 2), (4, 6), (8, 10)]) == [(2, 4), (6, 8)]


@pytest.mark.parametrize("axis", ["compute", "task"])
def test_cut_view_spans_required_endpoints_and_can_return_to_boundary(data, axis):
    data.select_axis(axis)
    state = data.trace.state(0, 2)
    renderer = Renderer(data, width=1280)
    args = (0, 2, state.spans, state.pressure, state.capacity)
    renderer.draw(*args, event=1)
    record = renderer.last_record
    journey = record["object_journey"]
    assert journey["prior_use_position"] == 2
    assert journey["departure_position"] == 3
    assert journey["next_use_position"] == 28
    assert record["focus_aliases"] == [int(data.trace.reductions[0].cuts[1, 4])]
    left, right = renderer.focus.get_xlim()
    assert left <= data.edges[2] and right >= data.edges[29]
    assert (
        len(renderer.tasks.patches)
        == record["focus_task_range"][1] - record["focus_task_range"][0]
    )
    renderer.draw(*args, event=1, journey=True, zoom_progress=0)
    zoomed_limits = renderer.focus.get_xlim()
    renderer.draw(*args, event=1, journey=False)
    assert "object_journey" not in renderer.last_record
    assert renderer.focus.get_xlim() == zoomed_limits
    assert len(renderer.last_record["focus_aliases"]) > 1
    np.testing.assert_array_equal(state.pressure, data.trace.state(0, 2).pressure)
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_paced_steps_keep_the_requested_duration_even_with_few_cuts():
    for first, total, frames, hold in [
        (0, 1, 1350, 3),
        (5, 9, 60, 6),
        (6, 7587, 1350, 3),
        (1, 2845, 720, 18),
    ]:
        steps = paced_steps(first, total, frames, hold_frames=hold)
        assert sum(duration for _, duration in steps) == frames
        targets = [target for target, _ in steps]
        assert targets == sorted(set(targets))
        assert first < targets[0] <= targets[-1] == total
        assert all(duration >= hold for _, duration in steps)
    assert paced_steps(7, 7, 100) == []


def test_auto_examples_follow_recorded_reasons_and_writeback():
    cuts = np.zeros((80, 64), dtype=np.uint64)
    cuts[40:, 8] = 1
    trace = SimpleNamespace(
        reductions=[SimpleNamespace(cuts=cuts)],
        alias_sizes=np.array([10, 1000]),
        comparisons=lambda rid, index: [
            {"criterion": "alias" if index < 10 else "negative_size_bytes"}
        ],
    )
    data = SimpleNamespace(trace=trace, aliases=[{"role": "parameter"}] * 2)
    # The early scan is deliberately bounded; use an early distinct reason.
    trace.comparisons = lambda rid, index: [
        {"criterion": "alias" if index < 7 else "negative_size_bytes"}
    ]
    assert explanation_events(data, 0, 3) == [0, 7, 40]
    # A much larger object with a longer gap beats a similar sibling even
    # when only the sibling has a different recorded comparison criterion.
    cuts[6, 4:7] = [1, 0, 30]
    assert explanation_events(data, 0, 3) == [0, 6, 40]
    # An early eviction must not put the third teaching card immediately
    # after the second; reserve a later part of the trajectory for it.
    cuts[13:, 8] = 1
    assert explanation_events(data, 0, 3) == [0, 6, len(cuts) // 3]
    assert explanation_events(data, 0, 0) == []
    trace.reductions[0].cuts = cuts[:2]
    assert explanation_events(data, 0, 3) == [0, 1]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is optional")
def test_spaced_movie_uses_real_states_and_exact_frame_timing(data, tmp_path):
    total = len(data.trace.reductions[0].cuts)
    examples = sorted({1, min(3, total), total})
    output = tmp_path / "spaced.mp4"
    report = render_movie(
        data,
        output,
        reductions=[0],
        width=640,
        fps=5,
        detail_at=examples,
        slow_seconds=0.4,
        explain_seconds=0.2,
        step_seconds=0.2,
        bridge_seconds=0.4,
        bridge_hold_seconds=0.2,
        accelerated_seconds=0.4,
        accelerated_hold_seconds=0.2,
        seed_seconds=0.2,
        final_seconds=0.2,
        zoom_seconds=0.2,
        zoom_hold_seconds=0.4,
    )
    explained = [s["event_index"] + 1 for s in report["shots"] if s.get("explanation")]
    assert explained == examples
    assert report["shots"][-1]["cut_count"] == total
    frame = 0
    for shot in report["shots"]:
        assert shot["first_video_frame"] == frame
        frame += shot["video_frame_count"]
        state = data.trace.state(0, shot["cut_count"])
        assert shot["peak_pressure_bytes"] == int(state.pressure.max())
    assert frame == report["frames"]
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(output),
            ]
        )
    )
    assert float(probe["format"]["duration"]) == report["seconds"]
