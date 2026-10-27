"""Trace analysis preserves planner state, accounting, and recorded decisions."""

from __future__ import annotations

import json
from dataclasses import replace

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
    task_memory,
)
from tools.pressurefit_trace.explain import decision
from tools.pressurefit_trace.pressure import PressureComposition


@pytest.mark.parametrize(
    ("saved_ns", "recompute_ns", "prefix_ns"),
    [(100, 160, 60), (200, 160, 0), (0, 160, 160), (0, 0, 0)],
)
def test_recompute_prefix_uses_matching_alternative(saved_ns, recompute_ns, prefix_ns):
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
