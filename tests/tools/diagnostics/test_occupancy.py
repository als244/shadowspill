"""Pool occupancy attributed to objects, on a plan small enough to check by hand."""

from __future__ import annotations

import json
from pathlib import Path

from tools.diagnostics.occupancy import (
    SUMMARY_COLUMNS,
    WORKSPACE,
    Clock,
    ProgramFacts,
    all_save,
    attribute,
    categorize,
    cheapest_selections,
    program_path_for,
    selected_task_ids,
    summarize,
    table,
    unconstrained,
    write_pages,
)

MIB = 1 << 20
S = 1_000_000_000  # nanoseconds in a second


def _program() -> dict:
    """Three stages of one microbatch and one weight.

    ``w`` is a retained weight with a checkpoint copy in the spill pool.
    ``t1`` (forward) makes ``a``, which ``t2`` (forward) reads as ``a_view``
    -- the same storage under a second object id -- and ``t3`` (backward)
    reads again; ``t3`` makes the model gradients ``g`` and ``gb`` that
    ``t4`` (the optimizer) consumes, and ``gb`` takes over ``b``'s storage:
    the layout gives it no lease of its own. ``t2``'s output ``b`` is
    recomputed by ``t3r`` in the alternative the selection does not choose.
    """

    return {
        "schema": "shadowspill.program/v1",
        "devices": [
            {"device_id": "cuda_0", "index": 0, "kind": "cuda", "process_id": "p"}
        ],
        "alias_groups": [
            {
                "alias_group_id": "A_w",
                "device_id": "cuda_0",
                "initial_version": 0,
                "shared_residency": None,
                "size_bytes": 100 * MIB,
                "retain_spill_copy": True,
            },
            {
                "alias_group_id": "A_a",
                "device_id": "cuda_0",
                "initial_version": 0,
                "shared_residency": None,
                "size_bytes": 64 * MIB,
                "retain_spill_copy": False,
            },
            {
                "alias_group_id": "A_b",
                "device_id": "cuda_0",
                "initial_version": 0,
                "shared_residency": None,
                "size_bytes": 32 * MIB,
                "retain_spill_copy": False,
            },
            {
                "alias_group_id": "A_g",
                "device_id": "cuda_0",
                "initial_version": 0,
                "shared_residency": None,
                "size_bytes": 100 * MIB,
                "retain_spill_copy": False,
            },
            {
                "alias_group_id": "A_x",
                "device_id": "cuda_0",
                "initial_version": 0,
                "shared_residency": None,
                "size_bytes": 8 * MIB,
                "retain_spill_copy": False,
            },
            {
                "alias_group_id": "A_gb",
                "device_id": "cuda_0",
                "initial_version": 0,
                "shared_residency": None,
                "size_bytes": 32 * MIB,
                "retain_spill_copy": False,
            },
        ],
        "objects": [
            {
                "object_id": "w",
                "alias_group_id": "A_w",
                "offset_bytes": 0,
                "size_bytes": 100 * MIB,
                "role": "parameter",
                "persistence": "checkpoint",
            },
            {
                "object_id": "x",
                "alias_group_id": "A_x",
                "offset_bytes": 0,
                "size_bytes": 8 * MIB,
                "role": "activation",
                "persistence": "step",
            },
            {
                "object_id": "a",
                "alias_group_id": "A_a",
                "offset_bytes": 0,
                "size_bytes": 64 * MIB,
                "role": "activation",
                "persistence": "step",
            },
            {
                "object_id": "a_view",
                "alias_group_id": "A_a",
                "offset_bytes": 0,
                "size_bytes": 64 * MIB,
                "role": "activation",
                "persistence": "step",
            },
            {
                "object_id": "b",
                "alias_group_id": "A_b",
                "offset_bytes": 0,
                "size_bytes": 32 * MIB,
                "role": "activation",
                "persistence": "step",
            },
            {
                "object_id": "b_again",
                "alias_group_id": "A_b",
                "offset_bytes": 0,
                "size_bytes": 32 * MIB,
                "role": "activation",
                "persistence": "step",
            },
            {
                "object_id": "g",
                "alias_group_id": "A_g",
                "offset_bytes": 0,
                "size_bytes": 100 * MIB,
                "role": "gradient",
                "persistence": "step",
            },
            {
                "object_id": "gb",
                "alias_group_id": "A_gb",
                "offset_bytes": 0,
                "size_bytes": 32 * MIB,
                "role": "gradient",
                "persistence": "step",
            },
        ],
        "profiles": [
            {
                "profile_id": "p_one",
                "runtime_ns": 1_000_000_000,
                "workspace_bytes": 0,
                "compatibility_digest": "abi",
            },
            {
                "profile_id": "p_save",
                "runtime_ns": 2_000_000_000,
                "workspace_bytes": 40 * MIB,
                "compatibility_digest": "abi",
            },
            {
                "profile_id": "p_recompute",
                "runtime_ns": 2_500_000_000,
                "workspace_bytes": 0,
                "compatibility_digest": "abi",
            },
        ],
        "tasks": [
            {
                "task_id": "t1",
                "phase": "forward",
                "resource": {"device_id": "cuda_0", "kind": "compute", "lane": 0},
                "dependencies": [],
                "requires_entrypoint": False,
                "inputs": ["x", "w"],
                "outputs": ["a"],
                "mutations": [],
                "profile_id": "p_one",
            },
            {
                "task_id": "t2",
                "phase": "forward",
                "resource": {"device_id": "cuda_0", "kind": "compute", "lane": 0},
                "dependencies": ["t1"],
                "requires_entrypoint": False,
                "inputs": ["a_view", "w"],
                "outputs": ["b"],
                "mutations": [],
                "profile_id": "p_one",
            },
            {
                "task_id": "t3",
                "phase": "backward",
                "resource": {"device_id": "cuda_0", "kind": "compute", "lane": 0},
                "dependencies": ["t1", "t2"],
                "requires_entrypoint": False,
                "inputs": ["a", "b", "w"],
                "outputs": ["g", "gb"],
                "mutations": [],
                "profile_id": "p_save",
            },
            {
                "task_id": "t3r",
                "phase": "backward",
                "resource": {"device_id": "cuda_0", "kind": "compute", "lane": 0},
                "dependencies": ["t1"],
                "requires_entrypoint": False,
                "inputs": ["a", "w"],
                "outputs": ["b_again", "g"],
                "mutations": [],
                "profile_id": "p_recompute",
            },
            {
                "task_id": "t4",
                "phase": "optimizer",
                "resource": {"device_id": "cuda_0", "kind": "compute", "lane": 0},
                "dependencies": ["t3", "t3r"],
                "requires_entrypoint": False,
                "inputs": ["g", "gb", "w"],
                "outputs": [],
                "mutations": [{"object_id": "w", "version_delta": 1}],
                "profile_id": "p_one",
            },
        ],
        "task_alternative_groups": [
            {
                "group_id": "stage_3",
                "options": [
                    {
                        "option_id": "save",
                        "active_task_ids": ["t3"],
                        "retained_alias_group_ids": ["A_b"],
                    },
                    {
                        "option_id": "recompute",
                        "active_task_ids": ["t3r"],
                        "retained_alias_group_ids": [],
                    },
                ],
            }
        ],
    }


def _selection(option: str = "save") -> dict:
    """The schedule and its simulated evidence for the program above.

    ``a`` is evicted after ``t2`` and fetched for ``t3``; ``b`` is never
    spilled, its storage is ``gb``'s from ``t3`` and released after ``t4``;
    the weight is fetched at the start and stays. Times are in whole seconds
    so the tables read plainly.
    """

    return {
        "program_digest": "ab" + "0" * 62,
        "selections": [{"group_id": "stage_3", "option_id": option}],
        "schedule": {
            "initial_residency": [
                {"alias_group_id": "A_w", "location": "host"},
                {"alias_group_id": "A_x", "location": "host"},
            ],
            "actions": [
                {"alias_group_id": "A_w", "kind": "fetch", "trigger_task_id": "t1"},
                {"alias_group_id": "A_x", "kind": "fetch", "trigger_task_id": "t1"},
                {"alias_group_id": "A_a", "kind": "evict", "trigger_task_id": "t2"},
                {"alias_group_id": "A_a", "kind": "fetch", "trigger_task_id": "t3"},
                {"alias_group_id": "A_b", "kind": "release", "trigger_task_id": "t4"},
                {"alias_group_id": "A_a", "kind": "release", "trigger_task_id": "t3"},
            ],
        },
        "simulation": {
            "spill_capacity_bytes": 400 * MIB,
            "devices": [
                {
                    "device_id": "d",
                    "fetch_bandwidth_bytes_per_second": 10_000_000_000,
                    "evict_bandwidth_bytes_per_second": 20_000_000_000,
                    "fetch_latency_ns": 4000,
                    "evict_latency_ns": 5000,
                }
            ],
        },
        "simulation_result": {
            "makespan_ns": 10 * S,
            "spill_peak_bytes": 164 * MIB,
            "device_peaks": [{"device_id": "d", "total_bytes": 240 * MIB}],
            "task_intervals": [
                {"task_id": "t1", "start_ns": 1 * S, "end_ns": 2 * S},
                {"task_id": "t2", "start_ns": 2 * S, "end_ns": 3 * S},
                {"task_id": "t3", "start_ns": 6 * S, "end_ns": 8 * S},
                {"task_id": "t3r", "start_ns": 6 * S, "end_ns": int(8.5 * S)},
                {"task_id": "t4", "start_ns": 8 * S, "end_ns": 9 * S},
            ],
            "transfer_intervals": [
                {
                    "sequence": 0,
                    "alias_group_id": "A_w",
                    "kind": "fetch",
                    "direction": "fetch",
                    "trigger_task_id": "t1",
                    "start_ns": 0,
                    "end_ns": 1 * S,
                    "bytes": 100 * MIB,
                },
                {
                    "sequence": 1,
                    "alias_group_id": "A_x",
                    "kind": "fetch",
                    "direction": "fetch",
                    "trigger_task_id": "t1",
                    "start_ns": 0,
                    "end_ns": 1 * S,
                    "bytes": 8 * MIB,
                },
                {
                    "sequence": 2,
                    "alias_group_id": "A_a",
                    "kind": "evict",
                    "direction": "evict",
                    "trigger_task_id": "t2",
                    "start_ns": 3 * S,
                    "end_ns": 4 * S,
                    "bytes": 64 * MIB,
                },
                {
                    "sequence": 3,
                    "alias_group_id": "A_a",
                    "kind": "fetch",
                    "direction": "fetch",
                    "trigger_task_id": "t3",
                    "start_ns": 5 * S,
                    "end_ns": 6 * S,
                    "bytes": 64 * MIB,
                },
            ],
        },
        "admission_certificate": {
            "layout": {
                "pool_capacity_bytes": 300 * MIB,
                "placements": [
                    {
                        "purpose": "fetch_destination",
                        "alias_group_id": "A_w",
                        "bytes": 100 * MIB,
                        "predicted_start_ns": 0,
                        "predicted_end_ns": 10 * S,
                    },
                    {
                        "purpose": "task_output",
                        "alias_group_id": "A_a",
                        "bytes": 64 * MIB,
                        "predicted_start_ns": 1 * S,
                        "predicted_end_ns": 4 * S,
                        "task_id": "t1",
                    },
                    {
                        "purpose": "fetch_destination",
                        "alias_group_id": "A_a",
                        "bytes": 64 * MIB,
                        "predicted_start_ns": 5 * S,
                        "predicted_end_ns": 8 * S,
                    },
                    {
                        "purpose": "task_output",
                        "alias_group_id": "A_b",
                        "bytes": 32 * MIB,
                        "predicted_start_ns": 2 * S,
                        "predicted_end_ns": 9 * S,
                        "task_id": "t2",
                    },
                    {
                        "purpose": "task_workspace",
                        "alias_group_id": None,
                        "bytes": 40 * MIB,
                        "predicted_start_ns": 6 * S,
                        "predicted_end_ns": 8 * S,
                        "task_id": "t3",
                    },
                    {
                        "purpose": "task_output",
                        "alias_group_id": "A_g",
                        "bytes": 100 * MIB,
                        "predicted_start_ns": 6 * S,
                        "predicted_end_ns": 9 * S,
                        "task_id": "t3",
                    },
                ],
            }
        },
    }


def _diagnostics() -> dict:
    """The step above as the device ran it: every task half a second, the
    transfers by the simulated sequence they keep."""

    def lane(started: float, finished: float) -> dict:
        return {
            "lane_issued_at_seconds": started,
            "lane_started_at_seconds": started,
            "lane_finished_at_seconds": finished,
        }

    return {
        "summary": {"simulator_makespan_seconds": 10.0},
        "tasks": {
            f"record_{index}": {
                "task_id": task_id,
                "semantic_name": f"microbatch_0.stage_{index}.{phase}",
                "compute": {
                    "compute_started_at_seconds": start,
                    "compute_finished_at_seconds": start + 0.5,
                },
            }
            for index, (task_id, phase, start) in enumerate(
                [
                    ("t1", "forward", 0.9),
                    ("t2", "forward", 1.5),
                    ("t3", "backward", 4.0),
                    ("t4", "optimizer", 5.0),
                ]
            )
        },
        "transfers": {
            "fetch": {
                "fetch_0": {"sequence": 0, "lane": lane(0.0, 0.8)},
                "fetch_1": {"sequence": 1, "lane": lane(0.0, 0.7)},
                "fetch_3": {"sequence": 3, "lane": lane(3.0, 3.9)},
            },
            "evict": {"evict_2": {"sequence": 2, "lane": lane(2.0, 2.6)}},
        },
    }


def test_only_the_selected_alternative_executes() -> None:
    assert selected_task_ids(
        _program(), [{"group_id": "stage_3", "option_id": "save"}]
    ) == [
        "t1",
        "t2",
        "t3",
        "t4",
    ]
    assert "t3r" in selected_task_ids(
        _program(), [{"group_id": "stage_3", "option_id": "recompute"}]
    )


def test_categories_follow_role_and_the_phases_that_touch_an_object() -> None:
    assert categorize("parameter", None, [], True) == "weights"
    assert categorize("optimizer_state", None, ["optimizer"], True) == "optimizer state"
    assert categorize("gradient", "backward", ["optimizer"], False) == "model gradients"
    assert categorize("gradient", "backward", ["backward"], True) == "model gradients"
    assert categorize("gradient", "backward", ["backward"], False) == "tangents"
    assert (
        categorize("activation", "forward", ["forward"], False) == "saved activations"
    )
    assert (
        categorize("activation", "forward", ["backward"], False) == "saved activations"
    )
    assert (
        categorize("activation", "backward", ["backward"], False)
        == "recomputed activations"
    )
    assert categorize("activation", None, ["forward"], False) == "inputs"
    assert categorize("control", "forward", [], False) == "control"


def test_a_view_is_not_a_generation_and_an_unselected_output_does_not_exist() -> None:
    facts = ProgramFacts.build(_program(), _selection()["selections"])
    # a_view reads a's storage; it is no generation of its own
    assert facts.generations["A_a"] == [(0, "a")]
    assert facts.objects["a"].category == "saved activations"
    # b_again belongs to the recompute alternative, which was not selected
    assert "b_again" not in facts.objects
    assert facts.generations["A_b"] == [(1, "b")]
    assert facts.objects["g"].category == "model gradients"
    assert facts.objects["x"].category == "inputs"
    assert facts.retained == {"A_w"}


def test_the_occupant_of_a_slot_is_the_generation_whose_producer_has_started() -> None:
    facts = ProgramFacts.build(_program(), _selection()["selections"])
    clock = Clock.simulated(_selection()["simulation_result"])
    # before t1 starts nothing was produced into A_a, so its only generation stands
    assert facts.occupant("A_a", clock.task_start_ns, 0).object_id == "a"
    assert facts.occupant("A_a", clock.task_start_ns, 7 * S).object_id == "a"
    assert facts.occupant("A_w", clock.task_start_ns, 5 * S).object_id == "w"


def test_the_spill_walk_follows_the_simulator_and_names_what_it_holds() -> None:
    result = attribute(_selection(), _program())
    spill = result.spill
    # the retained weight is there throughout; the input until its fetch lands
    assert spill.at(0) == {"weights": 100 * MIB, "inputs": 8 * MIB}
    assert spill.at(int(1.5 * S)) == {"weights": 100 * MIB}
    # a is spilled from the moment its evict is issued until its fetch completes
    assert spill.at(int(3.5 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB,
    }
    assert spill.at(int(5.5 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB,
    }
    assert spill.at(int(6.5 * S)) == {"weights": 100 * MIB}
    peak_bytes, peak_ns = spill.peak()
    assert peak_bytes == 164 * MIB == spill.reported_peak_bytes
    assert peak_ns == 3 * S
    assert spill.capacity_bytes == 400 * MIB
    assert spill.peaks_by()["saved activations"] == (64 * MIB, 3 * S)
    # the lanes: two fetches at the start, the evict, the fetch back
    assert [(span.direction, span.category) for span in result.transfers] == [
        ("fetch", "weights"),
        ("fetch", "inputs"),
        ("evict", "saved activations"),
        ("fetch", "saved activations"),
    ]
    assert [span.phase for span in result.tasks] == [
        "forward",
        "forward",
        "backward",
        "optimizer",
    ]


def test_the_execution_walk_is_the_layout_lease_by_lease() -> None:
    result = attribute(_selection(), _program())
    execution = result.execution
    assert execution is not None
    assert execution.at(int(2.5 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB + 32 * MIB,
    }
    # from t3 on, b's slot is gb's: a handoff the layout shows as no lease of gb's own
    assert execution.at(int(7 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB,
        WORKSPACE: 40 * MIB,
        "model gradients": 100 * MIB + 32 * MIB,
    }
    assert execution.peak() == (336 * MIB, 6 * S)
    assert execution.capacity_bytes == 300 * MIB
    text = table(execution, "category", [7 * S])
    assert "execution pool: peak 0.33 GiB at 6.00 s" in text
    assert "planner reported 0.23 GiB" in text
    assert "task workspace" in text and "7.00 s" in text


def test_a_traced_step_puts_the_walk_on_the_device_clock() -> None:
    result = attribute(_selection(), _program(), diagnostics=_diagnostics())
    assert result.view == "traced"
    assert result.spill.at(int(2.2 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB,
    }
    assert result.spill.at(int(3.95 * S)) == {"weights": 100 * MIB}
    assert result.tasks[0].name == "microbatch_0.stage_0.forward"
    assert result.tasks[0].start_ns == int(0.9 * S)


def test_a_transfer_the_trace_could_not_time_is_left_off_the_lanes() -> None:
    """A lane record without start and finish is not drawn; the walk that
    needs the input's fetch time takes its trigger task's end instead, and
    the page counts what it left off."""

    from tools.diagnostics.occupancy import page_data

    diagnostics = _diagnostics()
    diagnostics["transfers"]["fetch"]["fetch_1"]["lane"] = {
        "lane_issued_at_seconds": None,
        "lane_started_at_seconds": None,
        "lane_finished_at_seconds": None,
    }
    result = attribute(_selection(), _program(), diagnostics=diagnostics)
    assert result.untimed_transfers == 1
    assert [span.alias_group_id for span in result.transfers] == ["A_w", "A_a", "A_a"]
    # the input's spill copy now closes at t1's end on the device, 1.4 s
    assert result.spill.at(int(1.2 * S)) == {"weights": 100 * MIB, "inputs": 8 * MIB}
    assert result.spill.at(int(1.6 * S)) == {"weights": 100 * MIB}
    assert page_data(result)["untimed_transfers"] == 1


def test_a_traced_step_places_each_lease_at_its_events_device_times() -> None:
    """a's output lease opens at t1's start and closes when its evict
    finishes; its fetch destination opens when the fetch starts and closes
    at t3's end, which releases it; g's lease ends at 9 s, an instant no
    action names, so it is interpolated -- onto t4's end, a task boundary;
    the weight's lease runs to the step's end; b's slot is gb's from t3."""

    result = attribute(_selection(), _program(), diagnostics=_diagnostics())
    execution = result.execution
    assert execution is not None
    by_object = sorted(
        (item.object_id or WORKSPACE, item.start_ns / S, item.end_ns / S)
        for item in execution.intervals
    )
    assert by_object == [
        ("a", 0.9, 2.6),
        ("a", 3.0, 4.5),
        ("b", 1.5, 4.0),
        ("g", 4.0, 5.5),
        ("gb", 4.0, 5.5),
        (WORKSPACE, 4.0, 4.5),
        ("w", 0.0, 5.5),
    ]
    assert result.interpolated_leases == 1
    assert execution.at(int(2.2 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB + 32 * MIB,
    }
    assert execution.at(int(4.2 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB,
        WORKSPACE: 40 * MIB,
        "model gradients": 100 * MIB + 32 * MIB,
    }


def test_the_unconstrained_step_is_the_floor_with_everything_resident() -> None:
    """Every alternative at its cheapest (save), the tasks back to back at
    their profiles, the weight resident throughout, the input until t1 is
    done, a and b until t3 is done, the gradients until the optimizer is,
    t3's profiled workspace while it runs; nothing spilled, nothing moved."""

    assert cheapest_selections(_program()) == [
        {"group_id": "stage_3", "option_id": "save"}
    ]
    result = all_save(_program())
    assert result.view == "all_save" and result.clock.makespan_ns == 5 * S
    own = unconstrained(_program(), _selection()["selections"])
    assert own.view == "unconstrained" and own.clock.makespan_ns == 5 * S
    regenerating = unconstrained(_program(), _selection("recompute")["selections"])
    assert regenerating.clock.makespan_ns == int(5.5 * S)
    assert [
        (span.task_id, span.start_ns / S, span.end_ns / S) for span in result.tasks
    ] == [
        ("t1", 0.0, 1.0),
        ("t2", 1.0, 2.0),
        ("t3", 2.0, 4.0),
        ("t4", 4.0, 5.0),
    ]
    execution = result.execution
    assert execution is not None
    assert execution.at(int(0.5 * S)) == {
        "weights": 100 * MIB,
        "inputs": 8 * MIB,
        "saved activations": 64 * MIB,
    }
    assert execution.at(int(1.5 * S)) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB + 32 * MIB,
    }
    assert execution.at(3 * S) == {
        "weights": 100 * MIB,
        "saved activations": 64 * MIB + 32 * MIB,
        WORKSPACE: 40 * MIB,
        "model gradients": 100 * MIB + 32 * MIB,
    }
    assert execution.at(int(4.5 * S)) == {
        "weights": 100 * MIB,
        "model gradients": 100 * MIB + 32 * MIB,
    }
    assert execution.peak() == (368 * MIB, 2 * S)
    assert result.spill.intervals == () and result.transfers == ()
    summary = summarize(result, tokens_per_step=500)
    assert summary["seconds"] == 5.0 and summary["tokens_per_second"] == 100.0
    assert (
        summary["idle_percent"] == 0.0 and summary["fetch_utilization_percent"] == 0.0
    )
    assert summary["fetch_gbps"] is None and summary["spill_peak_gib"] == 0.0


def test_a_result_without_evidence_is_refused() -> None:
    selection = _selection()
    del selection["simulation_result"]
    try:
        attribute(selection, _program())
    except ValueError as error:
        assert "no simulation result" in str(error)
    else:
        raise AssertionError("a bare plan record has nothing to walk")


def test_pages_are_written_per_view_with_the_data_inline(tmp_path: Path) -> None:
    views = [attribute(_selection(), _program())]
    pages = write_pages(views, tmp_path / "pages", title="tiny", plan="a toy step")
    names = sorted(page.name for page in pages)
    assert names == ["index.html", "simulated.html"]
    page = (tmp_path / "pages" / "simulated.html").read_text()
    payload = json.loads(
        page.split('<script id="data" type="application/json">')[1].split("</script>")[
            0
        ]
    )
    assert payload["view"] == "simulated"
    assert payload["plan"] == "a toy step"
    assert [pool["name"] for pool in payload["pools"]] == ["execution", "spill"]
    assert "saved activations" in payload["pools"][0]["rows"]
    assert len(payload["compute"]) == 4
    assert len(payload["fetch"]) == 3 and len(payload["evict"]) == 1
    assert payload["summary"]["seconds"] == 10.0
    index = (tmp_path / "pages" / "index.html").read_text()
    assert "simulated.html" in index and "spill pool: peak" in index
    assert "a toy step" in index


def test_the_program_is_found_beside_the_selection_in_the_store(tmp_path: Path) -> None:
    digest = "ab" + "0" * 62
    selection_path = (
        tmp_path / "v1" / "planning" / "results" / "cd" / "cdef" / "selection.json"
    )
    assert program_path_for(selection_path, {"program_digest": digest}) == (
        tmp_path / "v1" / "planning" / "programs" / "ab" / digest / "program.json"
    )


def test_a_run_gets_pages_for_every_plan_and_both_clocks_for_a_budget_that_ran(
    tmp_path: Path,
) -> None:
    """The store layout the quickstart leaves behind, in miniature: one
    program, one selection the search made at 1 GiB, and the traced step of
    the budget that ran it."""

    from tools.diagnostics.occupancy import write_run_timelines

    run = tmp_path / "run"
    selection = _selection()
    digest = selection["program_digest"]
    planning = run / "plan_store" / "v1" / "planning"
    (planning / "programs" / digest[:2] / digest).mkdir(parents=True)
    (planning / "programs" / digest[:2] / digest / "program.json").write_text(
        json.dumps(_program())
    )
    key = "cd" + "1" * 62
    (planning / "results" / key[:2] / key).mkdir(parents=True)
    selection["admission_certificate"]["layout"]["pool_capacity_bytes"] = (1 << 30) - (
        128 << 20
    )
    (planning / "results" / key[:2] / key / "selection.json").write_text(
        json.dumps(selection)
    )
    # a resolution the search kept beside its answer: the recompute one
    kept = _selection("recompute")
    kept["resolution"] = {
        "label": "recompute_1",
        "selection_id": "stage_3=recompute",
        "candidate_id": "x",
        "recompute_share": "1",
        "selected": False,
    }
    kept["admission_certificate"]["layout"]["pool_capacity_bytes"] = (1 << 30) - (
        128 << 20
    )
    (planning / "results" / key[:2] / key / "resolutions" / "recompute_1").mkdir(
        parents=True
    )
    (
        planning
        / "results"
        / key[:2]
        / key
        / "resolutions"
        / "recompute_1"
        / "selection.json"
    ).write_text(json.dumps(kept))
    bare = "ef" + "2" * 62
    (planning / "results" / bare[:2] / bare).mkdir(parents=True)
    (planning / "results" / bare[:2] / bare / "selection.json").write_text(
        json.dumps({"program_digest": digest, "schema": "x"})
    )
    (run / "search.json").write_text(
        json.dumps(
            {
                "budgets": [[1 << 30, 4 << 30]],
                "points": [
                    {
                        "sequences_per_microbatch": 4,
                        "accumulation_count": 2,
                        "ordering_label": "1x2rp",
                        "makespan_seconds": 10.0,
                        "status": "succeeded",
                    }
                ],
            }
        )
    )
    (run / "steps").mkdir()
    (run / "steps" / "1gib.json").write_text(json.dumps(_diagnostics()))
    (run / "request.json").write_text(
        json.dumps(
            {"request": {"model": "toy", "sequence_length": 8, "sequences_per_step": 4}}
        )
    )

    heard: list[str] = []
    index = write_run_timelines(run, progress=heard.append)
    assert index == run / "timelines" / "index.html"
    assert heard[0].startswith(
        "timelines: writing pages for 1 plans and 1 kept resolutions"
    )
    assert "timelines: 1gib written" in heard

    def payload(page: Path) -> dict:
        text = page.read_text()
        return json.loads(
            text.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )

    # budget, then geometry, then the recompute share: the choice (all save,
    # recompute 0) with its traced step, the kept resolution beside it
    plan = run / "timelines" / "1gib" / "4x2_1x2rp"
    assert sorted(p.name for p in plan.iterdir()) == [
        "index.html",
        "recompute_0",
        "recompute_1",
    ]
    assert sorted(p.name for p in (plan / "recompute_0").iterdir()) == [
        "index.html",
        "simulated.html",
        "traced.html",
        "unconstrained.html",
    ]
    assert sorted(p.name for p in (plan / "recompute_1").iterdir()) == [
        "index.html",
        "simulated.html",
        "unconstrained.html",
    ]
    floors = run / "timelines" / "all_save" / "4x2_1x2rp"
    assert sorted(p.name for p in floors.iterdir()) == ["all_save.html", "index.html"]
    # the bare record, without evidence, was passed over
    assert sorted(p.name for p in (run / "timelines").iterdir() if p.is_dir()) == [
        "1gib",
        "all_save",
    ]
    # the budget's traced page, a copy at the budget's level
    assert (run / "timelines" / "1gib" / "traced.html").read_bytes() == (
        plan / "recompute_0" / "traced.html"
    ).read_bytes()
    # a table of contents at every level
    text = index.read_text()
    assert "1gib/4x2_1x2rp/recompute_0/traced.html" in text and "5.000 s" in text
    assert '"1gib/traced.html"' in text
    assert "all_save/4x2_1x2rp/all_save.html" in text
    budget_index = (run / "timelines" / "1gib" / "index.html").read_text()
    assert "4x2_1x2rp/recompute_0/simulated.html" in budget_index
    assert "ran with 4x2_1x2rp, recompute 0" in budget_index
    assert "4x2_1x2rp/recompute_1/simulated.html" in budget_index
    plan_index = (plan / "index.html").read_text()
    assert (
        "recompute_0/traced.html" in plan_index
        and "recompute_1/unconstrained.html" in plan_index
    )
    answer_index = (plan / "recompute_0" / "index.html").read_text()
    assert (
        "../recompute_1/index.html" in answer_index
        and "Around this plan" in answer_index
    )
    assert (
        "4x2_1x2rp/all_save.html"
        in (run / "timelines" / "all_save" / "index.html").read_text()
    )
    # every page names its plan under the title, each floor by whose alternatives
    traced = (plan / "recompute_0" / "traced.html").read_text()
    assert "<title>Occupancy 4x2_1x2rp at 1gib, recompute 0 · traced</title>" in traced
    assert payload(plan / "recompute_0" / "traced.html")["plan"] == (
        "toy · 4 sequences per microbatch, 2 microbatches, ordering 1x2rp"
        " · 8 tokens by 4 sequences = 32 tokens a step"
        " · execution budget 1gib (pool 0.88 GiB) · spill pool 0.39 GiB"
        " · resolution: 0 of the flexible groups recompute, the search's choice"
    )
    own = payload(plan / "recompute_0" / "unconstrained.html")
    assert own["end_seconds"] == 5.0
    assert "alternatives as this plan fixes them" in own["plan"]
    assert "execution budget" not in own["plan"]
    kept_floor = payload(plan / "recompute_1" / "unconstrained.html")
    assert kept_floor["end_seconds"] == 5.5
    assert "resolution: 1 of the flexible groups recompute" in kept_floor["plan"]
    assert "alternatives as this resolution fixes them" in kept_floor["plan"]
    kept_plan = payload(plan / "recompute_1" / "simulated.html")["plan"]
    assert "execution budget 1gib" in kept_plan and "unconstrained" not in kept_plan
    # the summary: one row per page
    import csv

    with (run / "timelines" / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(SUMMARY_COLUMNS)
    assert sorted((row["kind"], row["view"]) for row in rows) == [
        ("all_save", "all_save"),
        ("chosen", "simulated"),
        ("chosen", "traced"),
        ("chosen", "unconstrained"),
        ("resolution", "simulated"),
        ("resolution", "unconstrained"),
    ]
    traced_row = next(row for row in rows if row["view"] == "traced")
    assert traced_row["page"] == "1gib/4x2_1x2rp/recompute_0/traced.html"
    assert traced_row["resolution"] == "0" and traced_row["selected"] == "True"
    assert (
        traced_row["geometry"] == "4x2_1x2rp" and traced_row["untimed_transfers"] == "0"
    )
    assert float(traced_row["step_seconds"]) > 0
    kept_row = next(
        row
        for row in rows
        if row["kind"] == "resolution" and row["view"] == "simulated"
    )
    assert kept_row["resolution"] == "1" and kept_row["selected"] == "False"


def test_the_summary_reads_the_step_off_the_same_spans() -> None:
    result = attribute(_selection(), _program())
    summary = summarize(result, tokens_per_step=1000)
    # the step spans 0 to 10 s; t1..t4 compute for 1+1+2+1 = 5 s of it
    assert summary["seconds"] == 10.0
    assert summary["tokens_per_second"] == 100.0
    assert summary["idle_percent"] == 50.0
    assert summary["recompute_percent"] == 0.0
    # 3 s of fetch (0-1 twice, 5-6) and 1 s of evict on the lanes
    assert summary["fetch_utilization_percent"] == 30.0
    assert summary["evict_utilization_percent"] == 10.0
    assert round(summary["fetch_gib"] * 1024) == 172
    assert (
        summary["assumed_fetch_gbps"] == 10.0 and summary["assumed_evict_gbps"] == 20.0
    )
    assert summary["assumed_fetch_latency_us"] == 4.0
    # 172 MiB over 3 s of fetch lane time, 64 MiB over 1 s of evict
    assert round(summary["fetch_gbps"], 3) == round(172 * MIB / 3e9, 3)
    assert round(summary["evict_gbps"], 3) == round(64 * MIB / 1e9, 3)
    assert round(summary["evict_gib"] * 1024) == 64
    assert round(summary["spill_peak_gib"] * 1024) == 164
    assert round(summary["execution_peak_gib"] * 1024) == 336
    recomputed = attribute(_selection("recompute"), _program())
    assert "t3r" in recomputed.facts.recompute_tasks
    # regenerating costs t3r's 2.5 s less t3's 2 s: 0.5 s of a 10 s step
    assert recomputed.facts.recompute_overhead_ns == 500_000_000
    assert summarize(recomputed)["recompute_percent"] == 5.0
    assert recomputed.facts.overhead_by_task == {"t3r": 500_000_000}
    spans = {span.task_id: span for span in recomputed.tasks}
    assert spans["t3r"].overhead_ns == 500_000_000 and spans["t1"].overhead_ns == 0


def test_the_traced_page_shows_the_execution_pool_on_the_device_clock() -> None:
    from tools.diagnostics.occupancy import page_data

    traced = attribute(_selection(), _program(), diagnostics=_diagnostics())
    page = page_data(traced, tokens_per_step=100)
    assert [pool["name"] for pool in page["pools"]] == ["execution", "spill"]
    assert traced.execution is not None
    assert page["summary"]["execution_peak_gib"] == traced.execution.peak()[0] / (
        1 << 30
    )
    assert page["pools"][0]["note"].startswith("leases at the device times")
    assert "1 lease instants interpolated" in page["pools"][0]["note"]
    assert page["pools"][1]["note"] == ""
    assert len(page["compute"]) == 4 and page["compute"][0][4] == 0
    simulated = page_data(attribute(_selection(), _program()))
    assert simulated["pools"][0]["note"] == ""
