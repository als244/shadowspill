"""A program's alternatives are described once per form, with what was chosen."""

from shadowspill.diagnostics.occupancy.alternatives import alternatives_data
from shadowspill.planning.training.order import coordinates, task_id


def task(name, phase, profile, inputs=(), outputs=(), mutations=()):
    return {
        "task_id": name,
        "phase": phase,
        "profile_id": profile,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "mutations": [{"object_id": value} for value in mutations],
    }


def program(microbatches=2):
    objects, tasks, groups = [], [], []
    for index, (alias, offset, size) in enumerate(
        [("a", 0, 100), ("a", 50, 100), ("b", 0, 40), ("c", 0, 8), ("d", 0, 30)]
    ):
        objects.append(
            {
                "object_id": f"o{index}",
                "alias_group_id": alias,
                "offset_bytes": offset,
                "size_bytes": size,
            }
        )
    for batch in range(microbatches):
        options = []
        for variant, kept in (("save", ["b", "d"]), ("recompute", [])):
            forward, backward = (
                f"m{batch}.{variant}.{phase}" for phase in ("forward", "backward")
            )
            tasks.append(task(forward, "forward", f"{variant}-f", ["o0", "o1"], ["o2"]))
            tasks.append(
                task(backward, "backward", f"{variant}-b", ["o2"], ["o3"], ["o4"])
            )
            options.append(
                {
                    "option_id": variant,
                    "active_task_ids": [forward, backward],
                    "retained_alias_group_ids": kept,
                }
            )
        groups.append({"group_id": f"g{batch}", "options": options})
    tasks.append(task("update", "optimizer", "adam", ["o4"], (), ["o0"]))
    return {
        "tasks": tasks,
        "objects": objects,
        "alias_groups": [
            {"alias_group_id": alias, "size_bytes": size}
            for alias, size in (("a", 150), ("b", 40), ("c", 8), ("d", 30))
        ],
        "profiles": [
            {"profile_id": "save-f", "runtime_ns": 10, "workspace_bytes": 5},
            {"profile_id": "save-b", "runtime_ns": 20, "workspace_bytes": 7},
            {"profile_id": "recompute-f", "runtime_ns": 11, "workspace_bytes": None},
            {"profile_id": "recompute-b", "runtime_ns": None, "workspace_bytes": 9},
            {"profile_id": "adam", "runtime_ns": 3, "workspace_bytes": 0},
        ],
        "task_alternative_groups": groups,
    }


def chosen(*options):
    return [
        {"group_id": f"g{index}", "option_id": option}
        for index, option in enumerate(options)
    ]


def test_identical_groups_are_one_form_with_a_count_of_each_choice():
    data = alternatives_data(program(3), chosen("save", "recompute", "recompute"))
    assert data["options"] == ["save", "recompute"]
    (form,) = data["forms"]
    assert form["instances"] == 3 and not form["sizes_vary"]
    assert [(o["option"], o["chosen"]) for o in form["options"]] == [
        ("save", 1),
        ("recompute", 2),
    ]


def test_sizes_are_storage_reached_and_never_summed_across_views():
    save, recompute = alternatives_data(program(), chosen("save", "save"))["forms"][0][
        "options"
    ]
    forward, backward = save["tasks"]
    # Two inputs view one allocation: [0, 100) and [50, 150) reach 150 bytes.
    assert forward["input_bytes"] == 150 and forward["output_bytes"] == 40
    assert forward["mutated_bytes"] == 0 and forward["workspace_bytes"] == 5
    assert forward["total_bytes"] == 150 + 40 + 5
    assert (backward["input_bytes"], backward["output_bytes"]) == (40, 8)
    assert (
        backward["mutated_bytes"] == 30 and backward["total_bytes"] == 40 + 8 + 30 + 7
    )
    assert save["runtime_ns"] == 30
    assert (save["retained_groups"], save["retained_bytes"]) == (2, 70)
    assert (recompute["retained_groups"], recompute["retained_bytes"]) == (0, 0)
    # An unknown workspace or time stays unknown; it is never priced as zero.
    assert recompute["tasks"][0]["total_bytes"] is None
    assert recompute["tasks"][1]["runtime_ns"] is None
    assert recompute["runtime_ns"] is None


def test_measured_time_covers_only_the_occurrences_that_ran():
    measured = {
        "m0.save.forward": [12],
        "m1.save.forward": [16],
        "m2.recompute.forward": [40],
        "m0.recompute.forward": [999],  # not chosen for g0: never ran
        "update": [4, 6],
    }
    data = alternatives_data(
        program(3), chosen("save", "save", "recompute"), measured_ns=measured
    )
    save, recompute = data["forms"][0]["options"]
    assert save["tasks"][0]["measured"] == {
        "count": 2,
        "median_ns": 14,
        "min_ns": 12,
        "max_ns": 16,
    }
    assert save["tasks"][1]["measured"] is None
    assert recompute["tasks"][0]["measured"]["count"] == 1
    (single,) = data["single"]
    assert (single["phase"], single["count"]) == ("optimizer", 1)
    assert single["measured"]["median_ns"] == 5
    # In-place state is mutated, not added: o4 is read and o0 written in place.
    assert (single["input_bytes"], single["mutated_bytes"]) == (30, 100)


def test_different_profiles_are_different_forms_and_labels_come_from_profiles():
    described = program()
    described["tasks"][4]["profile_id"] = "save-f-first"
    described["profiles"].append(
        {"profile_id": "save-f-first", "runtime_ns": 1, "workspace_bytes": 0}
    )
    labels = {"save-f": "model.blocks.0-1", "save-f-first": "forward"}
    data = alternatives_data(described, chosen("save", "save"), labels=labels)
    first, second = data["forms"]
    assert (first["instances"], second["instances"]) == (1, 1)
    assert first["label"] == "model.blocks.0-1"
    # A label that only repeats the phase leaves the form to be named by place.
    assert second["label"] == ""


def test_the_map_places_every_occurrence_by_column_and_row():
    positions = {"g0": (0, 7), "g1": (1, 7), "g2": (1, 9)}
    data = alternatives_data(
        program(3), chosen("save", "recompute"), positions=positions
    )
    assert data["map"] == {
        "columns": [0, 1],
        "rows": [
            {"row": 7, "cells": [["save", 0], ["recompute", 0]]},
            {"row": 9, "cells": [None, [None, 0]]},
        ],
    }
    # Without a place for every group there is no map, and nothing is guessed.
    partial = alternatives_data(program(3), [], positions={"g0": (0, 0)})
    assert partial["map"] is None
    assert alternatives_data(program(3), [])["map"] is None


def test_a_program_without_alternatives_has_only_single_tasks():
    plain = program(0)
    data = alternatives_data(plain, [])
    assert data["forms"] == [] and data["map"] is None and data["options"] == []
    assert [row["phase"] for row in data["single"]] == ["optimizer"]


def test_task_names_give_back_their_microbatch_and_stage():
    assert coordinates(task_id("backward", 3, 47, "recompute")) == (3, 47)
    for name in ("invocation_start", "example_0001.stage_x.save.forward", "a.b.c.d"):
        assert coordinates(name) is None
