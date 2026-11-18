"""A program's tasks are named by what they do and by where in the model they are."""

from types import SimpleNamespace

from shadowspill.artifacts import TaskDefinition
from shadowspill.artifacts.task_names import (
    ProfileName,
    describe,
    part,
    profile_labels,
    profile_names,
    task_parts,
)
from shadowspill.ir import (
    MutationSpec,
    ObjectSpec,
    ResourceKind,
    ResourceSpec,
    TaskAlternativeGroup,
    TaskAlternativeOption,
    TaskSpec,
)

DASH, DOTS, DOT = chr(0x2013), chr(0x2026), f" {chr(0xB7)} "


def task(name, profile, phase, inputs=(), mutations=()):
    return TaskSpec(
        name,
        ResourceSpec("rank0", ResourceKind.COMPUTE),
        profile,
        inputs=tuple(inputs),
        mutations=tuple(MutationSpec(item) for item in mutations),
        phase=phase,
    )


def program(*, registered=True, saved_forms=False):
    """Three stages over three layers. Every forward shares a profile; the
    first stage also reads the embedding and has a backward of its own.
    saved_forms adds the second stage's saved forms, whose accumulating
    backward no task of this program runs."""
    objects = [
        ObjectSpec("embedding", "embedding_storage", 0, 8),
        ObjectSpec("activation", "activation_storage", 0, 8),
    ]
    registrations = [{"name": "embedding.weight", "object_id": "embedding"}]
    tasks, groups = [], []
    for stage in range(3):
        objects.append(ObjectSpec(f"weight{stage}", f"layer{stage}_storage", 0, 8))
        registrations.append(
            {"name": f"layers.{stage}.linear.weight", "object_id": f"weight{stage}"}
        )
        reads = [f"weight{stage}", *(["embedding"] if stage == 0 else [])]
        options = []
        for variant in ("save", "recompute"):
            forward, backward = (
                f"stage{stage}.{variant}.{phase}" for phase in ("forward", "backward")
            )
            first = "first-" if stage == 0 else ""
            tasks.append(task(forward, f"{variant}-forward", "forward", reads))
            tasks.append(
                task(
                    backward,
                    f"{first}{variant}-backward",
                    "backward",
                    ["activation"],
                )
            )
            options.append(TaskAlternativeOption(variant, (forward, backward)))
        groups.append(TaskAlternativeGroup(f"stage{stage}", tuple(options)))
    weights = [f"weight{stage}" for stage in range(3)]
    tasks += [
        task("gather", "gather", "weight_communication", mutations=weights),
        task("update", "adam", "optimizer", ["activation"], ["weight1"]),
        task("begin", "boundary", "control"),
    ]
    profiles = dict.fromkeys(item.profile_id for item in tasks)
    bindings = {"registrations": registrations} if registered else {}
    if saved_forms:
        profiles["save-backward-adding"] = None
        bindings["training_templates"] = [
            [
                [
                    {
                        "variant": "save",
                        "accumulates": adds,
                        "forward": {
                            "profile_key": "save-forward",
                            "inputs": [{"object_id": "weight1"}],
                        },
                        "backward": {
                            "profile_key": "save-backward" + "-adding" * adds,
                            "inputs": [{"object_id": "activation"}],
                        },
                    }
                    for adds in (False, True)
                ]
            ]
        ]
    named = {
        "gather": TaskDefinition("gather", "v1", 0, "all_gather"),
        "adam": TaskDefinition("adam", "v1", 0, "optimizer", ("optimizer",)),
    }
    return SimpleNamespace(
        objects=tuple(objects),
        tasks=tuple(tasks),
        task_alternative_groups=tuple(groups),
        task_definitions=tuple(
            named.get(key, TaskDefinition(key, "v1", 0, "", ("model.layers.0",)))
            if key != "boundary"
            else TaskDefinition(key, "v1", 0)
            for key in profiles
        ),
        bindings=bindings,
    )


def test_a_name_places_a_value_in_a_module_or_a_numbered_member():
    assert part("layers.3.attention.weight") == ("layers", 3)
    assert part("decoder.blocks.2.mlp.weight") == ("decoder.blocks", 2)
    assert part("embedding.weight") == ("embedding", None)
    assert part("scale") == ("scale", None)


def test_parts_read_in_a_few_words():
    assert describe([("embedding", None), ("layers", 0)]) == "embedding, layer 0"
    assert describe([("layers", 2), ("layers", 1), ("h", 4)]) == f"layers 1{DASH}2, h 4"
    # Past the limit a list keeps its start and its end.
    scattered = [("layers", number) for number in (3, 7, 11, 15, 43)]
    assert describe(scattered) == f"layers 3, 7, 11, {DOTS} 43"
    assert describe(scattered, limit=5) == "layers 3, 7, 11, 15, 43"
    assert describe([]) == ""


def test_a_task_covers_the_registered_storage_it_touches():
    covered = task_parts(program())
    assert covered["stage0.save.forward"] == (("layers", 0), ("embedding", None))
    assert covered["stage2.recompute.forward"] == (("layers", 2),)
    assert covered["stage1.save.backward"] == ()
    assert covered["update"] == (("layers", 1),)
    assert set(task_parts(program(registered=False)).values()) == {()}


def test_a_profile_is_named_by_what_it_does_and_where_its_stages_are():
    names = profile_names(program())
    # The forward is compiled once for every stage.
    assert names["save-forward"] == ProfileName(
        "forward",
        "save",
        (("layers", 0), ("embedding", None), ("layers", 1), ("layers", 2)),
    )
    assert names["save-forward"].label == f"save forward{DOT}layers 0{DASH}2, embedding"
    # A backward reads no weight itself; it is where its stage is.
    assert names["first-recompute-backward"].label == (
        f"recompute backward{DOT}layer 0, embedding"
    )
    assert names["save-backward"].label == f"save backward{DOT}layers 1{DASH}2"
    # Outside every alternative: a task by its definition's name, and one
    # whose definition has none by its phase.
    assert names["gather"] == ProfileName(
        "all_gather", "", (("layers", 0), ("layers", 1), ("layers", 2))
    )
    assert names["adam"].label == f"optimizer{DOT}layer 1"
    assert names["boundary"].label == "control"
    assert profile_labels(program()) == {key: name.label for key, name in names.items()}


def test_a_saved_form_names_a_profile_no_task_runs():
    names = profile_names(program(saved_forms=True))
    assert names["save-backward-adding"] == ProfileName(
        "backward (accumulating)", "save", (("layers", 1),)
    )
    assert names["save-backward-adding"].label == (
        f"save backward (accumulating){DOT}layer 1"
    )
    # A profile keeps one name whether a task or only a saved form uses it.
    plain = profile_names(program())
    assert all(names[key].label == name.label for key, name in plain.items())


def test_without_registrations_a_profile_keeps_what_it_does():
    labels = profile_labels(program(registered=False))
    assert labels["save-backward"] == "save backward"
    assert labels["gather"] == "all_gather"
