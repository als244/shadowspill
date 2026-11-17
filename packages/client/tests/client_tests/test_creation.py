"""Creation choices roundtrip without frameworks or server-owned Python objects."""

from dataclasses import replace

import pytest

from client_tests.test_training_config import resolved
from shadowspill_client.protocol.codec import decode, digest, encode
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.preparation import PREPARATION_MEDIA_TYPE
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.runs import RunInfo, RunPage


def test_creation_owns_all_choices_separately_from_the_resolved_setup():
    creation = TrainingCreation(
        resolved(),
        ArtifactRef("plans", "a" * 64, 100),
        ArtifactRef("checkpoint", "b" * 64, 200),
        wait_for_resources=True,
    )
    assert decode(encode(creation)) == creation
    assert TrainingCreation.from_dict(creation.to_dict()) == creation
    changed = replace(creation, wait_for_resources=False)
    assert changed.config == creation.config and digest(changed) != digest(creation)
    assert digest(replace(creation, plans=None)) != digest(creation)
    assert digest(replace(creation, checkpoint=None)) != digest(creation)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_waiting_is_an_explicit_boolean(value):
    with pytest.raises(TypeError, match="boolean"):
        TrainingCreation(resolved(), wait_for_resources=value)


def test_creation_does_not_accept_unknown_paths_or_executable_setup():
    for name, value in (("server_path", "/tmp/model"), ("factory", lambda: None)):
        with pytest.raises(ValueError, match="unknown fields"):
            TrainingCreation.from_dict(
                {**TrainingCreation(resolved()).to_dict(), name: value}
            )
    with pytest.raises(TypeError, match="ArtifactRef"):
        TrainingCreation(resolved(), plans="/client/private/training.ssp")


def test_creation_organization_is_owned_validated_and_part_of_the_digest():
    labels = {"experiment": "demo"}
    creation = TrainingCreation(resolved(), name="example", labels=labels)
    labels["experiment"] = "changed"
    assert creation.labels["experiment"] == "demo"
    assert decode(encode(creation)) == creation
    assert digest(replace(creation, name="new")) != digest(creation)
    with pytest.raises(ValueError, match="run name"):
        replace(creation, name="x" * 257)
    with pytest.raises(ValueError, match="label"):
        replace(creation, labels={"k": "x" * 257})


def test_run_page_roundtrip_requires_one_scope_and_a_valid_continuation():
    run = RunInfo(
        "owner",
        "session",
        "a",
        "server",
        "a" * 64,
        ArtifactRef("configuration", "b" * 64, 100),
    )
    page = RunPage([run, replace(run, run_id="b")], "b")
    assert decode(encode(page)) == page
    assert decode(encode(RunPage([]))).runs == ()
    for rows, cursor in (
        ([run, run], None),
        ([replace(run, run_id="b"), run], None),
        ([run, replace(run, owner_id="other", run_id="b")], None),
        ([run], "missing"),
        ([], "a"),
        ([run] * 129, None),
    ):
        with pytest.raises(ValueError):
            RunPage(rows, cursor)


def test_measurement_records_have_no_training_state_or_restore_choices():
    creation = TrainingCreation(resolved(), kind="preparation")
    assert decode(encode(creation)) == creation
    assert digest(creation) != digest(replace(creation, kind="training"))
    reference = ArtifactRef("measurements", "a" * 64, 100, PREPARATION_MEDIA_TYPE)
    info = RunInfo(
        "owner",
        "session",
        "run",
        "server",
        "b" * 64,
        ArtifactRef("configuration", "c" * 64, 100),
        kind="preparation",
        state="prepared",
        stage=None,
        preparation_export=reference,
    )
    assert decode(encode(info)) == info
    for change in (
        {"plans": reference},
        {"checkpoint": reference},
        {"kind": "unknown"},
    ):
        with pytest.raises(ValueError):
            replace(creation, **change)
    for change in (
        {"kind": "training"},
        {"preparation_export": None},
        {"plans": reference, "initial_history_id": "history"},
        {"state": "preparing", "stage": "plan", "preparation_export": None},
        {"preparation_export": replace(reference, media_type="application/zip")},
    ):
        with pytest.raises(ValueError):
            replace(info, **change)


def test_planned_records_require_plans_without_any_training_state():
    creation = TrainingCreation(resolved(), kind="planning")
    assert decode(encode(creation)) == creation
    plans = ArtifactRef("plans", "a" * 64, 100)
    info = RunInfo(
        "owner",
        "session",
        "run",
        "server",
        "b" * 64,
        ArtifactRef("configuration", "c" * 64, 100),
        kind="planning",
        state="planned",
        stage=None,
        plans=plans,
    )
    assert decode(encode(info)) == info
    for change in ({"plans": plans}, {"checkpoint": plans}):
        with pytest.raises(ValueError):
            replace(creation, **change)
    for change in (
        {"plans": None},
        {"initial_history_id": "history"},
        {"kind": "training"},
        {"kind": "preparation"},
        {"state": "ready"},
        {"state": "preparing", "stage": "plan"},
        {"state": "preparing", "stage": "initializing", "plans": None},
    ):
        with pytest.raises(ValueError):
            replace(info, **change)
