"""Creation choices roundtrip without frameworks or server-owned Python objects."""

from dataclasses import replace

import pytest

from client_tests.test_training_config import resolved
from shadowspill_client.protocol.codec import decode, digest, encode
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.requests import ArtifactRef


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
