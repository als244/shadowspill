"""Partial checkpoints carry portable, immutable base dependencies."""

from dataclasses import replace

import pytest

from client_tests.test_checkpoints import manifest
from shadowspill_client import Model, Weights
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef


def base():
    return FrozenBase(
        Model("external/model@1", initialize=Weights.random(seed=17)),
        DefinitionSource("external-model", "1", "a" * 64),
    )


def test_exact_base_identity_and_manifest_roundtrip():
    original = base()
    assert decode(encode(original)) == original
    changed = replace(
        original, model=replace(original.model, initialize=Weights.random(seed=18))
    )
    assert changed.identity != original.identity
    assert (
        replace(original, source=replace(original.source, digest="b" * 64)).identity
        != original.identity
    )
    saved = manifest()
    saved = replace(
        saved,
        info=replace(
            saved.info, frozen_base=original, frozen_state_id=original.identity
        ),
    )
    assert decode(encode(saved)) == saved
    assert saved.info.rank_metadata()["checkpoint"]["frozen_base"] == original.to_dict()
    with pytest.raises(ValueError, match="differs from its exact base"):
        replace(saved.info, frozen_base=changed)


def test_artifact_identity_includes_exact_bytes_and_reference():
    original = base()
    artifact = ArtifactRef("weights", "c" * 64, 42)
    original = replace(
        original,
        model=replace(original.model, initialize=Weights.artifact("weights")),
        artifact=artifact,
    )
    assert decode(encode(original)) == original
    assert (
        replace(original, artifact=replace(artifact, sha256="d" * 64)).identity
        != original.identity
    )
    for changes in ({"artifact": None}, {"artifact": replace(artifact, id="other")}):
        with pytest.raises(ValueError, match="exact weights artifact"):
            replace(original, **changes)


def test_incomplete_or_unpinned_base_is_rejected():
    original = base()
    for changes, message in (
        ({"model": replace(original.model, initialize=None)}, "resolved installed"),
        (
            {"model": replace(original.model, initialize=Weights.pretrained("model"))},
            "pinned",
        ),
        ({"artifact": ArtifactRef("weights", "e" * 64, 1)}, "only artifact"),
        ({"version": 2}, "version"),
        ({"version": True}, "version"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(original, **changes)
