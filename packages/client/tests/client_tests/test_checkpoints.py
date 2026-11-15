"""Checkpoint handles can describe complete generations without the engine."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.checkpoints import (
    RANK_MEDIA_TYPE,
    CheckpointInfo,
    CheckpointManifest,
    CheckpointRank,
)
from shadowspill_client.protocol.codec import ProtocolError, decode, encode
from shadowspill_client.protocol.requests import ArtifactRef


def manifest():
    return CheckpointManifest(
        CheckpointInfo(
            "session",
            "run",
            "history",
            "request",
            7,
            2,
            ArtifactRef("configuration", "a" * 64, 100),
            ArtifactRef("plans", "b" * 64, 1000),
            "little",
            metadata={"cursor": [3, 4]},
        ),
        [
            CheckpointRank(
                rank,
                ArtifactRef(f"rank{rank}", "c" * 64, 10, RANK_MEDIA_TYPE),
                "d" * 64,
            )
            for rank in range(2)
        ],
    )


def test_manifest_roundtrip_and_owned_metadata():
    value = manifest()
    assert decode(encode(value)) == value
    metadata = {"cursor": [5, 6]}
    updated = replace(value.info, metadata=metadata)
    metadata["cursor"].append(7)
    assert updated.metadata["cursor"] == (5, 6)
    assert updated.rank_metadata()["checkpoint"] == updated.to_dict()


@pytest.mark.parametrize("ranks", [(), (0,), (1, 0), (0, 0), (0, 1, 2)])
def test_complete_rank_set_is_required(ranks):
    value = manifest()
    with pytest.raises(ValueError, match="rank"):
        replace(value, ranks=[replace(value.ranks[0], rank=rank) for rank in ranks])


@pytest.mark.parametrize(
    "changes",
    [
        {"world_size": 0},
        {"completed_updates": -1},
        {"completed_updates": True},
        {"weights": "both"},
        {"byteorder": "native"},
        {"request_id": "../request"},
        {"metadata": {"cursor": object()}},
    ],
)
def test_invalid_info_is_not_portable(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(manifest().info, **changes)


def test_duplicate_archives_and_wrong_archive_types_or_state_identities_are_rejected():
    value = manifest()
    with pytest.raises(ValueError, match="distinct"):
        replace(value, ranks=[value.ranks[0], replace(value.ranks[0], rank=1)])
    with pytest.raises(ValueError, match="rank archive"):
        replace(
            value.ranks[0], archive=replace(value.ranks[0].archive, media_type="other")
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(value.ranks[0], state_sha256="not-a-digest")


def test_metadata_keeps_common_wire_bounds():
    value = manifest()
    with pytest.raises(ProtocolError, match="exceeds"):
        encode(
            replace(
                value, info=replace(value.info, metadata={"large": "x" * (1 << 20)})
            )
        )
