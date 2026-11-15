"""Checkpoint handles can describe complete generations without the engine."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    RANK_MEDIA_TYPE,
    CheckpointConversion,
    CheckpointManifest,
    SavedRank,
    SaveInfo,
)
from shadowspill_client.protocol.codec import ProtocolError, decode, encode
from shadowspill_client.protocol.requests import ArtifactRef


def manifest():
    return CheckpointManifest(
        SaveInfo(
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
            SavedRank(
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
    ordinary = updated.to_dict()
    ordinary.pop("conversion")
    ordinary.pop("frozen_base")
    ordinary.pop("kind")
    ordinary.pop("format")
    assert updated.rank_metadata()["checkpoint"] == ordinary


def test_conversion_provenance_roundtrips_and_is_included_in_rank_boundary():
    value = manifest()
    conversion = CheckpointConversion(
        ArtifactRef("source", "e" * 64, 100, CHECKPOINT_MEDIA_TYPE),
        value.info.plans,
        "f" * 64,
        randomness={
            "version": 1,
            "root_seed": 7,
            "restore_id": "conversion",
            "source_streams": [],
            "target_streams": [],
        },
    )
    converted = replace(value, info=replace(value.info, conversion=conversion))
    assert decode(encode(converted)) == converted
    metadata = converted.info.to_dict()
    metadata.pop("frozen_base")
    metadata.pop("kind")
    metadata.pop("format")
    assert converted.info.rank_metadata()["checkpoint"] == metadata
    assert conversion.randomness["source_streams"] == ()
    for changes in ({"version": 2}, {"version": True}, {"logical_state_sha256": "bad"}):
        with pytest.raises(ValueError):
            replace(conversion, **changes)


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


def test_save_settings_are_owned_and_have_an_intentional_result_reference():
    from shadowspill_client.protocol.checkpoint_settings import SaveSettings
    from shadowspill_client.protocol.checkpoints import CHECKPOINT_MEDIA_TYPE
    from shadowspill_client.protocol.requests import OperationRequest, RequestIdentity
    from shadowspill_client.protocol.results import OperationResult

    metadata = {"cursor": [1, 2]}
    settings = SaveSettings(name="latest", metadata=metadata, weights="compute")
    metadata["cursor"].append(3)
    identity = RequestIdentity("save", "run", 1, "writer", 1, "server")
    request = OperationRequest(identity, "save", save=settings)
    assert decode(encode(request)) == request and settings.metadata["cursor"] == (1, 2)
    ref = ArtifactRef("checkpoint", "f" * 64, 100, CHECKPOINT_MEDIA_TYPE)
    result = OperationResult(
        identity, "save", "history", "completed", 9, checkpoint=ref
    )
    assert decode(encode(result)) == result
    with pytest.raises(ValueError, match="only save"):
        replace(request, operation="close")
    with pytest.raises(ValueError, match="checkpoint reference"):
        replace(result, checkpoint=None)
    with pytest.raises(ValueError, match="compute outputs"):
        replace(result, outputs={"data": ref})
    with pytest.raises(ValueError, match="manifest"):
        replace(result, checkpoint=replace(ref, media_type="application/octet-stream"))


def test_restore_records_pin_reference_separately_from_submitted_name():
    from shadowspill_client.protocol.checkpoints import CHECKPOINT_MEDIA_TYPE
    from shadowspill_client.protocol.requests import OperationRequest, RequestIdentity
    from shadowspill_client.protocol.results import Acceptance, OperationResult

    identity = RequestIdentity("load", "run", 2, "writer", 1, "server")
    command = OperationRequest(identity, "restore", restore="latest")
    reference = ArtifactRef("generation", "a" * 64, 1024, CHECKPOINT_MEDIA_TYPE)
    acceptance = Acceptance(identity, 3, checkpoint=reference)
    metadata = {"cursor": [5, 7]}
    result = OperationResult(
        identity,
        "restore",
        "next-history",
        "completed",
        3,
        checkpoint=reference,
        metadata=metadata,
    )
    metadata["cursor"].append(9)
    assert result.metadata["cursor"] == (5, 7)
    for value in (command, acceptance, result):
        assert decode(encode(value)) == value
    assert command.restore == "latest"
    with pytest.raises(ValueError, match="only restore"):
        replace(command, operation="close")
    with pytest.raises(ValueError, match="checkpoint reference"):
        replace(result, checkpoint=None)
    with pytest.raises(ValueError, match="only restore"):
        replace(result, operation="save")
