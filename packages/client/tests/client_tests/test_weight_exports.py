"""Inference generations are distinct from resumable optimizer checkpoints."""

from dataclasses import replace

import pytest

from client_tests.test_checkpoints import manifest
from shadowspill_client.protocol.checkpoint_settings import SaveSettings
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    WEIGHTS_MEDIA_TYPE,
    WEIGHTS_RANK_MEDIA_TYPE,
    CheckpointManifest,
    WeightsManifest,
)
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import OperationResult


def exported():
    value = manifest()
    return WeightsManifest(
        replace(value.info, kind="weights"),
        [
            replace(
                row, archive=replace(row.archive, media_type=WEIGHTS_RANK_MEDIA_TYPE)
            )
            for row in value.ranks
        ],
    )


def test_inference_manifest_roundtrip_and_exact_kind_rank_checks():
    value = exported()
    assert decode(encode(value)) == value
    assert value.info.media_type == WEIGHTS_MEDIA_TYPE
    assert value.info.rank_media_type == WEIGHTS_RANK_MEDIA_TYPE
    assert set(value.info.rank_metadata()) == {"weights"}
    with pytest.raises(ValueError, match="kind"):
        CheckpointManifest(value.info, value.ranks)
    with pytest.raises(ValueError, match="kind"):
        replace(value, info=manifest().info)
    with pytest.raises(ValueError, match="kind"):
        replace(value, ranks=manifest().ranks)
    with pytest.raises(ValueError, match="rank"):
        replace(value, ranks=value.ranks[:1])
    with pytest.raises(ValueError, match="exact frozen base"):
        replace(value, info=replace(value.info, frozen_state_id="unresolved"))


def test_ordered_save_request_and_inference_result_have_no_checkpoint_reference():
    identity = RequestIdentity("export", "run", 1, "writer", 1, "server")
    request = OperationRequest(
        identity, "save", save=SaveSettings(kind="weights", metadata={"cursor": 8})
    )
    reference = ArtifactRef("weights", "a" * 64, 100, WEIGHTS_MEDIA_TYPE)
    result = OperationResult(
        identity, "save", "history", "completed", 7, weights=reference
    )
    assert decode(encode(request)) == request
    assert decode(encode(result)) == result and result.checkpoint is None
    for changes in (
        {"operation": "restore"},
        {"operation": "close"},
        {"weights": None},
        {"weights": replace(reference, media_type=CHECKPOINT_MEDIA_TYPE)},
        {"checkpoint": replace(reference, media_type=CHECKPOINT_MEDIA_TYPE)},
    ):
        with pytest.raises(ValueError):
            replace(result, **changes)
    for changes in ({"kind": "unknown"}, {"format": ""}, {"format": "x" * 129}):
        with pytest.raises(ValueError):
            replace(request.save, **changes)
