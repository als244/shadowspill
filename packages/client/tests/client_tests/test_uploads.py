"""Upload progress describes verified offsets without exposing executable state."""

import hashlib
from dataclasses import replace

import pytest

from shadowspill_client.protocol.codec import RecordCodec, decode, encode
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.uploads import ArtifactUpload, UploadStatus


def test_upload_request_and_progress_roundtrip_and_own_their_reference():
    body = b"artifact payload"
    reference = ArtifactRef("upload", hashlib.sha256(body).hexdigest(), len(body))
    raw = reference.to_dict()
    request = ArtifactUpload(raw)
    raw["size_bytes"] = 1
    assert request.reference == reference and decode(encode(request)) == request
    pending = UploadStatus(
        reference, 3, hashlib.sha256(body[:3]).hexdigest(), "uploading", 12345.0
    )
    assert decode(encode(pending)) == pending
    received = replace(pending, offset=len(body), prefix_sha256=reference.sha256)
    complete = replace(received, state="complete", expires_at=None)
    assert decode(encode(complete)) == complete
    # The dedicated store codec can persist exactly the same portable record.
    codec = RecordCodec({"upload.status": UploadStatus})
    assert (
        codec.decode(codec.encode(pending, max_bytes=4096), max_bytes=4096) == pending
    )


@pytest.mark.parametrize(
    "change",
    [
        {"offset": -1},
        {"offset": True},
        {"offset": 11},
        {"prefix_sha256": "invalid"},
        {"prefix_sha256": "a" * 65},
        {"prefix_sha256": "b" * 64},
        {"offset": 10, "prefix_sha256": "b" * 64},
        {"state": "complete"},
        {"state": "missing"},
        {"expires_at": None},
        {"expires_at": 0},
        {"expires_at": -1},
        {"expires_at": float("inf")},
        {"expires_at": True},
    ],
)
def test_progress_rejects_unverified_out_of_bounds_or_ambiguous_lifetimes(change):
    pending = UploadStatus(
        ArtifactRef("upload", "a" * 64, 10),
        0,
        hashlib.sha256().hexdigest(),
        "uploading",
        100.0,
    )
    with pytest.raises((TypeError, ValueError)):
        replace(pending, **change)


def test_empty_upload_requires_its_actual_empty_digest_and_publication_status():
    empty = hashlib.sha256().hexdigest()
    reference = ArtifactRef("empty", empty, 0)
    pending = UploadStatus(reference, 0, empty, "uploading", 100.0)
    complete = replace(pending, state="complete", expires_at=None)
    assert decode(encode(complete)) == complete
    with pytest.raises(ValueError, match="checksum"):
        replace(pending, prefix_sha256="a" * 64)
    with pytest.raises(ValueError, match="no upload expiry"):
        replace(complete, expires_at=100.0)
    with pytest.raises(ValueError, match="media type"):
        ArtifactUpload(replace(reference, media_type="x" * 257))


@pytest.mark.parametrize("media", ["text/é", "text/plain\r\nX: header", "text/\x7f"])
def test_upload_media_type_can_be_used_for_a_later_http_download(media):
    with pytest.raises(ValueError, match="media type"):
        ArtifactUpload(ArtifactRef("upload", "a" * 64, 1, media))
