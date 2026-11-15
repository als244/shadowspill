"""Authenticated resumed binary reads from the same immutable artifact store."""

import hashlib
import http.client
import io
from dataclasses import replace

import pytest

from engine_tests.server.test_http import OTHER_TOKEN, TOKEN, endpoint
from shadowspill_client.protocol.codec import ProtocolError, decode
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.transport.http import RemoteError

PAYLOAD = bytes(range(256)) * 1025
REFERENCE = ArtifactRef("download", hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD))


def test_verified_resume_reads_only_suffix_and_keeps_complete_file_integrity(tmp_path):
    offset = 100003
    with endpoint(tmp_path) as ctx:
        ctx.artifacts.put(REFERENCE, lambda _: io.BytesIO(PAYLOAD))
        destination = io.BytesIO(PAYLOAD[:offset])
        progress = []
        ctx.client.download(
            "/v1/artifacts/download",
            REFERENCE,
            destination,
            offset=offset,
            prefix_sha256=hashlib.sha256(PAYLOAD[:offset]).hexdigest(),
            progress=lambda size, digest: progress.append((size, digest)),
        )
        assert destination.getvalue() == PAYLOAD
        assert progress[-1] == (len(PAYLOAD), REFERENCE.sha256)
        assert all(size > offset for size, _ in progress)
        assert all(
            hashlib.sha256(PAYLOAD[:size]).hexdigest() == digest
            for size, digest in progress
        )
        assert ctx.entered == ["owner"]
        # A fully verified retained file needs no new exchange.
        ctx.client.download(
            "/v1/artifacts/download",
            REFERENCE,
            destination,
            offset=len(PAYLOAD),
            prefix_sha256=REFERENCE.sha256,
        )
        assert ctx.entered == ["owner"] and not ctx.artifacts._readers


@pytest.mark.parametrize("fault", ["short", "long", "corrupt", "complete_corrupt"])
def test_bad_prefix_is_rejected_before_network_or_destination_mutation(tmp_path, fault):
    offset = len(PAYLOAD) if fault == "complete_corrupt" else 100003
    prefix = PAYLOAD[:offset]
    digest = hashlib.sha256(prefix).hexdigest()
    if fault == "short":
        prefix = prefix[:-1]
    elif fault == "long":
        prefix += b"x"
    else:
        prefix = b"x" + prefix[1:]
        if fault == "complete_corrupt":
            digest = hashlib.sha256(prefix).hexdigest()
    with endpoint(tmp_path) as ctx:
        destination = io.BytesIO(prefix)
        with pytest.raises(ProtocolError):
            ctx.client.download(
                "/v1/artifacts/download",
                REFERENCE,
                destination,
                offset=offset,
                prefix_sha256=digest,
            )
        assert destination.getvalue() == prefix and not ctx.entered


def request(ctx, headers, token=TOKEN):
    connection = http.client.HTTPConnection(*ctx.service.server_address, timeout=5)
    try:
        connection.putrequest("GET", "/v1/artifacts/download")
        for key, value in [
            ("Authorization", "Bearer " + token),
            ("Content-Length", "0"),
            *headers,
        ]:
            connection.putheader(key, value)
        connection.endheaders()
        with connection.getresponse() as response:
            return response.status, response.headers, response.read()
    finally:
        connection.close()


def test_range_headers_bind_offset_to_the_authenticated_exact_artifact(tmp_path):
    with endpoint(tmp_path) as ctx:
        ctx.artifacts.put(REFERENCE, lambda _: io.BytesIO(PAYLOAD))
        etag = '"' + REFERENCE.sha256 + '"'
        headers = [("Range", "bytes=17-"), ("If-Match", etag)]
        status, reply, body = request(ctx, headers)
        assert status == 206 and body == PAYLOAD[17:]
        assert reply["ETag"] == etag and reply["Accept-Ranges"] == "bytes"
        assert reply["Content-Range"] == f"bytes 17-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
        assert int(reply["Content-Length"]) == len(PAYLOAD) - 17
        assert request(ctx, headers, OTHER_TOKEN)[0] == 404
        for invalid in (
            [("Range", "bytes=0-1")],
            [("Range", "bytes=-1-")],
            [("Range", "bytes=01-")],
            [("Range", "bytes=1-,2-")],
            [("Range", "bytes=0-"), ("Range", "bytes=0-")],
            [("Range", "bytes=" + "9" * 20 + "-")],
        ):
            assert request(ctx, [*invalid, ("If-Match", etag)])[0] == 400
        for invalid in (
            [("Range", "bytes=1-")],
            [("Range", "bytes=1-"), ("If-Match", '"wrong"')],
            [*headers, ("If-Match", etag)],
        ):
            status, _, body = request(ctx, invalid)
            assert status == 412 and decode(body).code == "artifact_changed"
        status, reply, body = request(
            ctx, [("Range", f"bytes={len(PAYLOAD)}-"), ("If-Match", etag)]
        )
        assert status == 416 and reply["Content-Range"] == f"bytes */{len(PAYLOAD)}"
        assert decode(body).code == "invalid_range"


def test_reference_change_does_not_append_to_the_validated_prefix(tmp_path):
    with endpoint(tmp_path) as ctx:
        ctx.artifacts.put(REFERENCE, lambda _: io.BytesIO(PAYLOAD))
        prefix = PAYLOAD[:17]
        destination = io.BytesIO(prefix)
        with pytest.raises(RemoteError) as error:
            ctx.client.download(
                "/v1/artifacts/download",
                replace(REFERENCE, sha256="a" * 64),
                destination,
                offset=len(prefix),
                prefix_sha256=hashlib.sha256(prefix).hexdigest(),
            )
        assert error.value.status == 412 and destination.getvalue() == prefix
