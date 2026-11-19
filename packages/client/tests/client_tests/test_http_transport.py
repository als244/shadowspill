"""Real loopback exchanges, borrowed binary chunks and lost-response boundaries."""

import hashlib
import io
import socket
import subprocess
import sys
from contextlib import contextmanager, suppress
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import (
    MAX_METADATA_BYTES,
    ProtocolError,
    decode,
    encode,
)
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.service import (
    CONTROL_MEDIA_TYPE,
    ServiceError,
    ServiceInfo,
)
from shadowspill_client.transport.http import (
    CHUNK_BYTES,
    HTTP_SCRATCH_BYTES,
    HTTPTransport,
    RemoteError,
    TransportError,
)

TOKEN = "private-test-token-" + "x" * 32
INFO = ServiceInfo("server", "owner")
PAYLOAD = bytes(range(256)) * 1025
REFERENCE = ArtifactRef("output", hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD))


@contextmanager
def endpoint(*, patch=None, block=False):
    events, entered, release = [], Event(), Event()
    if not block:
        release.set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def do_PUT(self):
            self.handle_request()

        def handle_request(self):
            self.connection.settimeout(5)
            size = int(self.headers["Content-Length"])
            value = self.rfile.read(size)
            events.append((self.command, self.path, self.headers, value))
            entered.set()
            assert release.wait(5)
            if self.path == "/v1/lost":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            binary = self.path == "/v1/binary"
            body = PAYLOAD if binary else encode(INFO)
            headers = {
                "Content-Type": REFERENCE.media_type if binary else CONTROL_MEDIA_TYPE,
                "Content-Length": str(len(body)),
                "ETag": '"' + REFERENCE.sha256 + '"',
                "Connection": "close",
            }
            status = 200
            if self.path == "/v1/error":
                status, body = (
                    403,
                    encode(ServiceError("not_authorized", "Access denied")),
                )
                headers["Content-Length"] = str(len(body))
            if patch:
                status, headers, body = patch(status, headers, body)
            self.send_response(status)
            for key, header in headers.items():
                self.send_header(key, header)
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    server_thread.start()
    memory = BufferBudget(HTTP_SCRATCH_BYTES, 1)
    client = HTTPTransport(
        f"http://127.0.0.1:{server.server_port}",
        token=TOKEN,
        memory=memory,
        timeout=5,
    )
    try:
        yield SimpleNamespace(
            client=client,
            events=events,
            entered=entered,
            release=release,
            memory=memory,
        )
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        assert not server_thread.is_alive() and memory.used == (0, 0)


def test_authenticated_metadata_and_binary_upload_are_distinct_wire_bodies():
    with endpoint() as ctx:
        reply = ctx.client.request("POST", "/v1/control", INFO)
        assert reply.status == 200 and reply.value == INFO
        closed = []

        def source():
            try:
                yield memoryview(PAYLOAD)
            finally:
                closed.append(True)

        assert ctx.client.upload("/v1/input", REFERENCE, source()).value == INFO
        assert closed == [True]
        assert decode(ctx.events[0][3]) == INFO and ctx.events[1][3] == PAYLOAD
        assert all(
            event[2]["Authorization"] == "Bearer " + TOKEN for event in ctx.events
        )
        assert ctx.events[0][2]["Content-Type"] == CONTROL_MEDIA_TYPE
        assert ctx.events[1][2]["Content-Type"] == REFERENCE.media_type


def test_partial_destination_writes_are_bounded_and_download_digest_is_checked():
    class Destination(io.BytesIO):
        def write(self, value):
            assert len(value) <= CHUNK_BYTES
            return super().write(value[:137])

    with endpoint() as ctx:
        destination = Destination()
        ctx.client.download("/v1/binary", REFERENCE, destination)
        assert destination.getvalue() == PAYLOAD


@pytest.mark.parametrize(
    "fault",
    [None, "full", "offset", "end", "total", "size", "digest", "truncated", "corrupt"],
)
def test_resume_requires_exact_range_headers_and_complete_checksum(fault):
    offset = 100003

    def patch(status, headers, body):
        if fault == "full":
            return status, headers, body
        headers["Content-Range"] = f"bytes {offset}-{len(body) - 1}/{len(body)}"
        body = body[offset:]
        headers["Content-Length"] = str(len(body))
        if fault in ("offset", "end", "total"):
            start = offset + (fault == "offset")
            end = len(PAYLOAD) - 1 - (fault == "end")
            total = len(PAYLOAD) + (fault == "total")
            headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        elif fault == "size":
            headers["Content-Length"] = str(len(body) + 1)
        elif fault == "digest":
            headers["ETag"] = '"' + "a" * 64 + '"'
        elif fault == "truncated":
            body = body[:-1]
        elif fault == "corrupt":
            body = b"x" + body[1:]
        return 206, headers, body

    with endpoint(patch=patch) as ctx:
        destination = io.BytesIO(PAYLOAD[:offset])

        def download():
            ctx.client.download(
                "/v1/binary",
                REFERENCE,
                destination,
                offset=offset,
                prefix_sha256=hashlib.sha256(PAYLOAD[:offset]).hexdigest(),
            )

        if fault is None:
            download()
            assert destination.getvalue() == PAYLOAD
        else:
            with pytest.raises((ProtocolError, TransportError)):
                download()
            if fault not in ("truncated", "corrupt"):
                assert destination.getvalue() == PAYLOAD[:offset]
        assert len(ctx.events) == 1  # No automatic retries of any exchange.
        assert ctx.events[0][2]["Range"] == f"bytes={offset}-"
        assert ctx.events[0][2]["If-Match"] == '"' + REFERENCE.sha256 + '"'


def test_destination_failure_releases_the_borrowed_view_even_with_retained_exception():
    borrowed = []

    class Destination:
        def write(self, value):
            borrowed.append(value)
            raise OSError("destination full")

    with endpoint() as ctx:
        with pytest.raises(OSError, match="destination full") as error:
            ctx.client.download("/v1/binary", REFERENCE, Destination())
        # A local file failure is reported as itself, not as a transfer failure.
        assert type(error.value) is OSError and ctx.memory.used == (0, 0)
        with pytest.raises(ValueError, match="released memoryview"):
            len(borrowed[0])


def test_lost_response_is_not_a_second_submission_or_a_known_execution_failure():
    with endpoint() as ctx:
        with pytest.raises(TransportError) as error:
            ctx.client.request("POST", "/v1/lost", INFO)
        assert len(ctx.events) == 1 and "original request ID" in str(error.value)
        assert TOKEN not in str(error.value)


def test_http_failure_is_typed_but_does_not_invent_a_training_outcome():
    with endpoint() as ctx:
        with pytest.raises(RemoteError) as error:
            ctx.client.request("GET", "/v1/error")
        assert error.value.status == 403 and error.value.code == "not_authorized"
        assert not hasattr(error.value, "execution_effect")


@pytest.mark.parametrize(
    "change",
    [
        {"Content-Length": str(MAX_METADATA_BYTES + 1)},
        {"Content-Length": "-1"},
        {"Content-Length": "5, 5"},
        {"Transfer-Encoding": "chunked"},
        {"Content-Encoding": "gzip"},
        {"Content-Type": "text/html"},
    ],
)
def test_malformed_metadata_headers_fail_before_unbounded_body_decoding(change):
    with (
        endpoint(
            patch=lambda status, headers, body: (status, {**headers, **change}, body)
        ) as ctx,
        pytest.raises(ProtocolError),
    ):
        ctx.client.request("GET", "/v1/control")


def test_redirects_never_forward_credentials_to_another_address():
    def redirect(status, headers, body):
        body = encode(ServiceError("redirect", "No redirect"))
        return (
            307,
            {
                **headers,
                "Location": "http://elsewhere.invalid/",
                "Content-Length": str(len(body)),
            },
            body,
        )

    with endpoint(patch=redirect) as ctx:
        with pytest.raises(RemoteError) as error:
            ctx.client.request("POST", "/v1/control", INFO)
        assert error.value.status == 307 and len(ctx.events) == 1


def test_total_header_bytes_are_bounded_before_standard_library_parsing():
    def large(status, headers, body):
        return status, {**headers, **{f"X-{i}": "x" * 4096 for i in range(20)}}, body

    with endpoint(patch=large) as ctx:
        with pytest.raises(TransportError, match="LineTooLong"):
            ctx.client.request("GET", "/v1/control")
        assert ctx.memory.used == (0, 0)


@pytest.mark.parametrize("fault", ["digest", "reference", "truncated"])
def test_binary_identity_and_integrity_are_required(fault):
    def patch(status, headers, body):
        if fault == "digest":
            body = b"X" + body[1:]
        elif fault == "truncated":
            body = body[:-1]
        return status, headers, body

    reference = (
        replace(REFERENCE, sha256="0" * 64) if fault == "reference" else REFERENCE
    )
    with endpoint(patch=patch) as ctx, pytest.raises(ProtocolError):
        ctx.client.download("/v1/binary", reference, io.BytesIO())


def test_exchange_capacity_applies_before_opening_another_connection():
    with endpoint(block=True) as ctx:
        replies = []
        first = Thread(
            target=lambda: replies.append(ctx.client.request("GET", "/v1/control"))
        )
        second = Thread(
            target=lambda: replies.append(ctx.client.request("GET", "/v1/control"))
        )
        first.start()
        assert ctx.entered.wait(5)
        second.start()
        assert ctx.memory.used == (HTTP_SCRATCH_BYTES, 1) and len(ctx.events) == 1
        ctx.release.set()
        for thread in (first, second):
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert len(replies) == 2 and len(ctx.events) == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://external.invalid",
        "https://name:password@host",
        "https://host/?token=x",
        "https://host/#part",
        "file:///tmp/service",
    ],
)
def test_service_url_excludes_credential_paths_and_plaintext_remote_hosts(url):
    with pytest.raises(ValueError):
        HTTPTransport(url, token=TOKEN, memory=BufferBudget(HTTP_SCRATCH_BYTES))


def test_transport_import_needs_no_gpu_framework_engine_or_catalog():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {
            'torch', 'triton', 'shadowspill', 'shadowspill_workloads'
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from shadowspill_client.transport.http import HTTPTransport
from shadowspill_client.protocol.codec import encode, decode
from shadowspill_client.protocol.service import ServiceInfo
assert decode(encode(ServiceInfo('server', 'owner'))).owner_id == 'owner'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
