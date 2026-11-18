"""Authenticated bounded HTTP adapters for existing run and artifact owners."""

from __future__ import annotations

import http.client
import socket
import ssl
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BufferedIOBase, BufferedReader, RawIOBase
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import parse_qs, unquote, urlsplit

from shadowspill.orchestration.nodes.authentication import NodeTokens
from shadowspill.orchestration.nodes.protocol import HTTP_BYTES as NODE_HTTP_BYTES
from shadowspill.orchestration.nodes.registry import NodeRegistry
from shadowspill.orchestration.nodes.selection import HardwareSelectionError
from shadowspill.training.session.writers import Attachment
from shadowspill_client._records import Record, number
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.codec import (
    MAX_METADATA_BYTES,
    ProtocolError,
    RecordCodec,
    decode,
    encode,
)
from shadowspill_client.protocol.hardware import HardwareInventory
from shadowspill_client.protocol.payload_io import BufferReader
from shadowspill_client.protocol.preview import TrainingDescription, TrainingPreview
from shadowspill_client.protocol.requests import OperationRequest, identifier
from shadowspill_client.protocol.results import RequestStatus
from shadowspill_client.protocol.service import (
    CONTROL_MEDIA_TYPE,
    ServiceError,
    ServiceInfo,
)

from .artifacts import ArtifactStore
from .authentication import BearerTokens
from .records.requests import RequestRecords, RunIdentity, WriterFenced
from .routes import artifacts as artifact_routes
from .routes import nodes as node_routes
from .routes import observations
from .routes import requests as request_routes
from .routes import runs as run_routes

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

    from .jobs import RunJobs

_CHUNK = 64 << 10
HTTP_SCRATCH_BYTES = 16 * MAX_METADATA_BYTES + 2 * _CHUNK
_BUSY_BODY = encode(ServiceError("backpressure", "Service exchange capacity is busy."))
_BUSY_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: "
    + CONTROL_MEDIA_TYPE.encode("ascii")
    + b"\r\nContent-Length: "
    + str(len(_BUSY_BODY)).encode("ascii")
    + b"\r\nConnection: close\r\n\r\n"
    + _BUSY_BODY
)


class Controller(Protocol):
    def register(self, request: OperationRequest) -> RequestStatus: ...

    def receive(
        self,
        request_id: str,
        stream: BufferReader | None = None,
        *,
        timeout: float | None = 0,
    ) -> RequestStatus: ...

    def status(self, request_id: str) -> RequestStatus: ...

    def wait(
        self, request_id: str, *, accepted: bool = False, timeout: float | None = None
    ) -> RequestStatus: ...

    def attach(self, writer_id: str) -> Attachment: ...


@dataclass(frozen=True)
class RunAccess:
    records: RequestRecords
    controller: Controller | None

    def live(self) -> Controller:
        if self.controller is None:
            raise RuntimeError(
                "Run has no live controller; inspect its saved requests."
            )
        return self.controller


class _Headers:
    def __init__(self, source: BufferedIOBase) -> None:
        self.source, self.remaining = source, _CHUNK

    def readline(self, limit: int = -1) -> bytes:
        maximum = self.remaining + 1
        value = self.source.readline(min(limit, maximum) if limit >= 0 else maximum)
        self.remaining -= len(value)
        if self.remaining < 0:
            raise http.client.LineTooLong("service request headers")
        return value


class _SocketReader(RawIOBase):
    """Check the whole exchange deadline before every underlying socket read.

    A per-read inactivity timeout alone does not bound slow partial headers or
    uploads. BufferedReader may need many receives for one readline/readinto.
    The handler owns the socket; closing this view never closes a training run.
    """

    def __init__(self, handler: _Handler) -> None:
        self.handler: _Handler | None = handler

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: WriteableBuffer, /) -> int:
        if self.handler is None:
            raise ValueError("HTTP input stream is closed")
        self.handler.remaining_time()
        return self.handler.connection.recv_into(buffer)

    def close(self) -> None:
        # Break rfile -> raw reader -> handler before releasing the exchange
        # reservation; headers and their credentials must not wait for cyclic GC.
        self.handler = None
        super().close()


class _Body:
    def __init__(self, handler: _Handler, length: int) -> None:
        self.handler, self.length, self.remaining = handler, length, length

    def readinto(self, buffer: memoryview, /) -> int:
        if not self.remaining or not buffer:
            return 0
        self.handler.remaining_time()
        with buffer[: min(len(buffer), self.remaining)] as part:
            count = self.handler.rfile.readinto(part)
            if type(count) is not int or not 0 < count <= len(part):
                raise ProtocolError("incomplete HTTP request body")
        self.remaining -= count
        return count

    def record(
        self, *, codec: RecordCodec | None = None, max_bytes: int = MAX_METADATA_BYTES
    ) -> Record:
        if not 0 < self.length <= max_bytes:
            raise ProtocolError("control body exceeds the portable metadata bound")
        payload = bytearray(self.length)
        try:
            with memoryview(payload) as view:
                offset = 0
                while offset < len(view):
                    with view[offset : offset + _CHUNK] as part:
                        offset += self.readinto(part)
            return (
                decode(bytes(payload))
                if codec is None
                else codec.decode(bytes(payload), max_bytes=max_bytes)
            )
        finally:
            del payload


class HTTPService(ThreadingHTTPServer):
    """Transport only: lookup contexts keep each existing owner alive during I/O.

    The parent service supplies node-backed scratch, authentication and owner/run
    lookups. Admission precedes thread creation. Shutdown closes the listener and
    waits for bounded handlers; it does not close training runs on disconnect.
    """

    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        incarnation: str,
        authentication: BearerTokens,
        runs: Callable[[RunIdentity], AbstractContextManager[RunAccess]],
        artifacts: Callable[[str], AbstractContextManager[ArtifactStore]],
        memory: BufferBudget,
        jobs: RunJobs | None = None,
        describe: Callable[[TrainingDescription], TrainingPreview] | None = None,
        hardware: Callable[[], HardwareInventory] | None = None,
        nodes: NodeRegistry | None = None,
        node_authentication: NodeTokens | None = None,
        connections: BufferBudget | None = None,
        request_timeout: float = 120,
        tls: ssl.SSLContext | None = None,
    ) -> None:
        self.incarnation = identifier(incarnation, "server incarnation")
        self.authentication, self.runs, self.artifacts = authentication, runs, artifacts
        self.memory = memory
        self.jobs = jobs
        # The deployment callback reserves its CPU resolution envelope. HTTP
        # keeps decoded request/response metadata in its existing scratch owner.
        self.describe = describe
        self.hardware = hardware
        if (nodes is None) != (node_authentication is None) or (
            nodes is not None and connections is None
        ):
            raise ValueError(
                "node control requires authentication and connection admission"
            )
        if node_authentication is not None:
            node_authentication.require_separate(authentication)
        self.nodes, self.node_authentication = nodes, node_authentication
        self.connections = connections
        if jobs is not None and jobs.storage.incarnation != self.incarnation:
            raise ValueError("creation registry belongs to another server incarnation")
        if memory.max_bytes < HTTP_SCRATCH_BYTES:
            raise ValueError("HTTP scratch cannot fit the configured node arena")
        self.request_timeout = number(
            request_timeout, "HTTP request timeout", minimum=0
        )
        if not self.request_timeout:
            raise ValueError("HTTP request timeout must be positive")
        # Listener addresses are deployment configuration, never request values.
        if tls is None and address[0] not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("a reachable service listener requires TLS")
        self._owners: dict[socket.socket, BufferLease] = {}
        self._mutex = Lock()
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _Handler, bind_and_activate=False)
        try:
            if tls is not None:
                self.socket = tls.wrap_socket(
                    self.socket, server_side=True, do_handshake_on_connect=False
                )
            self.server_bind()
            self.server_activate()
        except BaseException:
            self.server_close()
            raise

    def process_request(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: tuple[str, int],
    ) -> None:
        if not isinstance(request, socket.socket):
            raise TypeError("HTTP requires a stream socket")
        try:
            lease = (
                self.memory.reserve(HTTP_SCRATCH_BYTES, timeout=0)
                if self.connections is None
                else self.connections.reserve(NODE_HTTP_BYTES, timeout=0)
            )
        except (RuntimeError, TimeoutError):
            # TLS handshakes also need an admitted thread. Do not start one on
            # the listener thread merely to send a capacity error.
            if not isinstance(request, ssl.SSLSocket):
                request.settimeout(0.1)
                with suppress(OSError):
                    request.sendall(_BUSY_RESPONSE)
            self.shutdown_request(request)
            return
        with self._mutex:
            self._owners[request] = lease
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release(request)
            raise

    def process_request_thread(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: tuple[str, int],
    ) -> None:
        if not isinstance(request, socket.socket):
            raise TypeError("HTTP requires a stream socket")
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release(request)

    def _release(self, request: socket.socket) -> None:
        with self._mutex:
            lease = self._owners.pop(request)
        lease.close()

    def handle_error(self, request: object, client_address: object) -> None:
        # Request headers, tokens and client bodies must never enter default
        # socketserver traceback logging. Durable run failures have their journal.
        pass


class _Handler(BaseHTTPRequestHandler):
    server: HTTPService
    connection: socket.socket
    protocol_version = "HTTP/1.1"
    server_version = "ShadowSpill"
    sys_version = ""

    def setup(self) -> None:
        self._response_started = False
        self.request.settimeout(self.server.request_timeout)
        self.deadline = monotonic() + self.server.request_timeout
        if isinstance(self.request, ssl.SSLSocket):
            self.request.do_handshake()
        super().setup()
        self.rfile.close()
        self.rfile = BufferedReader(_SocketReader(self))

    def log_message(self, format: str, *args: object) -> None:
        pass

    def parse_request(self) -> bool:
        original = self.rfile
        self.rfile = cast(BufferedReader, _Headers(original))
        try:
            return super().parse_request()
        finally:
            self.rfile = original

    def remaining_time(self) -> float:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP exchange deadline exceeded")
        self.connection.settimeout(remaining)
        return remaining

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        self.close_connection = True
        self._reply(ServiceError("http_error", "Invalid HTTP request."), code)

    def handle_expect_100(self) -> bool:
        self.send_error(417)
        return False

    def do_GET(self) -> None:
        self._serve()

    def do_POST(self) -> None:
        self._serve()

    def do_PUT(self) -> None:
        self._serve()

    def do_DELETE(self) -> None:
        self._serve()

    def _serve(self) -> None:
        self.close_connection = True
        client_lease = None
        try:
            if self.path.startswith("/v1/nodes/"):
                # Small, separately admitted requests keep long polls from owning
                # the much larger client upload/metadata scratch arena.
                parsed = urlsplit(self.path)
                if parsed.query or parsed.fragment or len(self.path) > 256:
                    raise ProtocolError("invalid node route")
                node_routes.dispatch(
                    self, tuple(parsed.path.split("/")[3:]), self._body()
                )
                return
            if self.server.connections is not None:
                client_lease = self.server.memory.reserve(HTTP_SCRATCH_BYTES, timeout=0)
            owner = self.server.authentication.owner(
                self.headers.get_all("Authorization") or ()
            )
            self._dispatch(owner)
        except WriterFenced:
            self._reply(
                ServiceError(
                    "writer_fenced", "Writer lease is fenced; reattach explicitly."
                ),
                403,
            )
        except PermissionError:
            self._reply(ServiceError("not_authorized", "Service access denied."), 403)
        except KeyError:
            self._reply(
                ServiceError("not_found", "Requested record is unavailable."), 404
            )
        except HardwareSelectionError as error:
            self._reply(ServiceError("hardware_selection", str(error)), 400)
        except (ValueError, TypeError):
            self._reply(
                ServiceError("invalid_request", "Invalid request metadata or body."),
                400,
            )
        except TimeoutError:
            self._reply(
                ServiceError("backpressure", "Request could not acquire capacity."), 429
            )
        except RuntimeError:
            self._reply(
                ServiceError(
                    "run_unavailable", "Run is unavailable for this operation."
                ),
                409,
            )
        except (OSError, http.client.HTTPException):
            # A lost response never rolls back accepted work. The controller
            # owns uploads/execution and their durable failure classification.
            return
        except Exception:
            self._reply(ServiceError("service_failed", "Service request failed."), 500)
        finally:
            if client_lease is not None:
                client_lease.close()

    def _body(self) -> _Body:
        lengths = self.headers.get_all("Content-Length") or ()
        if (
            len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
            or len(lengths[0]) > 19
            or int(lengths[0]) > (1 << 63) - 1
            or self.headers.get_all("Transfer-Encoding")
            or self.headers.get("Content-Encoding", "identity") != "identity"
        ):
            raise ProtocolError("request needs one uncompressed byte length")
        return _Body(self, int(lengths[0]))

    def _dispatch(self, owner: str) -> None:
        if len(self.path) > 8192:
            raise ProtocolError("request path exceeds its bound")
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise ProtocolError("use a relative service path")
        parts = tuple(
            unquote(part, errors="strict") for part in parsed.path.split("/")[1:]
        )
        for part in parts:
            identifier(part, "route component")
        query = parse_qs(
            parsed.query, strict_parsing=True, keep_blank_values=True, max_num_fields=4
        )
        if any(len(values) != 1 for values in query.values()):
            raise ProtocolError("query fields must not repeat")
        body = self._body()
        if self.command == "GET" and body.length:
            raise ProtocolError("read requests have no body")
        if parts == ("v1", "info") and self.command == "GET" and not query:
            self._reply(ServiceInfo(self.server.incarnation, owner))
            return
        if parts == ("v1", "hardware") and self.command == "GET" and not query:
            if self.server.hardware is None:
                raise KeyError("hardware inventory is unavailable")
            self._reply(self.server.hardware())
            return
        if parts[:2] == ("v1", "uploads") and not query:
            artifact_routes.uploads(self, owner, parts[2:], body)
            return
        if (
            parts == ("v1", "training", "describe")
            and self.command == "POST"
            and not query
        ):
            if self.server.describe is None:
                raise KeyError("setup inspection is unavailable")
            if self.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
                raise ProtocolError("setup has no control media type")
            settings = body.record()
            if not isinstance(settings, TrainingDescription):
                raise ProtocolError(
                    "inspection requires a training description request"
                )
            self._reply(self.server.describe(settings))
            return
        if (
            len(parts) == 3
            and parts[:2] == ("v1", "artifacts")
            and self.command == "GET"
            and not query
        ):
            artifact_routes.download(self, owner, parts[2])
            return
        if parts[:2] == ("v1", "runs") and len(parts) in (3, 4):
            run_routes.creation(self, owner, parts, query, body)
            return
        if len(parts) < 5 or parts[:2] != ("v1", "runs"):
            raise KeyError("route")

        identity = RunIdentity(owner, parts[2], parts[3])
        if parts[4:] == ("progress",):
            run_routes.progress(self, identity, query, body)
            return
        if parts[4:] == ("plans",):
            run_routes.plans(self, identity, query, body)
            return
        if parts[4:] == ("plans", "report"):
            run_routes.plan_report(self, identity, query, body)
            return
        if parts[4:] == ("nsys",):
            observations.nsys(self, identity, query, body)
            return
        request_routes.dispatch(self, identity, parts[4:], query, body)

    def _headers(
        self,
        status: int,
        length: int,
        media: str,
        *,
        etag: str | None = None,
        content_range: str | None = None,
    ) -> None:
        if (
            len(media) > 256
            or not media.isascii()
            or any(ord(char) < 32 or ord(char) == 127 for char in media)
        ):
            raise ValueError("stored artifact has an invalid HTTP media type")
        self.remaining_time()
        self._response_started = True
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        if etag is not None:
            self.send_header("ETag", '"' + etag + '"')
            self.send_header("Accept-Ranges", "bytes")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.end_headers()

    def _reply(
        self,
        value: Record,
        status: int = 200,
        *,
        content_range: str | None = None,
        codec: RecordCodec | None = None,
        max_bytes: int = MAX_METADATA_BYTES,
    ) -> None:
        if self._response_started:
            # Never append an error record to a partially written binary body.
            return
        encoded = (
            encode(value) if codec is None else codec.encode(value, max_bytes=max_bytes)
        )
        self._headers(
            status, len(encoded), CONTROL_MEDIA_TYPE, content_range=content_range
        )
        self.wfile.write(encoded)
