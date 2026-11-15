"""Authenticated bounded HTTP adapters for existing run and artifact owners."""

from __future__ import annotations

import http.client
import re
import socket
import ssl
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BufferedIOBase, BufferedReader, RawIOBase
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import parse_qs, unquote, urlsplit

from shadowspill.training.session.writers import Attachment
from shadowspill_client._records import Record, number
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    WEIGHTS_MEDIA_TYPE,
)
from shadowspill_client.protocol.codec import (
    MAX_METADATA_BYTES,
    ProtocolError,
    decode,
    encode,
)
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.payload_io import BufferReader
from shadowspill_client.protocol.plans import SavedPlans
from shadowspill_client.protocol.preview import TrainingPreview
from shadowspill_client.protocol.requests import OperationRequest, identifier
from shadowspill_client.protocol.results import RequestStatus
from shadowspill_client.protocol.runs import RunInfo, RunPage
from shadowspill_client.protocol.service import (
    CONTROL_MEDIA_TYPE,
    ServiceError,
    ServiceInfo,
)
from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.protocol.training import ResolvedTraining
from shadowspill_client.protocol.uploads import ArtifactUpload

from .artifacts import ArtifactStore
from .authentication import BearerTokens
from .checkpoint_restore import (
    read_saved_manifest,
    require_checkpoint_dependencies,
)
from .creation_records import CreationRecord
from .nsight import missing_manifest, read_manifest
from .records.artifact_index import ArtifactAlias
from .records.requests import RequestRecords, RunIdentity, WriterFenced
from .records.run_index import RUN_BYTES

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

    def record(self) -> Record:
        if not 0 < self.length <= MAX_METADATA_BYTES:
            raise ProtocolError("control body exceeds the portable metadata bound")
        payload = bytearray(self.length)
        try:
            with memoryview(payload) as view:
                offset = 0
                while offset < len(view):
                    with view[offset : offset + _CHUNK] as part:
                        offset += self.readinto(part)
            return decode(bytes(payload))
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
        describe: Callable[[TrainingSetup | ResolvedTraining], TrainingPreview]
        | None = None,
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
            lease = self.memory.reserve(HTTP_SCRATCH_BYTES, timeout=0)
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
        try:
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
        if parts[:2] == ("v1", "uploads") and not query:
            self._uploads(owner, parts[2:], body)
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
            if not isinstance(settings, (TrainingSetup, ResolvedTraining)):
                raise ProtocolError("inspection requires setup or resolved settings")
            self._reply(self.server.describe(settings))
            return
        if (
            len(parts) == 3
            and parts[:2] == ("v1", "artifacts")
            and self.command == "GET"
            and not query
        ):
            with self.server.artifacts(owner) as store:
                if store.owner_id != owner:
                    raise PermissionError("artifact lookup crossed owner scope")
                reference = store.get(parts[2])
                if reference is None:
                    raise KeyError(parts[2])
                ranges = self.headers.get_all("Range") or []
                offset = 0
                if ranges:
                    if (
                        len(ranges) != 1
                        or re.fullmatch(r"bytes=(?:0|[1-9][0-9]{0,18})-", ranges[0])
                        is None
                    ):
                        raise ProtocolError("artifact range requires one start offset")
                    offset = int(ranges[0][6:-1])
                matches = self.headers.get_all("If-Match") or []
                if (ranges or matches) and matches != ['"' + reference.sha256 + '"']:
                    self._reply(
                        ServiceError(
                            "artifact_changed",
                            "Range requires the exact artifact checksum.",
                        ),
                        412,
                    )
                    return
                if ranges and offset >= reference.size_bytes:
                    self._reply(
                        ServiceError(
                            "invalid_range",
                            "Artifact range starts outside its content.",
                        ),
                        416,
                        content_range=f"bytes */{reference.size_bytes}",
                    )
                    return
                with store.open(reference) as stream:
                    stream.seek(offset)
                    self._headers(
                        206 if ranges else 200,
                        reference.size_bytes - offset,
                        reference.media_type,
                        etag=reference.sha256,
                        content_range=(
                            f"bytes {offset}-{reference.size_bytes - 1}"
                            f"/{reference.size_bytes}"
                            if ranges
                            else None
                        ),
                    )
                    remaining = reference.size_bytes - offset
                    while remaining:
                        self.remaining_time()
                        block = stream.read(min(_CHUNK, remaining))
                        if not block:
                            raise OSError("stored artifact became truncated")
                        self.wfile.write(block)
                        remaining -= len(block)
            return
        if parts[:2] == ("v1", "runs") and len(parts) in (3, 4):
            self._creation(owner, parts, query, body)
            return
        if len(parts) < 5 or parts[:2] != ("v1", "runs"):
            raise KeyError("route")

        identity = RunIdentity(owner, parts[2], parts[3])
        if parts[4:] == ("plans",):
            self._plans(identity, query, body)
            return
        if parts[4:] == ("nsys",):
            self._nsys(identity, query, body)
            return
        with self.server.runs(identity) as run:
            if run.records.identity != identity:
                raise PermissionError("run lookup crossed authenticated scope")
            route = parts[4:]
            if route == ("metrics",) and self.command == "GET":
                if set(query) - {"after", "through", "limit"}:
                    raise ProtocolError("unknown metric query field")
                self._reply(
                    run.records.metrics(
                        after=int(query.get("after", ["0"])[0]),
                        through=None
                        if "through" not in query
                        else int(query["through"][0]),
                        limit=int(query.get("limit", ["100"])[0]),
                    )
                )
                return
            if (
                len(route) == 2
                and route[0] == "writers"
                and self.command == "POST"
                and not query
                and not body.length
            ):
                attachment = run.live().attach(route[1])
                self._reply(attachment.lease, 200 if attachment.ready else 202)
                return
            if route == ("requests",) and self.command == "POST" and not query:
                if self.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
                    raise ProtocolError("request has no control media type")
                value = body.record()
                if (
                    not isinstance(value, OperationRequest)
                    or value.identity.run_id != identity.run_id
                ):
                    raise ProtocolError("request identity differs from its run")
                if run.controller is None:
                    if run.records.get(value.identity.request_id) is None:
                        run.live()
                    status = run.records.submit(value)  # Known digest only, no replay.
                else:
                    status = run.controller.register(value)
                self._reply(status)
                return
            if len(route) in (2, 3) and route[0] == "requests":
                row = run.records.get(route[1])
                if row is None:
                    raise KeyError(route[1])
                request, status = row
                if len(route) == 2 and self.command == "GET":
                    if set(query) - {"wait", "timeout"}:
                        raise ProtocolError("unknown request query")
                    if query:
                        mode = query.get("wait", [""])[0]
                        timeout = number(
                            float(query.get("timeout", ["10"])[0]), "wait", minimum=0
                        )
                        if mode not in ("accepted", "completed") or timeout > 10:
                            raise ProtocolError("request wait exceeds its bound")
                        if run.controller is not None:
                            try:
                                status = run.controller.wait(
                                    route[1],
                                    accepted=mode == "accepted",
                                    timeout=min(timeout, self.remaining_time()),
                                )
                            except TimeoutError:
                                status = run.controller.status(route[1])
                    self._reply(status)
                    return
                if route[2:] == ("input",) and self.command == "PUT" and not query:
                    expected = request.payload
                    if body.length != (0 if expected is None else expected.size_bytes):
                        raise ProtocolError("body differs from registered input length")
                    if (
                        expected is not None
                        and self.headers.get("Content-Type") != expected.media_type
                    ):
                        raise ProtocolError("body differs from registered input type")
                    if status.state == "uploading":
                        status = run.live().receive(
                            route[1], None if expected is None else body
                        )
                    self._reply(status)
                    return
            raise KeyError("route")

    def _uploads(self, owner: str, parts: tuple[str, ...], body: _Body) -> None:
        """Authenticated artifact registration; never creates or mutates a run."""
        with self.server.artifacts(owner) as store:
            if store.owner_id != owner:
                raise PermissionError("upload crossed artifact owner scope")
            if not parts and self.command == "POST":
                if self.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
                    raise ProtocolError("upload requires control metadata")
                request = body.record()
                if type(request) is not ArtifactUpload:
                    raise ProtocolError(
                        "upload requires an immutable artifact reference"
                    )
                self._reply(store.uploads.begin(request.reference))
                return
            if not parts:
                raise KeyError("unknown upload route")
            name = parts[0]
            status = store.uploads.get(name)
            if status is None:
                raise KeyError("unknown upload")
            if len(parts) == 1 and self.command == "GET":
                self._reply(status)
                return
            if len(parts) == 4 and parts[1] == "chunks" and self.command == "PUT":
                if (
                    re.fullmatch(r"0|[1-9][0-9]{0,18}", parts[2]) is None
                    or self.headers.get("Content-Type") != status.reference.media_type
                ):
                    raise ProtocolError("invalid upload chunk offset or media type")
                self._reply(
                    store.uploads.chunk(
                        name, int(parts[2]), body.length, parts[3], body
                    )
                )
                return
            if body.length:
                raise ProtocolError("upload action has an unexpected body")
            if len(parts) == 2 and parts[1] == "complete" and self.command == "POST":

                def validate(path: Path) -> None:
                    if status.reference.media_type in (
                        CHECKPOINT_MEDIA_TYPE,
                        WEIGHTS_MEDIA_TYPE,
                    ):
                        with path.open("rb") as stream:
                            manifest = read_saved_manifest(status.reference, stream)
                        # A local backup contains ranks, not implicit copies of
                        # Plans or frozen bases. Restore checks those separately
                        # before starting workers, through the same resolver.
                        require_checkpoint_dependencies(store, manifest, setup=False)

                self._reply(store.uploads.complete(name, validate=validate))
                return
            if len(parts) == 1 and self.command == "DELETE":
                store.uploads.abort(name)
                self._reply(ArtifactUpload(status.reference))
                return
            raise KeyError("unknown upload route")

    def _nsys(
        self, identity: RunIdentity, query: dict[str, list[str]], body: _Body
    ) -> None:
        if self.command != "GET" or set(query) != {"rank"} or body.length:
            raise ProtocolError("invalid capture route")
        rank = int(query["rank"][0])
        jobs = self.server.jobs
        if jobs is None:
            raise KeyError("capture routes are unavailable")
        info = jobs.storage.catalog.get(identity)
        if info is None:
            raise KeyError("capture run")
        self._run_scope(info, identity)
        with self.server.artifacts(identity.owner_id) as store:
            value = read_manifest(store, identity, rank)
            if value is None:
                raise KeyError("rank capture")
            if info.plans is not None and value.plans != info.plans:
                raise ValueError("capture differs from its source Plans")
            if value.status == "pending" and info.state not in ("preparing", "ready"):
                # A restart may recover a terminal run but only a pending record.
                # Report the missing finalization; never infer a usable report.
                value = missing_manifest(
                    value, "Run ended without a finalized capture record."
                )
            self._reply(value)

    def _plans(
        self, identity: RunIdentity, query: dict[str, list[str]], body: _Body
    ) -> None:
        jobs = self.server.jobs
        if jobs is None:
            raise KeyError("saved Plan routes are unavailable")
        info = jobs.storage.catalog.get(identity)
        if info is None or info.plans is None:
            raise KeyError("run Plans")
        self._run_scope(info, identity)
        with self.server.artifacts(identity.owner_id) as store:
            if store.owner_id != identity.owner_id:
                raise PermissionError("Plan lookup crossed owner scope")
            if self.command == "POST" and not query:
                if self.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
                    raise ProtocolError("Plan name has no control media type")
                saved = body.record()
                if not isinstance(saved, SavedPlans) or (
                    saved.session_id,
                    saved.run_id,
                    saved.plans,
                ) != (identity.session_id, identity.run_id, info.plans):
                    raise ProtocolError("saved Plans differ from their source run")
                store.name(
                    saved.plans,
                    ArtifactAlias(
                        saved.session_id,
                        saved.run_id,
                        saved.name,
                        saved.plans.id,
                        "plans",
                    ),
                )
            elif self.command == "GET" and set(query) == {"name"}:
                name = query["name"][0]
                reference = store.resolve_alias(
                    identity.session_id, identity.run_id, name, kind="plans"
                )
                if reference is None:
                    raise KeyError("Plan name")
                if reference != info.plans:
                    raise ValueError("saved Plans differ from their source run")
                saved = SavedPlans(
                    identity.session_id, identity.run_id, name, reference
                )
            else:
                raise ProtocolError("invalid saved Plan route")
            self._reply(saved)

    def _creation(
        self,
        owner: str,
        parts: tuple[str, ...],
        query: dict[str, list[str]],
        body: _Body,
    ) -> None:
        jobs = self.server.jobs
        if jobs is None:
            raise KeyError("creation routes are unavailable")
        catalog = jobs.storage.catalog
        if len(parts) == 3 and self.command == "GET":
            if set(query) - {"after", "limit"}:
                raise ProtocolError("unknown run-list query")
            limit = int(query.get("limit", ["32"])[0])
            if not 1 <= limit <= 128:
                raise ProtocolError("run page limit must be in 1..128")
            limit = min(limit, MAX_METADATA_BYTES // RUN_BYTES)
            rows = catalog.page(
                owner, parts[2], after=query.get("after", [""])[0], limit=limit
            )
            if any(row.owner_id != owner or row.session_id != parts[2] for row in rows):
                raise PermissionError("run page crossed authenticated scope")
            cursor = rows[-1].run_id if len(rows) == limit else None
            # Full lifecycle rows include failure/provenance headroom. Return
            # fewer rows when needed to fit the common control envelope.
            self._reply(RunPage(rows, cursor))
            return
        if len(parts) != 4 or query:
            raise KeyError("route")
        identity = RunIdentity(owner, parts[2], parts[3])
        if self.command == "POST":
            if self.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
                raise ProtocolError("creation has no control media type")
            creation = body.record()
            if not isinstance(creation, TrainingCreation):
                raise ProtocolError("creation requires resolved training choices")
            record = CreationRecord.encode(creation)

            def publish() -> None:
                with jobs.storage.artifacts(owner) as artifacts:
                    record.publish(artifacts)

            jobs.create(
                identity,
                request_sha256=record.reference.sha256,
                configuration=record.configuration,
                name=creation.name,
                labels=creation.labels,
                publish=publish,
            )
        elif self.command != "GET":
            raise KeyError("route")
        info = catalog.get(identity)
        if info is None:
            raise KeyError("run")
        self._run_reply(info, identity)

    @staticmethod
    def _run_scope(info: RunInfo, identity: RunIdentity) -> None:
        if (info.owner_id, info.session_id, info.run_id) != (
            identity.owner_id,
            identity.session_id,
            identity.run_id,
        ):
            raise PermissionError("run catalog crossed authenticated scope")

    def _run_reply(self, info: RunInfo, identity: RunIdentity) -> None:
        self._run_scope(info, identity)
        assert self.server.jobs is not None
        pending = info.state == "preparing" or (
            info.state == "ready" and not self.server.jobs.available(identity)
        )
        self._reply(info, 202 if pending else 200)

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
        self, value: Record, status: int = 200, *, content_range: str | None = None
    ) -> None:
        if self._response_started:
            # Never append an error record to a partially written binary body.
            return
        encoded = encode(value)
        self._headers(
            status, len(encoded), CONTROL_MEDIA_TYPE, content_range=content_range
        )
        self.wfile.write(encoded)
