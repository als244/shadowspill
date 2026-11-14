"""Bounded HTTP metadata and borrowed binary streams, without redirects or replay."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import ssl
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BufferedReader
from typing import BinaryIO, cast
from urllib.parse import urlsplit

from .._records import Record, number
from ..buffers import BufferBudget
from ..protocol.codec import MAX_METADATA_BYTES, ProtocolError, decode, encode
from ..protocol.requests import ArtifactRef
from ..protocol.service import CONTROL_MEDIA_TYPE, ServiceError, bearer_token

CHUNK_BYTES = 64 << 10
HTTP_SCRATCH_BYTES = 16 * MAX_METADATA_BYTES + 2 * CHUNK_BYTES


class _Headers:
    """Limit the complete status/header section before the standard parser."""

    def __init__(self, source: BinaryIO) -> None:
        self.source, self.remaining = source, CHUNK_BYTES

    def readline(self, limit: int = -1) -> bytes:
        maximum = self.remaining + 1
        line = self.source.readline(min(limit, maximum) if limit >= 0 else maximum)
        self.remaining -= len(line)
        if self.remaining < 0:
            raise http.client.LineTooLong("service response headers")
        return line

    def close(self) -> None:
        self.source.close()


class _Response(http.client.HTTPResponse):
    def begin(self) -> None:
        if self.headers is not None:
            return
        source = self.fp
        if source is None:
            raise ProtocolError("HTTP response has no open stream")
        bounded = cast(BufferedReader, _Headers(source))
        # HTTPResponse.begin uses only readline/close. Restore the actual binary
        # stream before any body I/O; its bytes already buffered remain intact.
        self.fp = bounded
        try:
            super().begin()
        finally:
            if self.fp is bounded:
                self.fp = source


class TransportError(ConnectionError):
    """The exchange failed; query the original request ID before another mutation."""


class RemoteError(RuntimeError):
    def __init__(self, status: int, error: ServiceError) -> None:
        self.status, self.code, self.message = status, error.code, error.message
        super().__init__(f"{error.code}: {error.message}")


@dataclass(frozen=True)
class Reply:
    status: int
    value: Record


def loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HTTPTransport:
    """One authenticated connection per exchange; no implicit request retry.

    The caller owns binary source chunks and destination streams. Scratch and
    simultaneous exchanges use the supplied budget; the higher-level submission
    queue separately owns snapshots, futures and their bounded pending window.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        memory: BufferBudget,
        timeout: float = 120,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not isinstance(base_url, str) or len(base_url) > 4096:
            raise ValueError("provide a bounded HTTP(S) service URL")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "service URL must not contain credentials, query or fragment"
            )
        if parsed.scheme == "http" and not loopback(parsed.hostname):
            raise ValueError("use HTTPS or loopback HTTP through an SSH forward")
        self._scheme, self._host, self._port = (
            parsed.scheme,
            parsed.hostname,
            parsed.port,
        )
        self._prefix = parsed.path.rstrip("/")
        self._token = bearer_token(token)
        self._memory = memory
        if memory.max_bytes < HTTP_SCRATCH_BYTES:
            raise ValueError("HTTP metadata scratch cannot fit the configured budget")
        self._timeout = number(timeout, "HTTP timeout", minimum=0)
        if not self._timeout:
            raise ValueError("HTTP timeout must be positive")
        self._ssl = ssl_context

    @contextmanager
    def _connection(self) -> Iterator[http.client.HTTPConnection]:
        with self._memory.reserve(HTTP_SCRATCH_BYTES):
            connection = (
                http.client.HTTPSConnection(
                    self._host, self._port, timeout=self._timeout, context=self._ssl
                )
                if self._scheme == "https"
                else http.client.HTTPConnection(
                    self._host, self._port, timeout=self._timeout
                )
            )
            connection.response_class = _Response
            try:
                yield connection
            except (OSError, http.client.HTTPException) as error:
                kind = type(error).__name__
                # Socket/file failures must not keep borrowed input/output
                # chunks alive through an exception's lower-level stack.
                error.__traceback__ = None
                error.__cause__ = error.__context__ = None
                raise TransportError(
                    f"HTTP transfer failed ({kind}); inspect the original "
                    "request ID before retrying."
                ) from None
            finally:
                connection.close()

    def _start(
        self,
        connection: http.client.HTTPConnection,
        method: str,
        path: str,
        size: int,
        media_type: str,
    ) -> None:
        if (
            method not in ("GET", "POST", "PUT", "DELETE")
            or not path.startswith("/v1/")
            or len(path) > 8192
            or not path.isascii()
            or any(ord(char) < 33 or ord(char) == 127 for char in path)
            or "\\" in path
        ):
            raise ValueError("invalid service method or relative protocol path")
        if (
            len(media_type) > 256
            or not media_type.isascii()
            or any(ord(char) < 32 or ord(char) == 127 for char in media_type)
        ):
            raise ValueError("invalid binary media type")
        connection.putrequest(method, self._prefix + path, skip_accept_encoding=True)
        connection.putheader("Authorization", "Bearer " + self._token)
        connection.putheader("Content-Type", media_type)
        connection.putheader("Content-Length", str(size))
        connection.putheader("Accept", CONTROL_MEDIA_TYPE)
        connection.putheader("Connection", "close")
        connection.endheaders()

    @staticmethod
    def _length(response: http.client.HTTPResponse, *, maximum: int) -> int:
        lengths = response.headers.get_all("Content-Length") or []
        if (
            len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
            or len(lengths[0]) > 20
            or response.headers.get_all("Transfer-Encoding")
            or response.headers.get("Content-Encoding", "identity") != "identity"
        ):
            raise ProtocolError(
                "response requires one bounded uncompressed byte length"
            )
        length = int(lengths[0])
        if length > maximum:
            raise ProtocolError("response exceeds the expected byte bound")
        return length

    def _metadata(self, response: http.client.HTTPResponse) -> Reply:
        length = self._length(response, maximum=MAX_METADATA_BYTES)
        if response.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
            raise ProtocolError("response has no portable control metadata")
        data = response.read(length)
        if len(data) != length:
            raise ProtocolError("truncated control response")
        value = decode(data)
        if not 200 <= response.status < 300:
            if not isinstance(value, ServiceError):
                raise ProtocolError("unsuccessful response has no service error record")
            raise RemoteError(response.status, value)
        if isinstance(value, ServiceError):
            raise ProtocolError("successful response contains a service error")
        return Reply(response.status, value)

    def request(self, method: str, path: str, value: Record | None = None) -> Reply:
        with self._connection() as connection:
            body = b"" if value is None else encode(value)
            try:
                self._start(connection, method, path, len(body), CONTROL_MEDIA_TYPE)
                connection.send(body)
            finally:
                del body
            with connection.getresponse() as response:
                return self._metadata(response)

    def upload(
        self,
        path: str,
        reference: ArtifactRef,
        chunks: Iterable[bytes | memoryview],
    ) -> Reply:
        with self._connection() as connection:
            self._start(
                connection, "PUT", path, reference.size_bytes, reference.media_type
            )
            remaining = reference.size_bytes
            iterator = iter(chunks)
            chunks = ()
            chunk = None
            try:
                for chunk in iterator:
                    with memoryview(chunk).cast("B") as view:
                        if len(view) > remaining:
                            raise ValueError("binary upload exceeds its declared size")
                        for offset in range(0, len(view), CHUNK_BYTES):
                            with view[offset : offset + CHUNK_BYTES] as part:
                                connection.send(part)
                        remaining -= len(view)
                if remaining:
                    raise ValueError("binary upload ended before its declared size")
            finally:
                del chunk
                try:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                finally:
                    del iterator
            with connection.getresponse() as response:
                return self._metadata(response)

    def download(
        self, path: str, reference: ArtifactRef, destination: BinaryIO
    ) -> None:
        """Verify identity/length and hash while copying into caller storage."""
        with self._connection() as connection:
            self._start(connection, "GET", path, 0, CONTROL_MEDIA_TYPE)
            with connection.getresponse() as response:
                if response.status != 200:
                    self._metadata(response)
                    raise ProtocolError("binary download requires a complete response")
                if (
                    self._length(response, maximum=reference.size_bytes)
                    != reference.size_bytes
                    or response.headers.get("ETag") != '"' + reference.sha256 + '"'
                    or response.headers.get("Content-Type") != reference.media_type
                ):
                    raise ProtocolError("download differs from its immutable reference")
                digest = hashlib.sha256()
                remaining = reference.size_bytes
                scratch = bytearray(CHUNK_BYTES)
                try:
                    with memoryview(scratch) as view:
                        while remaining:
                            with view[: min(remaining, CHUNK_BYTES)] as part:
                                count = response.readinto(part)
                                if type(count) is not int or not 0 < count <= len(part):
                                    raise ProtocolError("binary download is truncated")
                                digest.update(part[:count])
                                offset = 0
                                while offset < count:
                                    with part[offset:count] as write_view:
                                        written = destination.write(write_view)
                                    if (
                                        type(written) is not int
                                        or not 0 < written <= count - offset
                                    ):
                                        raise OSError(
                                            "download destination made no progress"
                                        )
                                    offset += written
                                remaining -= count
                finally:
                    del scratch
                if digest.hexdigest() != reference.sha256:
                    raise ProtocolError(
                        "binary download checksum differs from its reference"
                    )
