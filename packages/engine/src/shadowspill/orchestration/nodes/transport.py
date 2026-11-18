"""Outbound authenticated node control; one bounded request per connection."""

from __future__ import annotations

import http.client
import ssl
from urllib.parse import urlsplit

from shadowspill_client._records import Record, number
from shadowspill_client.protocol.codec import decode
from shadowspill_client.protocol.service import (
    CONTROL_MEDIA_TYPE,
    ServiceError,
    bearer_token,
)

from .protocol import CODEC, CONTROL_BYTES
from .registry import NodeUnavailable


class NodeTransport:
    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        ssl_context: ssl.SSLContext | None = None,
        timeout: float = 15,
    ) -> None:
        url = urlsplit(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("node service URL requires an HTTP(S) origin")
        if url.scheme == "http" and url.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            raise ValueError("remote node control requires HTTPS")
        if ssl_context is not None and (
            ssl_context.verify_mode != ssl.CERT_REQUIRED
            or not ssl_context.check_hostname
        ):
            raise ValueError(
                "node TLS must verify the service certificate and hostname"
            )
        self._url, self._token, self._tls = url, bearer_token(token), ssl_context
        self.timeout = number(timeout, "node HTTP timeout", minimum=0)
        if not self.timeout:
            raise ValueError("node HTTP timeout must be positive")

    def exchange(self, route: str, request: Record) -> Record:
        if route not in ("join", "heartbeat", "poll", "receipt", "leave"):
            raise ValueError("unknown node control route")
        url = self._url
        assert url.hostname is not None
        connection = (
            http.client.HTTPSConnection(
                url.hostname, url.port, timeout=self.timeout, context=self._tls
            )
            if url.scheme == "https"
            else http.client.HTTPConnection(
                url.hostname, url.port, timeout=self.timeout
            )
        )
        try:
            connection.request(
                "POST",
                "/v1/nodes/" + route,
                CODEC.encode(request, max_bytes=CONTROL_BYTES),
                headers={
                    "Authorization": "Bearer " + self._token,
                    "Content-Type": CONTROL_MEDIA_TYPE,
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            length = response.getheader("Content-Length")
            if (
                length is None
                or not length.isdecimal()
                or int(length) > CONTROL_BYTES
                or response.getheader("Content-Type") != CONTROL_MEDIA_TYPE
                or response.getheader("Transfer-Encoding") is not None
            ):
                raise ValueError("node response exceeds its bounded control contract")
            body = response.read(int(length))
            if len(body) != int(length):
                raise OSError("incomplete node control response")
            if response.status != 200:
                error = decode(body, max_bytes=CONTROL_BYTES)
                if response.status in (429, 503):
                    raise TimeoutError("node control is busy")
                if isinstance(error, ServiceError):
                    raise NodeUnavailable(f"node control rejected: {error.code}")
                raise NodeUnavailable("node control rejected")
            return CODEC.decode(body, max_bytes=CONTROL_BYTES)
        finally:
            connection.close()
