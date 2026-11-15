"""Public lightweight connection to an existing authenticated training service."""

from __future__ import annotations

import os
import ssl
from typing import Any, Self, TypeVar
from uuid import uuid4

from ._async import CallFuture, ConnectionWork
from ._records import Record, integer
from .artifacts import ArtifactHandle
from .buffers import BufferBudget
from .protocol.codec import ProtocolError, encode
from .protocol.preview import TrainingPreview
from .protocol.requests import ArtifactRef, identifier
from .protocol.service import ServiceInfo
from .protocol.setup import TrainingSetup
from .protocol.training import ResolvedTraining
from .transport.http import HTTP_SCRATCH_BYTES, HTTPTransport
from .transport.ssh import SSH, SSH_BYTES, SSHTunnel

R = TypeVar("R", bound=Record)


class ServiceClient:
    """Connect over HTTPS or loopback HTTP, optionally using an owned SSH forward.

    Credentials come from ``token`` or ``SHADOWSPILL_TOKEN``. Connection closure
    ends local transfers and polling; it never closes a server training run.
    Pending metadata and network concurrency are bounded independently of the
    training configuration's input snapshots. No mutation is automatically retried.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        ssh: SSH | None = None,
        token: str | None = None,
        session_id: str | None = None,
        timeout: float = 120,
        max_connections: int = 4,
        max_pending_requests: int = 16,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if (base_url is None) == (ssh is None):
            raise ValueError("provide exactly one of base_url or ssh")
        integer(max_connections, "max_connections", minimum=1)
        self.session_id = identifier(session_id or uuid4().hex, "session ID")
        self._work = ConnectionWork(max_pending_requests, max_connections)
        self._memory = BufferBudget(
            max_connections * HTTP_SCRATCH_BYTES + SSH_BYTES, max_connections + 1
        )
        self._tunnel = None if ssh is None else SSHTunnel(ssh, memory=self._memory)
        try:
            if self._tunnel is not None:
                base_url = self._tunnel.start()
            assert base_url is not None
            self._transport = HTTPTransport(
                base_url,
                token=token
                if token is not None
                else os.environ.get("SHADOWSPILL_TOKEN", ""),
                memory=self._memory,
                timeout=timeout,
                ssl_context=ssl_context,
            )
            self.info = self._request("GET", "/v1/info", ServiceInfo)
        except BaseException:
            self.close()
            raise

    def _request(
        self, method: str, path: str, kind: type[R], value: Record | None = None
    ) -> R:
        self._work.check()
        result = self._transport.request(method, path, value).value
        if not isinstance(result, kind):
            raise ProtocolError(f"service response requires {kind.__name__}")
        return result

    @staticmethod
    def _setup(
        config: ResolvedTraining | None, setup: dict[str, Any]
    ) -> TrainingSetup | ResolvedTraining:
        if config is not None:
            if setup or not isinstance(config, ResolvedTraining):
                raise ValueError(
                    "config requires resolved settings without overlapping keywords"
                )
            settings: TrainingSetup | ResolvedTraining = config
        else:
            settings = TrainingSetup(**setup)
        # Bound the control tree before retaining it in the connection queue.
        encode(settings)
        return settings

    def describe_training(
        self, *, config: ResolvedTraining | None = None, **setup: Any
    ) -> CallFuture[TrainingPreview]:
        settings = self._setup(config, setup)
        return self._work.submit(
            lambda: self._request(
                "POST", "/v1/training/describe", TrainingPreview, settings
            )
        )

    def artifact(self, reference: ArtifactRef) -> ArtifactHandle:
        if not isinstance(reference, ArtifactRef):
            raise TypeError("artifact requires an immutable server reference")
        return ArtifactHandle(reference, self)

    def close(self) -> None:
        self._work.close()
        if self._tunnel is not None:
            self._tunnel.close()
        self._memory.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
