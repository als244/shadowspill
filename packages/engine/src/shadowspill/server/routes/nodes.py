"""Node-daemon-only routes on the service listener; never use client owner authority."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shadowspill.orchestration.nodes.protocol import (
    CODEC,
    CONTROL_BYTES,
    Heartbeat,
    Join,
    Leave,
    Poll,
    Receipt,
    Reconcile,
)
from shadowspill_client._records import Record
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE

if TYPE_CHECKING:
    from shadowspill.server.http import _Body, _Handler


def dispatch(handler: _Handler, parts: tuple[str, ...], body: _Body) -> None:
    registry, authentication = handler.server.nodes, handler.server.node_authentication
    if registry is None or authentication is None:
        raise KeyError("node membership is unavailable")
    access = authentication.authorize(handler.headers.get_all("Authorization") or ())
    if (
        handler.command != "POST"
        or handler.headers.get("Content-Type") != CONTROL_MEDIA_TYPE
    ):
        raise ProtocolError("node control requires POST with the control media type")
    request = body.record(codec=CODEC, max_bytes=CONTROL_BYTES)
    response: Record
    if parts == ("join",) and isinstance(request, Join):
        response = registry.join(access, request)
    elif parts == ("heartbeat",) and isinstance(request, Heartbeat):
        response = registry.heartbeat(access.node_id, request)
    elif parts == ("poll",) and isinstance(request, Poll):
        response = registry.poll(access.node_id, request)
    elif parts == ("receipt",) and isinstance(request, Receipt):
        response = registry.complete(access.node_id, request)
    elif parts == ("leave",) and isinstance(request, Leave):
        response = registry.leave(access.node_id, request)
    elif parts == ("reconcile",) and isinstance(request, Reconcile):
        response = registry.reconcile(access.node_id, request)
    else:
        raise ProtocolError("node record does not match its route")
    handler._reply(response, codec=CODEC, max_bytes=CONTROL_BYTES)
