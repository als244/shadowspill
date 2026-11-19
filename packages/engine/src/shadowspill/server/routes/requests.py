"""Writer attachment, ordered request submission and bounded request polling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shadowspill_client._records import number
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import OperationRequest
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE

from ..records.requests import RunIdentity
from . import observations

if TYPE_CHECKING:
    from ..http import RunAccess, _Body, _Handler


def dispatch(
    exchange: _Handler,
    identity: RunIdentity,
    route: tuple[str, ...],
    query: dict[str, list[str]],
    body: _Body,
) -> None:
    with exchange.server.runs(identity) as run:
        if run.records.identity != identity:
            raise PermissionError("run lookup crossed authenticated scope")
        if route == ("report",) and exchange.command == "POST" and not query:
            observations.report(exchange, identity, run, body)
            return
        if route == ("metrics",):
            observations.metrics(exchange, run, query, body)
            return
        if (
            len(route) == 2
            and route[0] == "writers"
            and exchange.command == "POST"
            and not query
            and not body.length
        ):
            attachment = run.live().attach(route[1])
            exchange._reply(attachment.lease, 200 if attachment.ready else 202)
            return
        if route == ("requests",) and exchange.command == "POST" and not query:
            if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
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
            exchange._reply(status)
            return
        if len(route) in (2, 3) and route[0] == "requests":
            _request(exchange, run, route, query, body)
            return
        raise KeyError("route")


def _request(
    exchange: _Handler,
    run: RunAccess,
    route: tuple[str, ...],
    query: dict[str, list[str]],
    body: _Body,
) -> None:
    row = run.records.get(route[1])
    if row is None:
        raise KeyError(route[1])
    request, status = row
    if len(route) == 2 and exchange.command == "GET":
        if set(query) - {"wait", "timeout"}:
            raise ProtocolError("unknown request query")
        if query:
            mode = query.get("wait", [""])[0]
            timeout = number(float(query.get("timeout", ["10"])[0]), "wait", minimum=0)
            if mode not in ("accepted", "completed") or timeout > 10:
                raise ProtocolError("request wait exceeds its bound")
            if run.controller is not None:
                try:
                    status = run.controller.wait(
                        route[1],
                        accepted=mode == "accepted",
                        timeout=min(timeout, exchange.remaining_time()),
                    )
                except TimeoutError:
                    status = run.controller.status(route[1])
        exchange._reply(status)
        return
    if route[2:] == ("input",) and exchange.command == "PUT" and not query:
        expected = request.payload
        if body.length != (0 if expected is None else expected.size_bytes):
            raise ProtocolError("body differs from registered input length")
        if (
            expected is not None
            and exchange.headers.get("Content-Type") != expected.media_type
        ):
            raise ProtocolError("body differs from registered input type")
        if status.state == "uploading":
            # An earlier request can hold the staging this one needs until it
            # finishes. Wait for room for up to half of the exchange's time,
            # keeping the rest for the body; past that the client is told the
            # service is busy and sends the input again.
            status = run.live().receive(
                route[1],
                None if expected is None else body,
                timeout=exchange.remaining_time() / 2,
            )
        exchange._reply(status)
        return
    raise KeyError("route")
