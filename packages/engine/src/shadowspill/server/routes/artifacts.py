"""Bounded artifact downloads and resumable uploads within one owner scope."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill.server.startup.checkpoint import (
    read_saved_manifest,
    require_checkpoint_dependencies,
)
from shadowspill_client.protocol.checkpoints import (
    CHECKPOINT_MEDIA_TYPE,
    WEIGHTS_MEDIA_TYPE,
)
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.service import (
    CONTROL_MEDIA_TYPE,
    ServiceError,
)
from shadowspill_client.protocol.uploads import ArtifactUpload

if TYPE_CHECKING:
    from ..http import _Body, _Handler


_CHUNK = 64 << 10


def download(exchange: _Handler, owner: str, name: str) -> None:
    with exchange.server.artifacts(owner) as store:
        if store.owner_id != owner:
            raise PermissionError("artifact lookup crossed owner scope")
        reference = store.get(name)
        if reference is None:
            raise KeyError(name)
        ranges = exchange.headers.get_all("Range") or []
        offset = 0
        if ranges:
            if (
                len(ranges) != 1
                or re.fullmatch(r"bytes=(?:0|[1-9][0-9]{0,18})-", ranges[0]) is None
            ):
                raise ProtocolError("artifact range requires one start offset")
            offset = int(ranges[0][6:-1])
        matches = exchange.headers.get_all("If-Match") or []
        if (ranges or matches) and matches != ['"' + reference.sha256 + '"']:
            exchange._reply(
                ServiceError(
                    "artifact_changed",
                    "Range requires the exact artifact checksum.",
                ),
                412,
            )
            return
        if ranges and offset >= reference.size_bytes:
            exchange._reply(
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
            exchange._headers(
                206 if ranges else 200,
                reference.size_bytes - offset,
                reference.media_type,
                etag=reference.sha256,
                content_range=(
                    f"bytes {offset}-{reference.size_bytes - 1}/{reference.size_bytes}"
                    if ranges
                    else None
                ),
            )
            remaining = reference.size_bytes - offset
            while remaining:
                exchange.remaining_time()
                block = stream.read(min(_CHUNK, remaining))
                if not block:
                    raise OSError("stored artifact became truncated")
                exchange.wfile.write(block)
                remaining -= len(block)
    return


def uploads(
    exchange: _Handler, owner: str, parts: tuple[str, ...], body: _Body
) -> None:
    """Authenticated artifact registration; never creates or mutates a run."""
    with exchange.server.artifacts(owner) as store:
        if store.owner_id != owner:
            raise PermissionError("upload crossed artifact owner scope")
        if not parts and exchange.command == "POST":
            if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
                raise ProtocolError("upload requires control metadata")
            request = body.record()
            if type(request) is not ArtifactUpload:
                raise ProtocolError("upload requires an immutable artifact reference")
            exchange._reply(store.uploads.begin(request.reference))
            return
        if not parts:
            raise KeyError("unknown upload route")
        name = parts[0]
        status = store.uploads.get(name)
        if status is None:
            raise KeyError("unknown upload")
        if len(parts) == 1 and exchange.command == "GET":
            exchange._reply(status)
            return
        if len(parts) == 4 and parts[1] == "chunks" and exchange.command == "PUT":
            if (
                re.fullmatch(r"0|[1-9][0-9]{0,18}", parts[2]) is None
                or exchange.headers.get("Content-Type") != status.reference.media_type
            ):
                raise ProtocolError("invalid upload chunk offset or media type")
            exchange._reply(
                store.uploads.chunk(name, int(parts[2]), body.length, parts[3], body)
            )
            return
        if body.length:
            raise ProtocolError("upload action has an unexpected body")
        if len(parts) == 2 and parts[1] == "complete" and exchange.command == "POST":

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

            exchange._reply(store.uploads.complete(name, validate=validate))
            return
        if len(parts) == 1 and exchange.command == "DELETE":
            store.uploads.abort(name)
            exchange._reply(ArtifactUpload(status.reference))
            return
        raise KeyError("unknown upload route")
