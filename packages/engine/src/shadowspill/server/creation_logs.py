"""Read current or published creation logs within authenticated run ownership."""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Mapping

from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill_client.protocol.progress import (
    LOG_CHUNK_BYTES,
    LOG_STREAMS,
    LogChunk,
    ProgressRequest,
)
from shadowspill_client.protocol.requests import ArtifactRef

from .artifacts import ArtifactStore
from .creation_records import creation_key
from .records.requests import RunIdentity


def live_chunks(
    logs: Mapping[str, LogFiles], request: ProgressRequest
) -> tuple[LogChunk, ...]:
    chunks = []
    for name in LOG_STREAMS:
        phase, stream = name.split(".")
        if phase not in logs:
            continue
        offset = request.offsets.get(name, 0)
        result = logs[phase].read(stream, offset)
        if result is not None:
            data, complete, capped = result
            chunks.append(
                LogChunk(
                    name, offset, b64encode(data).decode("ascii"), complete, capped
                )
            )
    return tuple(chunks)


def archived_chunks(
    store: ArtifactStore,
    identity: RunIdentity,
    request: ProgressRequest,
    live: tuple[LogChunk, ...],
) -> tuple[LogChunk, ...]:
    if store.owner_id != identity.owner_id:
        raise PermissionError("creation logs crossed owner scope")
    result = {chunk.stream: chunk for chunk in live}
    for phase in dict.fromkeys(name.split(".")[0] for name in LOG_STREAMS):
        names = [f"{phase}.{stream}" for stream in ("stdout", "stderr")]
        if all(name in result for name in names):
            continue
        key = "conversion" if phase == "converting" else "phase:" + phase
        metadata = store.get("preparation." + creation_key(identity, key))
        if metadata is None:
            continue
        if metadata.size_bytes > 16 << 10 or metadata.media_type != "application/json":
            raise ValueError("invalid creation log manifest")
        with store.open(metadata) as source:
            manifest = json.loads(source.read((16 << 10) + 1))
        if (
            any(manifest.get(key) != value for key, value in identity.to_dict().items())
            or manifest.get("phase") != phase
        ):
            raise PermissionError("creation log manifest crossed run scope")
        for name in names:
            if name in result:
                continue
            stream = name.split(".")[1]
            reference = ArtifactRef.from_dict(manifest["streams"][stream])
            if (
                reference.media_type != "text/plain"
                or store.get(reference.id) != reference
            ):
                raise ValueError("creation log differs from its published stream")
            offset = request.offsets.get(name, 0)
            if offset > reference.size_bytes:
                raise ValueError("log offset exceeds available bytes")
            with store.open(reference) as source:
                source.seek(offset)
                data = source.read(min(LOG_CHUNK_BYTES, reference.size_bytes - offset))
            result[name] = LogChunk(
                name,
                offset,
                b64encode(data).decode("ascii"),
                offset + len(data) == reference.size_bytes,
                reference.size_bytes == manifest.get("limit_bytes"),
            )
    return tuple(result[name] for name in LOG_STREAMS if name in result)
