"""Read current or published creation logs within authenticated run ownership."""

from __future__ import annotations

import hashlib
import io
import json
from base64 import b64encode
from collections.abc import Mapping
from typing import Protocol

from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill_client.protocol.progress import (
    LOG_CHUNK_BYTES,
    LOG_STREAMS,
    MAX_LOG_STREAMS,
    LogChunk,
    ProgressRequest,
    log_stream,
)
from shadowspill_client.protocol.requests import ArtifactRef

from .artifacts import ArtifactStore
from .creation_records import creation_key
from .records.requests import RunIdentity


class LogReader(Protocol):
    def read(self, name: str, offset: int) -> tuple[bytes, bool, bool] | None: ...


def publish_metadata(
    store: ArtifactStore, identity: RunIdentity, key: str, value: dict[str, object]
) -> ArtifactRef:
    body = json.dumps({**identity.to_dict(), **value}, separators=(",", ":")).encode()
    if len(body) > 1 << 20:
        raise ValueError("creation log manifest exceeds its metadata bound")
    reference = ArtifactRef(
        "preparation." + creation_key(identity, key),
        hashlib.sha256(body).hexdigest(),
        len(body),
        "application/json",
    )
    store.put(reference, lambda _: io.BytesIO(body))
    return reference


def publish_logs(
    store: ArtifactStore, identity: RunIdentity, phase: str, logs: LogFiles
) -> ArtifactRef:
    streams = {}
    if logs.directory.exists():
        for name, reference in logs.references().items():
            store.put(reference, logs.open)
            streams[name] = reference.to_dict()
    return publish_metadata(
        store,
        identity,
        "phase:" + phase,
        {
            "phase": phase,
            "limit_bytes": logs.limit_bytes,
            "streams": streams,
        },
    )


def live_chunks(
    logs: Mapping[str, LogReader], request: ProgressRequest
) -> tuple[LogChunk, ...]:
    chunks = []
    names = [f"{phase}.{stream}" for phase in logs for stream in ("stdout", "stderr")]
    if len(names) > MAX_LOG_STREAMS or not all(log_stream(name) for name in names):
        raise ValueError("creation has too many or invalid log streams")
    chunk_bytes = min(LOG_CHUNK_BYTES, (256 << 10) // max(1, len(names)))
    for name in names:
        phase, stream = name.split(".")
        offset = request.offsets.get(name, 0)
        result = logs[phase].read(stream, offset)
        if result is not None:
            data, complete, capped = result
            complete = complete and len(data) <= chunk_bytes
            data = data[:chunk_bytes]
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
    phases = list(dict.fromkeys(name.split(".")[0] for name in LOG_STREAMS))
    index = store.get("preparation." + creation_key(identity, "worker-logs"))
    if index is not None:
        if index.size_bytes > 1 << 20 or index.media_type != "application/json":
            raise ValueError("invalid worker log index")
        with store.open(index) as source:
            listing = json.loads(source.read((1 << 20) + 1))
        if any(listing.get(key) != value for key, value in identity.to_dict().items()):
            raise PermissionError("worker log index crossed run scope")
        workers = listing["workers"]
        if (
            not isinstance(workers, list)
            or len(workers) * 2 + len(LOG_STREAMS) > MAX_LOG_STREAMS
            or any(not log_stream(str(name) + ".stdout") for name in workers)
            or len(set(workers)) != len(workers)
        ):
            raise ValueError("invalid worker log index")
        phases.extend(workers)
    chunk_bytes = min(LOG_CHUNK_BYTES, (256 << 10) // max(1, 2 * len(phases)))
    for phase in phases:
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
            if stream not in manifest["streams"]:
                continue  # Admission may fail before a worker has a log producer.
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
                data = source.read(min(chunk_bytes, reference.size_bytes - offset))
            result[name] = LogChunk(
                name,
                offset,
                b64encode(data).decode("ascii"),
                offset + len(data) == reference.size_bytes,
                reference.size_bytes == manifest.get("limit_bytes"),
            )
    return tuple(result.values())
