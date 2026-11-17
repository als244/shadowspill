"""Bound and publish preparation artifacts and metadata."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

from shadowspill.server.artifacts import ArtifactStore
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.startup.records import creation_key
from shadowspill_client.protocol.requests import ArtifactRef


def publish_file(
    artifacts: ArtifactStore,
    identity: RunIdentity,
    path: Path,
    name: str,
    *,
    media_type: str = "application/octet-stream",
) -> ArtifactRef:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    reference = ArtifactRef(
        "prepared." + creation_key(identity, name),
        digest,
        path.stat().st_size,
        media_type,
    )
    artifacts.put(reference, lambda _: path.open("rb"))
    return reference


def publish_metadata(
    artifacts: ArtifactStore, identity: RunIdentity, name: str, value: dict[str, object]
) -> None:
    body = json.dumps(value, separators=(",", ":")).encode()
    if len(body) > 16 << 10:
        raise ValueError("preparation manifest exceeds its per-record metadata bound")
    reference = ArtifactRef(
        "preparation." + creation_key(identity, name),
        hashlib.sha256(body).hexdigest(),
        len(body),
        "application/json",
    )
    artifacts.put(reference, lambda _: io.BytesIO(body))


def preparation_files(
    root: Path | None, *, max_files: int, max_disk_bytes: int
) -> tuple[Path, ...]:
    if root is None:
        return ()
    files, total, count = [], 0, 0
    if not root.exists():
        return ()
    for directory, children, filenames in os.walk(root, followlinks=False):
        for name in (*children, *filenames):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("preparation artifacts cannot contain symbolic links")
            count += 1
            if count > max_files:
                raise ValueError(
                    "preparation exceeds its configured file-count envelope"
                )
            total += 4096 + (path.stat().st_size if path.is_file() else 0)
            if total > max_disk_bytes:
                raise ValueError("preparation exceeds its configured disk envelope")
            if path.is_file() and path.suffix == ".ssp":
                files.append(path)
    return tuple(sorted(files))
