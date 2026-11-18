"""Content fingerprints for stable local files, with a deployment-owned cache.

The cache reuses a checksum only while device, inode, size, mtime and ctime all
match. It is an optimization for trusted local checkpoint storage, not a storage
scrubber: callers can omit the cache to force a fresh content read. Cache files
are expendable and never modify the checkpoint directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time_ns
from typing import BinaryIO

from shadowspill.diagnostics.progress import phase

CHUNK_BYTES = 1 << 20
_cache: ContextVar[Path | None] = ContextVar("checkpoint_digest_cache", default=None)


@contextmanager
def checksum_cache(directory: Path | None) -> Iterator[None]:
    """Select a service-owned cache for this resolution/initialization context."""
    token = _cache.set(directory)
    try:
        yield
    finally:
        _cache.reset(token)


def file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def settled_identity(facts: tuple[int, ...]) -> bool:
    # Even nanosecond stat fields can be updated only once per filesystem/kernel
    # clock tick. Do not trust a just-written file's timestamps as a cache key.
    return facts[-1] < time_ns() - 2_000_000_000


def _entry(directory: Path, path: Path) -> Path:
    return directory / (hashlib.sha256(os.fsencode(path)).hexdigest() + ".json")


def _read(directory: Path | None, path: Path, facts: tuple[int, ...]) -> str | None:
    if directory is None or not settled_identity(facts):
        return None
    try:
        with _entry(directory, path).open("rb") as stream:
            data = stream.read(4097)
        if len(data) > 4096:
            return None
        value = json.loads(data)
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or value.get("path") != str(path)
            or value.get("identity") != list(facts)
            or any(type(item) is not int for item in value["identity"])
            or not isinstance(value.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
        ):
            return None
        return str(value["sha256"])
    except (OSError, ValueError):
        return None


def _write(
    directory: Path | None, path: Path, facts: tuple[int, ...], checksum: str
) -> None:
    if directory is None:
        return
    temporary = None
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "path": str(path), "identity": facts, "sha256": checksum},
            separators=(",", ":"),
        ).encode()
        if len(payload) > 4096:
            return
        with NamedTemporaryFile(dir=directory, prefix=".digest-", delete=False) as out:
            temporary = Path(out.name)
            out.write(payload)
        # Concurrent readers see an entire old or new entry. A cache write does
        # not need fsync: missing/invalid entries simply trigger another scan.
        temporary.replace(_entry(directory, path))
        temporary = None
    except OSError:
        pass  # A read-only/unavailable cache cannot change initialization.
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _digest(source: BinaryIO, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        data = source.read(min(CHUNK_BYTES, remaining))
        if not data:
            raise ValueError("truncated checkpoint file")
        digest.update(data)
        remaining -= len(data)
    if source.read(1):
        raise ValueError("checkpoint file grew")
    return digest.hexdigest()


def sha256_file(path: Path, size: int) -> str:
    """Hash once per file identity; reject changes during a read or cache lookup."""
    path = path.resolve(strict=True)
    directory = _cache.get()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError(f"checkpoint file size/type differs: {path.name}")
        facts = file_identity(before)
        settled = settled_identity(facts)
        checksum = _read(directory, path, facts)
        cached = checksum is not None
        if checksum is None:
            with phase(f"Fingerprint {path.name} ({size / (1 << 30):.2f} GiB)"):
                checksum = _digest(stream, size)
                if not settled:
                    # Recent files can change within one timestamp tick. Preserve
                    # content checking in this unusual case; old model checkpoints
                    # take the single-read/cache path.
                    stream.seek(0)
                    if _digest(stream, size) != checksum:
                        raise ValueError(
                            f"checkpoint file changed while pinning: {path.name}"
                        )
        if (
            file_identity(os.fstat(stream.fileno())) != facts
            or file_identity(path.stat()) != facts
        ):
            raise ValueError(f"checkpoint file changed while pinning: {path.name}")
    if not cached and settled:
        _write(directory, path, facts, checksum)
    return checksum
