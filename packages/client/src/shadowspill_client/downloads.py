"""Bounded, verified partial files retained only for explicit download resume."""

import hashlib
import os
import re
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ._records import Record, integer, record
from .buffers import BufferBudget
from .protocol.checkpoints import CheckpointManifest, WeightsManifest
from .protocol.codec import MAX_METADATA_BYTES, RecordCodec
from .protocol.requests import ArtifactRef
from .transport.http import HTTPTransport

PROGRESS_BYTES = 4096
CHECKPOINT_BYTES = 1 << 20
FILE_SCRATCH_BYTES = 128 << 10
CHECKPOINT_SCRATCH_BYTES = 16 * MAX_METADATA_BYTES


@dataclass(frozen=True)
class DownloadProgress(Record):
    reference: ArtifactRef
    offset: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference", record(ArtifactRef, self.reference, "download artifact")
        )
        integer(self.offset, "download prefix bytes")
        if self.offset > self.reference.size_bytes:
            raise ValueError("download progress exceeds its artifact")
        if (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("download progress requires its prefix checksum")


_CODEC = RecordCodec(
    {"artifact.download": DownloadProgress, "artifact.reference": ArtifactRef}
)


def partial_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".shadowspill-download")


@contextmanager
def _regular(path: Path, flags: int) -> Iterator[BinaryIO]:
    descriptor = os.open(
        path, (flags & ~os.O_TRUNC) | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (
            flags != os.O_RDONLY and info.st_nlink != 1
        ):
            raise ValueError("download progress requires regular files")
        if flags & os.O_TRUNC:
            os.ftruncate(descriptor, 0)
        stream = os.fdopen(
            descriptor, "rb" if flags == os.O_RDONLY else "r+b", buffering=0
        )
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resume_download(
    transport: HTTPTransport,
    reference: ArtifactRef,
    destination: Path,
    *,
    memory: BufferBudget,
) -> Path:
    """Retain a constant-size prefix record across errors, never retry an exchange.

    Linux/macOS advisory locking excludes concurrent writers and is released by
    process exit. Progress commits follow data fsync; uncommitted trailing bytes
    are discarded on explicit resume. The transport verifies the recorded prefix
    and complete file, and promotion never overwrites an existing destination.
    """
    with memory.reserve(FILE_SCRATCH_BYTES):
        return _resume(transport, reference, destination)


@contextmanager
def _directory_lock(directory: Path, name: str = "lock") -> Iterator[Path]:
    import fcntl

    directory.mkdir(mode=0o700, exist_ok=True)
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("download partial storage must be a directory")
    lock_path = directory / name
    with _regular(lock_path, os.O_RDWR | os.O_CREAT) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "download destination already has an active writer"
            ) from None
        owned, current = os.fstat(lock.fileno()), lock_path.lstat()
        if (owned.st_dev, owned.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("download partial ownership changed")
        yield lock_path


def _write_record(path: Path, value: Record) -> None:
    raw = _CODEC.encode(value, max_bytes=PROGRESS_BYTES)
    temporary = path.with_suffix(".pending")
    with _regular(temporary, os.O_RDWR | os.O_CREAT | os.O_TRUNC) as stream:
        part = memoryview(raw)
        while part:
            count = stream.write(part)
            if type(count) is not int or not 0 < count <= len(part):
                raise OSError("download progress write made no progress")
            part = part[count:]
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _resume(
    transport: HTTPTransport, reference: ArtifactRef, destination: Path
) -> Path:
    initial = DownloadProgress(reference, 0, hashlib.sha256().hexdigest())
    _CODEC.encode(initial, max_bytes=PROGRESS_BYTES)  # Bound before creating files.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    directory = partial_path(destination)
    with _directory_lock(directory) as lock_path:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name not in (
                    "lock",
                    "data",
                    "progress.json",
                    "progress.pending",
                ):
                    raise ValueError("download directory contains an unrelated file")
        metadata = directory / "progress.json"
        payload = directory / "data"
        existing = metadata.exists() or metadata.is_symlink()
        if existing:
            with _regular(metadata, os.O_RDONLY) as stream:
                saved = _CODEC.decode(
                    stream.read(PROGRESS_BYTES + 1), max_bytes=PROGRESS_BYTES
                )
            if not isinstance(saved, DownloadProgress) or saved.reference != reference:
                raise ValueError("download progress belongs to another artifact")
            flags = os.O_RDWR
        else:
            saved = initial
            flags = os.O_RDWR | os.O_CREAT
        with _regular(payload, flags) as target:
            size = os.fstat(target.fileno()).st_size
            if (
                not existing and size
            ) or not saved.offset <= size <= reference.size_bytes:
                raise ValueError("download partial size differs from retained progress")
            target.truncate(saved.offset)
            completed, checksum = saved.offset, saved.sha256
            checkpoint_failed = False

            def checkpoint() -> None:
                nonlocal saved, checkpoint_failed
                value = DownloadProgress(reference, completed, checksum)
                try:
                    os.fsync(target.fileno())
                    _write_record(metadata, value)
                except BaseException:
                    checkpoint_failed = True
                    raise
                saved = value

            def progress(offset: int, digest: str) -> None:
                nonlocal completed, checksum
                completed, checksum = offset, digest
                if (
                    offset - saved.offset >= CHECKPOINT_BYTES
                    or offset == reference.size_bytes
                ):
                    checkpoint()

            if not metadata.exists():
                checkpoint()
            try:
                transport.download(
                    "/v1/artifacts/" + reference.id,
                    reference,
                    target,
                    offset=saved.offset,
                    prefix_sha256=saved.sha256,
                    progress=progress,
                )
            except BaseException:
                if not checkpoint_failed and completed > saved.offset:
                    # Preserve the last complete write after a lost connection.
                    # Failed progress persistence retains the previous durable
                    # prefix; it must not obscure the original transfer failure.
                    with suppress(OSError):
                        checkpoint()
                raise
            # The existing transport checked the entire content digest. Link on
            # the same filesystem gives an atomic, non-overwriting publication.
            os.link(payload, destination, follow_symlinks=False)
            _sync_directory(destination.parent)
        for name in ("data", "progress.json", "progress.pending"):
            (directory / name).unlink(missing_ok=True)
        lock_path.unlink()
        directory.rmdir()
        _sync_directory(destination.parent)
    return destination


def resume_checkpoint(
    transport: HTTPTransport,
    reference: ArtifactRef,
    destination: Path,
    load: Callable[[], tuple[CheckpointManifest | WeightsManifest, bytes]],
    *,
    memory: BufferBudget,
) -> Path:
    """Retain verified ranks, resume partial ranks, and publish the manifest last."""
    with memory.reserve(CHECKPOINT_SCRATCH_BYTES):
        if destination.exists() or destination.is_symlink():
            if not stat.S_ISDIR(destination.lstat().st_mode) or (
                (destination / "manifest.json").exists()
                or (destination / "manifest.json").is_symlink()
            ):
                raise FileExistsError(destination)
            identity = destination / ".shadowspill-checkpoint.json"
            if not identity.exists() and not identity.is_symlink():
                with os.scandir(destination) as entries:
                    if any(
                        entry.name
                        not in (".shadowspill-lock", ".shadowspill-checkpoint.pending")
                        for entry in entries
                    ):
                        raise ValueError(
                            "checkpoint destination has no retained identity"
                        )
        manifest, body = load()
        with _directory_lock(destination, ".shadowspill-lock") as lock:
            identity = destination / ".shadowspill-checkpoint.json"
            allowed = {
                lock.name,
                identity.name,
                identity.with_suffix(".pending").name,
                "ranks",
                ".manifest.pending",
            }
            with os.scandir(destination) as entries:
                for entry in entries:
                    if entry.name not in allowed:
                        raise ValueError(
                            "checkpoint destination contains unrelated files"
                        )
            if identity.exists() or identity.is_symlink():
                with _regular(identity, os.O_RDONLY) as stream:
                    retained = _CODEC.decode(
                        stream.read(PROGRESS_BYTES + 1), max_bytes=PROGRESS_BYTES
                    )
                if retained != reference:
                    raise ValueError("checkpoint copy belongs to another artifact")
            else:
                if (destination / "ranks").exists() or (
                    destination / "ranks"
                ).is_symlink():
                    raise ValueError("checkpoint ranks have no retained identity")
                _write_record(identity, reference)
            ranks = destination / "ranks"
            ranks.mkdir(mode=0o700, exist_ok=True)
            if not stat.S_ISDIR(ranks.lstat().st_mode):
                raise ValueError("checkpoint ranks must be a directory")
            names = {f"rank-{row.rank:05d}.ssp" for row in manifest.ranks}
            with os.scandir(ranks) as entries:
                for entry in entries:
                    if entry.name not in names and not (
                        entry.name.endswith(".shadowspill-download")
                        and entry.name.removesuffix(".shadowspill-download") in names
                    ):
                        raise ValueError("checkpoint ranks contain an unrelated file")
            for row in manifest.ranks:
                target = ranks / f"rank-{row.rank:05d}.ssp"
                if target.exists() or target.is_symlink():
                    # Reuse the same bounded full-prefix verifier, without I/O
                    # to the service or overwriting a previously completed rank.
                    with _regular(target, os.O_RDONLY) as stream:
                        transport.download(
                            "/v1/artifacts/" + row.archive.id,
                            row.archive,
                            stream,
                            offset=row.archive.size_bytes,
                            prefix_sha256=row.archive.sha256,
                        )
                else:
                    resume_download(transport, row.archive, target, memory=memory)
            temporary = destination / ".manifest.pending"
            with _regular(temporary, os.O_RDWR | os.O_CREAT | os.O_TRUNC) as stream:
                part = memoryview(body)
                while part:
                    count = stream.write(part[: 64 << 10])
                    if type(count) is not int or not 0 < count <= min(
                        len(part), 64 << 10
                    ):
                        raise OSError("checkpoint manifest write made no progress")
                    part = part[count:]
                os.fsync(stream.fileno())
            os.link(temporary, destination / "manifest.json", follow_symlinks=False)
            _sync_directory(destination)
            temporary.unlink()
            identity.unlink()
            identity.with_suffix(".pending").unlink(missing_ok=True)
            lock.unlink()
            _sync_directory(destination)
        return destination
