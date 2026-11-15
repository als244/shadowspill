"""Explicit laptop artifact transfers with verified server-side resume progress."""

from __future__ import annotations

import hashlib
import os
import stat
from _hashlib import HASH
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from .artifacts import ArtifactHandle
from .downloads import CHECKPOINT_SCRATCH_BYTES
from .protocol.checkpoints import CheckpointManifest, WeightsManifest
from .protocol.codec import MAX_METADATA_BYTES, ProtocolError, decode
from .protocol.requests import ArtifactRef
from .protocol.uploads import MAX_UPLOAD_CHUNK_BYTES, ArtifactUpload, UploadStatus

if TYPE_CHECKING:
    from .service import ServiceClient

_COPY_BYTES = 64 << 10
UPLOAD_SCRATCH_BYTES = MAX_UPLOAD_CHUNK_BYTES + 2 * _COPY_BYTES


def _stamp(facts: os.stat_result) -> tuple[int, ...]:
    return (
        facts.st_dev,
        facts.st_ino,
        facts.st_size,
        facts.st_mtime_ns,
        facts.st_ctime_ns,
    )


@contextmanager
def _source(path: Path) -> Iterator[BinaryIO]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("artifact upload requires a regular local file")
        stream = os.fdopen(fd, "rb", buffering=0)
    except BaseException:
        os.close(fd)
        raise
    with stream:
        yield stream


def _unchanged(path: Path, stream: BinaryIO, original: tuple[int, ...]) -> None:
    if (
        _stamp(os.fstat(stream.fileno())) != original
        or _stamp(path.lstat()) != original
    ):
        raise ValueError("local upload source changed during transfer")


def _hash_prefix(service: ServiceClient, stream: BinaryIO, count: int) -> HASH:
    digest = hashlib.sha256()
    while count:
        service._work.check()
        block = stream.read(min(count, _COPY_BYTES))
        try:
            if not block:
                raise ValueError("local upload source is truncated")
            digest.update(block)
            count -= len(block)
        finally:
            del block
    return digest


def _progress(value: object, reference: ArtifactRef) -> UploadStatus:
    if not isinstance(value, UploadStatus) or value.reference != reference:
        raise ProtocolError("upload reply differs from the immutable artifact")
    return value


def _file(
    service: ServiceClient, path: Path, reference: ArtifactRef | None
) -> ArtifactHandle:
    with service._memory.reserve(UPLOAD_SCRATCH_BYTES), _source(path) as stream:
        original = _stamp(os.fstat(stream.fileno()))
        size = original[2]
        if reference is None:
            # Establish a stable immutable identity before opening server state.
            checksum = _hash_prefix(service, stream, size).hexdigest()
            _unchanged(path, stream, original)
            reference = ArtifactRef("upload." + checksum, checksum, size)
            stream.seek(0)
        if reference.size_bytes != size:
            raise ValueError("local upload source size differs from its reference")
        status = _progress(
            service._request(
                "POST", "/v1/uploads", UploadStatus, ArtifactUpload(reference)
            ),
            reference,
        )
        route = "/v1/uploads/" + reference.id
        digest = _hash_prefix(service, stream, status.offset)
        if digest.hexdigest() != status.prefix_sha256:
            raise ValueError("local upload source differs from the retained prefix")
        _unchanged(path, stream, original)
        if status.state == "complete":
            return service.artifact(reference)
        offset = status.offset
        while offset < size:
            service._work.check()
            length = min(MAX_UPLOAD_CHUNK_BYTES, size - offset)
            block = stream.read(length)
            try:
                if len(block) != length:
                    raise ValueError("local upload source is truncated")
                digest.update(block)
                chunk_hash = hashlib.sha256(block).hexdigest()
                _unchanged(path, stream, original)
                if offset + length == size and digest.hexdigest() != reference.sha256:
                    raise ValueError(
                        "local upload source checksum differs from its reference"
                    )
                part = ArtifactRef(
                    reference.id, chunk_hash, length, reference.media_type
                )
                result = service._transport.upload(
                    f"{route}/chunks/{offset}/{chunk_hash}", part, (block,)
                ).value
                status = _progress(result, reference)
                offset += length
                if (
                    status.state != "uploading"
                    or status.offset != offset
                    or status.prefix_sha256 != digest.hexdigest()
                ):
                    raise ProtocolError(
                        "upload reply has inconsistent verified progress"
                    )
            finally:
                del block
        _unchanged(path, stream, original)
        status = _progress(
            service._request("POST", route + "/complete", UploadStatus), reference
        )
        if status.state != "complete":
            raise ProtocolError("upload completion did not publish its artifact")
        return service.artifact(reference)


def _checkpoint(
    service: ServiceClient, path: Path, reference: ArtifactRef | None
) -> ArtifactHandle:
    with service._memory.reserve(CHECKPOINT_SCRATCH_BYTES):
        manifest_path = path / "manifest.json"
        with _source(manifest_path) as stream:
            original = _stamp(os.fstat(stream.fileno()))
            if original[2] > MAX_METADATA_BYTES:
                raise ValueError(
                    "checkpoint upload manifest exceeds its metadata bound"
                )
            body = stream.read(MAX_METADATA_BYTES + 1)
            digest = hashlib.sha256(body).hexdigest()
            value = decode(body)
            if not isinstance(value, (CheckpointManifest, WeightsManifest)):
                raise ValueError("directory upload requires a saved-state manifest")
            manifest = value
            del value, body
            expected = reference or ArtifactRef(
                "upload." + digest, digest, original[2], manifest.info.media_type
            )
            if (
                expected.sha256 != digest
                or expected.size_bytes != original[2]
                or expected.media_type != manifest.info.media_type
            ):
                raise ValueError(
                    "checkpoint upload differs from its immutable reference"
                )
            # Check every local member before starting any server transfer.
            ranks = path / "ranks"
            if not stat.S_ISDIR(ranks.lstat().st_mode):
                raise ValueError("checkpoint ranks require a real directory")
            for row in manifest.ranks:
                target = ranks / f"rank-{row.rank:05d}.ssp"
                with _source(target) as source:
                    if os.fstat(source.fileno()).st_size != row.archive.size_bytes:
                        raise ValueError(
                            "checkpoint rank size differs from its reference"
                        )
            for row in manifest.ranks:
                _file(service, ranks / f"rank-{row.rank:05d}.ssp", row.archive)
            _unchanged(manifest_path, stream, original)
            # Register the manifest last, after all exact rank references exist.
            return _file(service, manifest_path, expected)


def upload(
    service: ServiceClient, path: Path, reference: ArtifactRef | None
) -> ArtifactHandle:
    if stat.S_ISDIR(path.lstat().st_mode):
        return _checkpoint(service, path, reference)
    return _file(service, path, reference)
