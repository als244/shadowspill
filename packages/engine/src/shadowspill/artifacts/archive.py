"""One atomic archive per artifact; no code execution or cache dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Self, cast

from ._bounded_io import _BoundedWriter
from ._values import encoded


class Artifact:
    """A pipeline result that saves to one archive file.

    `Program`, `Profiles`, `ProfiledProgram`, `PoolMemoryProfile`, `Plan`,
    `SweepResult`, `CommunicationInfo` and `LocalPlans` are artifacts: `save`
    writes one and `load` reads it back. An archive is a ZIP file of data: a
    manifest, the metadata, and binary payloads such as task code, each
    payload stored once by its SHA-256. Loading runs nothing from the archive
    and needs no cache.
    """

    artifact_kind: ClassVar[str]
    metadata_limit_bytes: ClassVar[int | None] = None
    payload_limit_bytes: ClassVar[int | None] = None
    shared_metadata: ClassVar[bool] = False

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Self:
        raise NotImplementedError

    def _archive_payloads(self) -> Mapping[str, bytes]:
        return {}

    def _archive_value(self) -> dict[str, Any]:
        return self.to_dict()

    def save(
        self, path: str | os.PathLike[str], *, max_bytes: int | None = None
    ) -> None:
        """Save the artifact as an archive file.

        The archive is written to a temporary file beside ``path`` and then
        replaces ``path`` atomically, so a failed save leaves ``path`` as it
        was. The file and the directory entry naming it are synced before
        `save` returns, so a saved archive survives a crash.

        Parameters
        ----------
        path : str or os.PathLike
            The file to write; its directory is created.
        max_bytes : int, optional
            Largest size of the new archive, in bytes. Writing stops before
            the temporary file exceeds it. By default no limit.

        Raises
        ------
        ValueError
            If ``max_bytes`` is negative or not an integer, or the archive
            exceeds it.
        """
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("archive byte limit must be a nonnegative integer")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.shared_metadata:
            from shadowspill.serialization.sharing import writing_metadata
            from shadowspill.serialization.tables import encode as encode_tables

            with writing_metadata():
                data = encode_tables(self._archive_value())
            metadata_name, version = "artifact.bin", 2
        else:
            data = encoded(self._archive_value())
            metadata_name, version = "artifact.json", 1
        payloads = self._archive_payloads()
        manifest = {
            "format": "shadowspill.phase-artifact",
            "version": version,
            "kind": self.artifact_kind,
            "payload": metadata_name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "blobs": {key: len(value) for key, value in payloads.items()},
        }
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as stream:
                temporary = stream.name
                with (
                    (
                        nullcontext(stream)
                        if max_bytes is None
                        else _BoundedWriter(cast(BinaryIO, stream), max_bytes)
                    ) as target,
                    zipfile.ZipFile(
                        cast(BinaryIO, target), "w", zipfile.ZIP_DEFLATED
                    ) as archive,
                ):
                    archive.writestr("manifest.json", encoded(manifest))
                    archive.writestr(metadata_name, data)
                    for key, value in payloads.items():
                        if hashlib.sha256(value).hexdigest() != key:
                            raise ValueError(
                                "artifact binary payload has invalid digest"
                            )
                        archive.writestr(f"blobs/{key}", value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
            # The rename is durable only once the directory is synced.
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str] | BinaryIO,
        *,
        max_metadata_bytes: int | None = None,
        max_payload_bytes: int | None = None,
    ) -> Self:
        """Read an artifact that `save` wrote.

        Call it on the artifact's class, as in ``Plan.load(path)``. The
        decompressed sizes the archive declares are checked against the
        limits before any of it is read; then its manifest, its set of files
        and each file's SHA-256. The limits count serialized bytes, not the
        memory of the loaded objects. `save` writes the metadata as JSON
        (archive version 1) or, for `LocalPlans`, as tables of shared values
        (version 2); either version loads.

        Parameters
        ----------
        path : str, os.PathLike or binary file
            The archive, or a seekable binary file holding it.
        max_metadata_bytes : int, optional
            Largest decompressed size of the manifest and metadata, in bytes;
            for version 2, also the largest size of the metadata expanded to
            JSON. By default no limit, or 8 GiB for `LocalPlans`.
        max_payload_bytes : int, optional
            Largest decompressed size of the binary payloads, in bytes. By
            default no limit, or zero for `LocalPlans`, which holds none.

        Returns
        -------
        Artifact
            The artifact, of the class ``load`` is called on.

        Raises
        ------
        ValueError
            If a limit is negative or not an integer, the archive exceeds a
            limit, it holds another kind of artifact or an unsupported
            version, its files differ from its manifest, a file fails its
            checksum, or its metadata is malformed.
        zipfile.BadZipFile
            If ``path`` is not a readable ZIP file.
        """
        metadata_limit = (
            cls.metadata_limit_bytes
            if max_metadata_bytes is None
            else max_metadata_bytes
        )
        payload_limit = (
            cls.payload_limit_bytes if max_payload_bytes is None else max_payload_bytes
        )
        for name, limit in (("metadata", metadata_limit), ("payload", payload_limit)):
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError(f"{name} byte limit must be a nonnegative integer")
        with zipfile.ZipFile(path) as archive:
            entries = {entry.filename: entry for entry in archive.infolist()}
            metadata_names = entries.keys() & {"artifact.json", "artifact.bin"}
            if (
                len(entries) != len(archive.infolist())
                or "manifest.json" not in entries
                or len(metadata_names) != 1
            ):
                raise ValueError("invalid phase archive contents")
            metadata_name = next(iter(metadata_names))
            metadata_bytes = sum(
                entries[name].file_size for name in ("manifest.json", metadata_name)
            )
            payload_bytes = sum(
                entry.file_size
                for name, entry in entries.items()
                if name not in {"manifest.json", metadata_name}
            )
            if metadata_limit is not None and metadata_bytes > metadata_limit:
                raise ValueError("phase archive exceeds metadata byte limit")
            if payload_limit is not None and payload_bytes > payload_limit:
                raise ValueError("phase archive exceeds payload byte limit")
            manifest = _json_object(archive.read("manifest.json"))
            version = manifest.get("version")
            if (
                manifest.get("format") != "shadowspill.phase-artifact"
                or type(version) is not int
                or version not in (1, 2)
                or manifest.get("kind") != cls.artifact_kind
                or manifest.get("payload") != metadata_name
                or metadata_name
                != ("artifact.bin" if version == 2 else "artifact.json")
            ):
                raise ValueError(f"unsupported archive for {cls.artifact_kind}")
            blobs = manifest.get("blobs", {})
            if not isinstance(blobs, dict) or any(
                not isinstance(key, str)
                or len(key) != 64
                or any(c not in "0123456789abcdef" for c in key)
                or type(size) is not int
                or size < 0
                for key, size in blobs.items()
            ):
                raise ValueError("invalid phase archive binary manifest")
            expected = [
                metadata_name,
                "manifest.json",
                *[f"blobs/{key}" for key in blobs],
            ]
            if sorted(archive.namelist()) != sorted(expected):
                raise ValueError("invalid phase archive contents")
            data = archive.read(metadata_name)
            if hashlib.sha256(data).hexdigest() != manifest.get("sha256"):
                raise ValueError("phase archive payload checksum mismatch")
            payloads = {}
            for key, size in blobs.items():
                if entries[f"blobs/{key}"].file_size != size:
                    raise ValueError("phase archive binary payload checksum mismatch")
                payload = archive.read(f"blobs/{key}")
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != key:
                    raise ValueError("phase archive binary payload checksum mismatch")
                payloads[key] = payload
        if version == 2:
            from shadowspill.serialization.sharing import shared_metadata
            from shadowspill.serialization.tables import decode as decode_tables

            value, shared = decode_tables(data, expanded_limit=metadata_limit)
            with shared_metadata(shared):
                return cls.from_dict(value, payloads=payloads)
        return cls.from_dict(_json_object(data), payloads=payloads)


def _json_object(data: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in items:
            if name in result:
                raise ValueError(f"duplicate archive JSON key: {name}")
            result[name] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"nonfinite archive JSON value: {value}")

    def floating(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            constant(value)
        return result

    value = json.loads(
        data, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating
    )
    if not isinstance(value, dict):
        raise ValueError("archive JSON must contain an object")
    return value
