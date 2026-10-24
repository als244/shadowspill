"""One atomic archive per artifact; no code execution or cache dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

from ._values import encoded


class Artifact:
    """Explicit persistence shared by the public phase results."""

    artifact_kind: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Self:
        raise NotImplementedError

    def _archive_payloads(self) -> Mapping[str, bytes]:
        return {}

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = encoded(self.to_dict())
        payloads = self._archive_payloads()
        manifest = {
            "format": "shadowspill.phase-artifact",
            "version": 1,
            "kind": self.artifact_kind,
            "payload": "artifact.json",
            "sha256": hashlib.sha256(data).hexdigest(),
            "blobs": {key: len(value) for key, value in payloads.items()},
        }
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as stream:
                temporary = stream.name
                with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("manifest.json", encoded(manifest))
                    archive.writestr("artifact.json", data)
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
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if (
                manifest.get("format") != "shadowspill.phase-artifact"
                or manifest.get("version") != 1
                or manifest.get("kind") != cls.artifact_kind
                or manifest.get("payload") != "artifact.json"
            ):
                raise ValueError(f"unsupported archive for {cls.artifact_kind}")
            blobs = manifest.get("blobs", {})
            expected = [
                "artifact.json",
                "manifest.json",
                *[f"blobs/{key}" for key in blobs],
            ]
            if sorted(archive.namelist()) != sorted(expected):
                raise ValueError("invalid phase archive contents")
            data = archive.read("artifact.json")
            if hashlib.sha256(data).hexdigest() != manifest.get("sha256"):
                raise ValueError("phase archive payload checksum mismatch")
            payloads = {}
            for key, size in blobs.items():
                payload = archive.read(f"blobs/{key}")
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != key:
                    raise ValueError("phase archive binary payload checksum mismatch")
                payloads[key] = payload
        return cls.from_dict(json.loads(data), payloads=payloads)
