"""Atomic content-addressed storage for task measurements."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from .records import PROFILE_SCHEMA, ProfileKey, TaskMeasurement


class PlanningArtifactRecorder(Protocol):
    """Callback used to publish planning-cache evidence."""

    def __call__(
        self,
        *,
        category: str,
        kind: str,
        digest: str | None,
        path: str | Path,
        access: str,
        schema: str | None,
        dependencies: tuple[str, ...] = (),
    ) -> None: ...


class ProfileRepository:
    """Atomic per-key JSON cache independent of planning task identity."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        compiled_manifest_root: str | Path | None = None,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: PlanningArtifactRecorder | None = None,
    ) -> None:
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path.home() / ".cache" / "shadowspill" / "profiles"
        )
        self.compiled_manifest_root = (
            Path(compiled_manifest_root).expanduser()
            if compiled_manifest_root is not None
            else self.root / "compiled_manifests" / "v2"
        )
        self.read_enabled = read_enabled
        self.write_enabled = write_enabled
        self.overwrite = overwrite
        self.artifact_recorder = artifact_recorder

    def path(self, key: ProfileKey) -> Path:
        return self.root / key.digest[:2] / f"{key.digest}.json"

    def read(self, key: ProfileKey) -> TaskMeasurement | None:
        if not self.read_enabled:
            return None
        path = self.path(key)
        payload = self._read_payload(path)
        if payload is None:
            return None
        self._validate_payload(path, key, payload)
        measurement = TaskMeasurement.from_dict(payload.get("measurement"))
        self._record(key, path, "read")
        return measurement

    def _read_payload(self, path: Path) -> dict[str, object] | None:
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"profile cache entry {path} cannot be read") from error
        if not isinstance(value, dict):
            raise ValueError(f"profile cache entry {path} has an invalid schema")
        return value

    @staticmethod
    def _validate_payload(
        path: Path,
        key: ProfileKey,
        payload: dict[str, object],
    ) -> None:
        if payload.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"profile cache entry {path} has an invalid schema")
        if payload.get("key_digest") != key.digest:
            raise ValueError(f"profile cache entry {path} has the wrong identity")

    def write(
        self,
        key: ProfileKey,
        measurement: TaskMeasurement,
        *,
        replace_invalid: bool = False,
    ) -> None:
        if not self.write_enabled:
            return
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = self._encode(key, measurement)
        if self._match_existing(path, encoded, replace_invalid):
            self._record(key, path, "matched")
            return
        self._atomic_write(path, key.digest, encoded)
        self._record(key, path, "write")

    @staticmethod
    def _encode(key: ProfileKey, measurement: TaskMeasurement) -> str:
        payload = {
            "schema": PROFILE_SCHEMA,
            "key_digest": key.digest,
            "measurement": measurement.to_dict(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _match_existing(
        self,
        path: Path,
        encoded: str,
        replace_invalid: bool,
    ) -> bool:
        if not path.exists() or self.overwrite or replace_invalid:
            return False
        try:
            existing = path.read_text()
        except OSError as error:
            raise ValueError(f"profile cache entry {path} cannot be read") from error
        if existing != encoded:
            raise ValueError(
                "fresh profiling differs from an existing cache entry; "
                "use overwrite_plan=True or a new implementation_revision: "
                f"{path}"
            )
        return True

    @staticmethod
    def _atomic_write(path: Path, digest: str, encoded: str) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def _record(self, key: ProfileKey, path: Path, access: str) -> None:
        if self.artifact_recorder is not None:
            self.artifact_recorder(
                category="profiling",
                kind="task_measurement",
                digest=key.digest,
                path=path,
                access=access,
                schema=PROFILE_SCHEMA,
                dependencies=(key.graph_digest,),
            )


__all__ = ["PlanningArtifactRecorder", "ProfileRepository"]
