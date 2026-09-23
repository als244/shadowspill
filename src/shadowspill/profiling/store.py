"""Atomic content-addressed storage for task measurements."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from shadowspill.store import (
    CONTRIBUTE,
    ArtifactRecorder,
    StorePolicy,
    digest_directory,
)
from shadowspill.task.profiles import PROFILE_SCHEMA, ProfileKey, TaskMeasurement


class ProfileStore:
    """Atomic per-key JSON cache independent of planning task identity."""

    def __init__(
        self,
        root: str | Path,
        *,
        compiled_manifest_root: str | Path | None = None,
        policy: StorePolicy = CONTRIBUTE,
        artifact_recorder: ArtifactRecorder | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.compiled_manifest_root = (
            Path(compiled_manifest_root).expanduser()
            if compiled_manifest_root is not None
            else self.root / "compiled_manifests"
        )
        self.policy = policy
        self.artifact_recorder = artifact_recorder

    def path(self, key: ProfileKey) -> Path:
        return digest_directory(self.root, key.digest) / "measurement.json"

    def read(self, key: ProfileKey) -> TaskMeasurement | None:
        if not self.policy.read_enabled:
            return None
        path = self.path(key)
        payload = self._read_payload(path)
        if payload is None:
            self.policy.refuse_miss("profile", key.digest)
            return None
        self._validate_payload(path, key, payload)
        measurement = self._decode(payload)
        if measurement is None:
            self.policy.refuse_miss("profile", key.digest)
            return None
        self._record(key, path, "read")
        return measurement

    @staticmethod
    def _decode(payload: dict[str, object]) -> TaskMeasurement | None:
        """The measurement one entry holds, or nothing this build can read.

        The envelope says which build wrote the entry and which key it
        answers, and an envelope that disagrees is corruption. The record
        inside it is a snapshot of the measurement contract that wrote it,
        and none is migrated: a record this build cannot read is one this
        build has not got, so it is measured again and written over.
        """

        try:
            return TaskMeasurement.from_dict(payload.get("measurement"))
        except ValueError:
            return None

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
        if not self.policy.write_enabled:
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
        if not path.exists() or self.policy.overwrite or replace_invalid:
            return False
        try:
            existing = path.read_text()
        except OSError as error:
            raise ValueError(f"profile cache entry {path} cannot be read") from error
        if existing == encoded:
            return True
        if self._decode(self._parsed(existing)) is None:
            return False
        raise ValueError(
            "fresh profiling differs from an existing cache entry; "
            "use a 'refresh' store mode or a new export_bypass_key: "
            f"{path}"
        )

    @staticmethod
    def _parsed(encoded: str) -> dict[str, object]:
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

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


__all__ = ["ArtifactRecorder", "ProfileStore"]
