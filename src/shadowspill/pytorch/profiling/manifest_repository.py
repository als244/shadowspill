"""Profile-keyed sidecars for compiler-owned task storage manifests.

The ordinary profile cache already proves that a structural ABI has measured
timing and allocation geometry.  This adjacent sidecar preserves the exact
Inductor storage contract from that same compilation, allowing planning to
lower and select graph-pair variants before rebuilding executable callables.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.inductor import ExecutableTaskManifest
from shadowspill.pytorch.profiling.records import ProfileKey
from shadowspill.pytorch.profiling.repository import PlanningArtifactRecorder

_SCHEMA = "shadowspill.compiled_task_manifest/v2"


class CompiledManifestRepository:
    """Atomic compiler-manifest sidecars keyed by a structural profile key."""

    def __init__(
        self,
        root: Path,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: PlanningArtifactRecorder | None = None,
    ) -> None:
        self.root = root
        self.read_enabled = read_enabled
        self.write_enabled = write_enabled
        self.overwrite = overwrite
        self.artifact_recorder = artifact_recorder

    def path(self, key: ProfileKey) -> Path:
        return self.root / key.digest[:2] / f"{key.digest}.json"

    def read(
        self,
        key: ProfileKey,
        *,
        semantic_contract: TaskStorageContract,
    ) -> ExecutableTaskManifest | None:
        """Return a validated manifest or ``None`` when it needs hydration."""

        if not self.read_enabled:
            return None
        path = self.path(key)
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                return None
            expected = {
                "schema",
                "profile_key_digest",
                "graph_digest",
                "manifest",
            }
            if (
                set(payload) != expected
                or payload["schema"] != _SCHEMA
                or payload["profile_key_digest"] != key.digest
                or payload["graph_digest"] != key.graph_digest
                or not isinstance(payload["manifest"], dict)
            ):
                return None
            manifest = ExecutableTaskManifest.from_dict(
                payload["manifest"],
                semantic_contract=semantic_contract,
            )
            self._record(key, path, "read", manifest.compatibility_digest)
            return manifest
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def write(
        self,
        key: ProfileKey,
        manifest: ExecutableTaskManifest,
    ) -> None:
        """Atomically publish a manifest; cache failure never changes semantics."""

        if not self.write_enabled:
            return
        path = self.path(key)
        payload = {
            "schema": _SCHEMA,
            "profile_key_digest": key.digest,
            "graph_digest": key.graph_digest,
            "manifest": manifest.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not self.overwrite:
                if path.read_text() != encoded:
                    raise ValueError(
                        "fresh compiled manifest differs from an existing cache "
                        "entry; use overwrite_plan=True or a new "
                        f"implementation_revision: {path}"
                    )
                self._record(key, path, "matched", manifest.compatibility_digest)
                return
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{key.digest}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(descriptor, "w") as temporary:
                    temporary.write(encoded)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, path)
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary_name)
        except OSError:
            # This sidecar only avoids rebuilding unselected entrypoints. The
            # current process still owns the complete validated manifest.
            return
        self._record(key, path, "write", manifest.compatibility_digest)

    def _record(
        self,
        key: ProfileKey,
        path: Path,
        access: str,
        manifest_digest: str,
    ) -> None:
        if self.artifact_recorder is None:
            return
        self.artifact_recorder(
            category="profiling",
            kind="compiled_task_manifest",
            digest=key.digest,
            path=path,
            access=access,
            schema=_SCHEMA,
            dependencies=(key.graph_digest, manifest_digest),
        )


__all__ = ["CompiledManifestRepository"]
