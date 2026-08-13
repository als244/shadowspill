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

from .inductor_adapter import ExecutableTaskManifest
from .output_contract import TaskStorageContract
from .profiling import ProfileKey

_SCHEMA = "shadowspill.compiled_task_manifest/v2"


class CompiledManifestCache:
    """Atomic compiler-manifest sidecars keyed by a structural profile key."""

    def __init__(self, profile_cache_root: Path) -> None:
        self.root = profile_cache_root / "compiled_manifests" / "v2"

    def read(
        self,
        key: ProfileKey,
        *,
        semantic_contract: TaskStorageContract,
    ) -> ExecutableTaskManifest | None:
        """Return a validated manifest or ``None`` when it needs hydration."""

        path = self.root / f"{key.digest}.json"
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
            return ExecutableTaskManifest.from_dict(
                payload["manifest"],
                semantic_contract=semantic_contract,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def write(
        self,
        key: ProfileKey,
        manifest: ExecutableTaskManifest,
    ) -> None:
        """Atomically publish a manifest; cache failure never changes semantics."""

        payload = {
            "schema": _SCHEMA,
            "profile_key_digest": key.digest,
            "graph_digest": key.graph_digest,
            "manifest": manifest.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.root,
                prefix=f".{key.digest}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(descriptor, "w") as temporary:
                    temporary.write(encoded)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, self.root / f"{key.digest}.json")
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary_name)
        except OSError:
            # This sidecar only avoids rebuilding unselected entrypoints. The
            # current process still owns the complete validated manifest.
            return


__all__ = ["CompiledManifestCache"]
