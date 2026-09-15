"""Step programs filed by the identity a build has before any capture.

A `StepProgram` is content-addressed by what it contains, which exists only
after the capture, profiling and lowering that produce it. This archive files
one under a second key, computed by the frontend from what it knows before
that work: the caller's export bypass key and the structural facts of the
request. A build with the key looks here first and, on a hit, skips the work.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill.schema import artifact_schema
from shadowspill.store import (
    CONTRIBUTE,
    ArtifactRecorder,
    StorePolicy,
    atomic_json,
    atomic_text,
    digest_directory,
)

if TYPE_CHECKING:
    from .program import StepProgram

_SCHEMA = artifact_schema("step_archive")


class StepArchive:
    """Step programs on disk, keyed by their pre-capture identity.

    Each entry holds `step_program.json`, the program exactly as the build
    returned it, and `manifest.json`, the identity it was filed under, so a
    reader can see why two builds were the same step.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        policy: StorePolicy = CONTRIBUTE,
        artifact_recorder: ArtifactRecorder | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.policy = policy
        self.artifact_recorder = artifact_recorder

    def path(self, key: str) -> Path:
        return digest_directory(self.root, key)

    def read(self, key: str) -> StepProgram | None:
        """The program filed under `key`, or None when there is none."""

        if not self.policy.read_enabled:
            return None
        directory = self.path(key)
        program_path = directory / "step_program.json"
        if not program_path.exists():
            return None
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"step archive entry {manifest_path} cannot be read"
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
            raise ValueError(
                f"step archive entry {manifest_path} has an invalid schema"
            )
        if manifest.get("key_digest") != key:
            raise ValueError(
                f"step archive entry {manifest_path} has the wrong identity"
            )
        from .program import StepProgram

        program = StepProgram.from_json(program_path.read_text())
        if program.digest != manifest.get("step_program_digest"):
            raise ValueError(
                f"step archive entry {program_path} does not match its manifest"
            )
        self._record(key, program, program_path, "read")
        return program

    def write(
        self, key: str, program: StepProgram, identity: Mapping[str, object]
    ) -> None:
        """File `program` under `key`, beside the identity that produced it."""

        if not self.policy.write_enabled:
            return
        directory = self.path(key)
        program_path = directory / "step_program.json"
        manifest_path = directory / "manifest.json"
        if program_path.exists() and not self.policy.overwrite:
            from .program import StepProgram

            existing = StepProgram.from_json(program_path.read_text())
            if existing.digest != program.digest:
                raise ValueError(
                    "a fresh build differs from the step program filed under its"
                    f" identity; use a 'refresh' store mode or a new export"
                    f" bypass key: {program_path}"
                )
            self._record(key, program, program_path, "matched")
            return
        directory.mkdir(parents=True, exist_ok=True)
        atomic_text(program_path, program.to_json())
        atomic_json(
            manifest_path,
            {
                "schema": _SCHEMA,
                "key_digest": key,
                "step_program_digest": program.digest,
                "identity": dict(identity),
            },
        )
        self._record(key, program, program_path, "write")

    def _record(self, key: str, program: StepProgram, path: Path, access: str) -> None:
        if self.artifact_recorder is None:
            return
        self.artifact_recorder(
            category="build",
            kind="step",
            digest=key,
            path=path,
            access=access,
            schema=_SCHEMA,
            dependencies=(program.digest,),
        )


__all__ = ["StepArchive"]
