"""Persistent repository of recurrent optimizer captures.

Tracing an optimizer's update is the one capture that depends on nothing a
graph pair already keys: the optimizer's type and step code, the tensors it
binds (parameters, gradients, state and hyperparameters, by name and
geometry), its hyperparameter values, how its update is split across the
stages that own its parameters, and the framework versions. That identity is
known before the trace, so a step whose optimizer has been traced before, by
any caller in any process, is a lookup.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from shadowspill.errors import CaptureError
from shadowspill.planner.artifact_store import digest_directory
from shadowspill.pytorch.accelerator import provider_version
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.graph_pairs.serialization import (
    CachedGraphArtifact,
    atomic_json,
)
from shadowspill.schema import artifact_schema

from .artifacts import (
    OptimizerTensorBinding,
    optimizer_step_identity,
    optimizer_type_name,
    optimizer_value_identity,
)

if TYPE_CHECKING:
    from shadowspill.pytorch.profiling import PlanningArtifactRecorder

_OPTIMIZER_CAPTURE_SCHEMA = artifact_schema("optimizer_capture")


def recurrent_capture_identity(
    optimizer: torch.optim.Optimizer,
    bindings: tuple[OptimizerTensorBinding, ...],
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> str:
    """The digest of everything that determines a traced recurrent update.

    The bound tensors enter by name and geometry, never by device: the trace
    always runs on fake accelerator tensors, whatever the sandbox holds when
    the key is taken.
    """

    identity = {
        "kind": "recurrent_optimizer_capture",
        "schema": _OPTIMIZER_CAPTURE_SCHEMA,
        # What the entry holds. An entry written before this contract keys
        # differently, so it is unreachable rather than misread.
        "format": "traced_recurrent_graph/v1",
        "optimizer_type": optimizer_type_name(optimizer),
        "step": optimizer_step_identity(optimizer),
        "bindings": [
            {
                "name": binding.name,
                "role": binding.role.value,
                "mutable": binding.mutable,
                "spillable": binding.spillable,
                "shape": tuple(binding.tensor.shape),
                "stride": tuple(binding.tensor.stride()),
                "dtype": str(binding.tensor.dtype),
            }
            for binding in bindings
        ],
        "groups": [
            optimizer_value_identity(
                {key: value for key, value in group.items() if key != "params"}
            )
            for group in optimizer.param_groups
        ],
        "stage_owners": sorted(
            (name, list(owners))
            for name, owners in (parameter_stage_owners or {}).items()
        ),
        "torch": torch.__version__,
        "provider": provider_version(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredOptimizerCapture:
    """One traced recurrent update, free of the values it was traced over."""

    recurrent: CachedGraphArtifact

    @classmethod
    def capture(cls, artifact: GraphArtifact) -> StoredOptimizerCapture:
        return cls(CachedGraphArtifact.capture(artifact))

    def restore(
        self,
        bindings: tuple[OptimizerTensorBinding, ...],
        provenance: tuple[TaskInputProvenance, ...],
    ) -> GraphArtifact:
        """Return the traced update bound to this step's own tensors.

        Restoring synthesizes arguments of the recorded geometry, which must
        cost no memory: it happens in the mode the sandbox's tensors already
        live in, and the result is rebound to those tensors, exactly as a
        stage's graph pairs are rebound to each occurrence. The update
        mutates its parameters in place, which propagating values through it
        allows only where the trace ran, under no_grad.
        """

        mode = getattr(bindings[0].tensor, "fake_mode", None) if bindings else None
        with torch.no_grad():
            if mode is None:
                artifact = self.recurrent.restore()
            else:
                with mode:
                    artifact = self.recurrent.restore()
        return artifact.rebind_examples(
            tuple(binding.tensor for binding in bindings),
            input_provenance=provenance,
        )


class OptimizerCaptureStore:
    """Serve a traced recurrent update to every step that binds the same one."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: PlanningArtifactRecorder | None = None,
    ) -> None:
        self._root = None if root is None else Path(root).expanduser()
        self._read_enabled = read_enabled
        self._write_enabled = write_enabled
        self._overwrite = overwrite
        self._artifact_recorder = artifact_recorder
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path | None:
        if self._root is None:
            return None
        return digest_directory(self._root, key) / "optimizer_capture.pt"

    def read(self, key: str) -> StoredOptimizerCapture | None:
        """The stored capture under ``key``, or ``None`` when there is none."""

        path = self._path(key)
        if path is None or not self._read_enabled:
            self.misses += 1
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            self.misses += 1
            return None
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            raise CaptureError(
                f"optimizer capture store entry {path} is invalid"
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _OPTIMIZER_CAPTURE_SCHEMA
            or payload.get("key") != key
        ):
            raise CaptureError(
                f"optimizer capture store entry {path} has the wrong key"
            )
        stored = payload.get("capture")
        if not isinstance(stored, StoredOptimizerCapture):
            raise CaptureError(f"optimizer capture store entry {path} has invalid data")
        self.hits += 1
        self._record(key, path, "read", stored)
        return stored

    def write(
        self,
        key: str,
        artifact: GraphArtifact,
        *,
        optimizer_type: str,
    ) -> None:
        """Store a traced update under ``key``; an existing entry is kept."""

        path = self._path(key)
        if path is None or not self._write_enabled:
            return
        stored = StoredOptimizerCapture.capture(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self._overwrite:
            self._record(key, path, "matched", stored)
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                torch.save(
                    {
                        "schema": _OPTIMIZER_CAPTURE_SCHEMA,
                        "key": key,
                        "capture": stored,
                    },
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
        atomic_json(
            path.with_name("manifest.json"),
            {
                "schema": _OPTIMIZER_CAPTURE_SCHEMA,
                "key": key,
                "optimizer_type": optimizer_type,
                "recurrent_digest": artifact.compatibility_digest,
                "binding_count": len(artifact.tensor_inputs),
            },
        )
        self._record(key, path, "write", stored)

    def _record(
        self, key: str, path: Path, access: str, stored: StoredOptimizerCapture
    ) -> None:
        if self._artifact_recorder is None:
            return
        self._artifact_recorder(
            category="optimizers",
            kind="optimizer_capture",
            digest=key,
            path=path,
            access=access,
            schema=_OPTIMIZER_CAPTURE_SCHEMA,
            dependencies=(stored.recurrent.compatibility_digest,),
        )


__all__ = [
    "OptimizerCaptureStore",
    "StoredOptimizerCapture",
    "recurrent_capture_identity",
]
