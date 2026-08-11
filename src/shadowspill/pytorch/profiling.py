"""Content-addressed profiling over unique structural graph ABIs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .capture import GraphArtifact

PROFILE_SCHEMA = "shadowspill.pytorch.profile/v1"


@dataclass(frozen=True, slots=True)
class ProfileEnvironment:
    """Implementation identity that can change executable task cost."""

    torch_version: str
    cuda_version: str | None
    device_name: str
    compute_capability: tuple[int, int]
    compiler_id: str
    provider_id: str

    def identity(self) -> dict[str, object]:
        return {
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "compiler_id": self.compiler_id,
            "provider_id": self.provider_id,
        }


@dataclass(frozen=True, slots=True)
class TaskMeasurement:
    """Calibrated task time and exact anonymous workspace live set."""

    runtime_ns: int
    workspace_requested_bytes: int
    workspace_charged_bytes: int
    workspace_extent_bytes: tuple[int, ...]
    samples_ns: tuple[int, ...]
    provenance: str

    def __post_init__(self) -> None:
        values = (
            self.runtime_ns,
            self.workspace_requested_bytes,
            self.workspace_charged_bytes,
            *self.workspace_extent_bytes,
            *self.samples_ns,
        )
        if any(value < 0 for value in values):
            raise ValueError("profile measurements must be non-negative")
        if not self.samples_ns:
            raise ValueError("profile measurement requires at least one sample")
        if not self.provenance:
            raise ValueError("profile provenance must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_ns": self.runtime_ns,
            "workspace_requested_bytes": self.workspace_requested_bytes,
            "workspace_charged_bytes": self.workspace_charged_bytes,
            "workspace_extent_bytes": list(self.workspace_extent_bytes),
            "samples_ns": list(self.samples_ns),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskMeasurement:
        if not isinstance(value, dict):
            raise ValueError("cached task measurement must be an object")
        try:
            return cls(
                runtime_ns=int(value["runtime_ns"]),
                workspace_requested_bytes=int(value["workspace_requested_bytes"]),
                workspace_charged_bytes=int(value["workspace_charged_bytes"]),
                workspace_extent_bytes=tuple(
                    int(item) for item in value["workspace_extent_bytes"]
                ),
                samples_ns=tuple(int(item) for item in value["samples_ns"]),
                provenance=str(value["provenance"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("cached task measurement has an invalid schema") from exc


@dataclass(frozen=True, slots=True)
class ProfileKey:
    graph_digest: str
    environment: ProfileEnvironment

    @property
    def digest(self) -> str:
        payload = {
            "schema": PROFILE_SCHEMA,
            "graph_digest": self.graph_digest,
            "environment": self.environment.identity(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ProfilingResult:
    """Measurements scattered back to artifact positions with cache evidence."""

    measurements: tuple[TaskMeasurement, ...]
    unique_keys: int
    cache_hits: int
    cache_misses: int


class ProfileCache:
    """Atomic per-key JSON cache with no dependency on planning task identity."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = os.environ.get("SHADOWSPILL_PROFILE_CACHE")
        selected = root if root is not None else configured
        self.root = (
            Path(selected).expanduser()
            if selected is not None
            else Path.home() / ".cache" / "shadowspill" / "profiles"
        )

    def read(self, key: ProfileKey) -> TaskMeasurement | None:
        path = self.root / f"{key.digest}.json"
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"profile cache entry {path} cannot be read") from exc
        if not isinstance(value, dict) or value.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"profile cache entry {path} has an invalid schema")
        if value.get("key_digest") != key.digest:
            raise ValueError(f"profile cache entry {path} has the wrong identity")
        return TaskMeasurement.from_dict(value.get("measurement"))

    def write(self, key: ProfileKey, measurement: TaskMeasurement) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": PROFILE_SCHEMA,
            "key_digest": key.digest,
            "measurement": measurement.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{key.digest}.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.root / f"{key.digest}.json")
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)


def profile_unique_artifacts(
    artifacts: Iterable[GraphArtifact],
    *,
    environment: ProfileEnvironment,
    measure: Callable[[GraphArtifact], TaskMeasurement],
    cache: ProfileCache,
) -> ProfilingResult:
    """Measure each structural key once and scatter it to every occurrence."""

    sequence = tuple(artifacts)
    by_key: dict[str, list[int]] = {}
    key_objects: dict[str, ProfileKey] = {}
    representatives: dict[str, GraphArtifact] = {}
    for position, artifact in enumerate(sequence):
        key = ProfileKey(artifact.compatibility_digest, environment)
        by_key.setdefault(key.digest, []).append(position)
        key_objects[key.digest] = key
        representatives.setdefault(key.digest, artifact)
    results: list[TaskMeasurement | None] = [None] * len(sequence)
    hits = 0
    misses = 0
    for digest in sorted(by_key):
        key = key_objects[digest]
        measurement = cache.read(key)
        if measurement is None:
            measurement = measure(representatives[digest])
            cache.write(key, measurement)
            misses += 1
        else:
            hits += 1
        for position in by_key[digest]:
            results[position] = measurement
    if any(measurement is None for measurement in results):
        raise AssertionError("profiling result scatter is incomplete")
    return ProfilingResult(
        measurements=tuple(
            measurement for measurement in results if measurement is not None
        ),
        unique_keys=len(by_key),
        cache_hits=hits,
        cache_misses=misses,
    )
