"""Content-addressed profiling over unique structural graph ABIs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

# v9 records each returned view's byte offset inside its compiled allocation.
# Semantic storage identity remains part of GraphArtifact; this observation is
# used only for physical layout, workspace replay, and admission.
PROFILE_SCHEMA = "shadowspill.pytorch.profile/v9"


class TaskAllocationOperation(StrEnum):
    """One physical transition in a profiled task's allocator trace."""

    ALLOCATE = "allocate"
    FREE = "free"


@dataclass(frozen=True, slots=True)
class TaskAllocationEvent:
    """Task-local allocation event with stable identities and output leaves."""

    allocation_ordinal: int
    operation: TaskAllocationOperation
    requested_bytes: int
    charged_bytes: int
    output_leaf_indices: tuple[int, ...] = ()
    output_view_offsets: tuple[int, ...] = ()
    reuses_ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.allocation_ordinal < 0:
            raise ValueError("task allocation ordinal must be non-negative")
        if not isinstance(self.operation, TaskAllocationOperation):
            raise TypeError("task allocation operation has an invalid type")
        if self.requested_bytes < 0 or self.charged_bytes <= 0:
            raise ValueError("task allocation sizes are invalid")
        if any(index < 0 for index in self.output_leaf_indices):
            raise ValueError("task output leaf indices must be non-negative")
        if len(set(self.output_leaf_indices)) != len(self.output_leaf_indices):
            raise ValueError("task output leaf indices must be unique")
        if len(self.output_view_offsets) != len(self.output_leaf_indices):
            raise ValueError("task output leaves and view offsets must align")
        if any(offset < 0 for offset in self.output_view_offsets):
            raise ValueError("task output view offsets must be non-negative")
        if self.reuses_ordinal is not None:
            if self.operation is not TaskAllocationOperation.ALLOCATE:
                raise ValueError("only an allocation may reuse a retired extent")
            if self.reuses_ordinal < 0:
                raise ValueError("reused task allocation ordinal must be non-negative")
            if self.reuses_ordinal == self.allocation_ordinal:
                raise ValueError("task allocation cannot reuse itself")

    def to_dict(self) -> dict[str, object]:
        return {
            "allocation_ordinal": self.allocation_ordinal,
            "operation": self.operation.value,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
            "output_leaf_indices": list(self.output_leaf_indices),
            "output_view_offsets": list(self.output_view_offsets),
            "reuses_ordinal": self.reuses_ordinal,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationEvent:
        if not isinstance(value, dict):
            raise ValueError("cached allocation event must be an object")
        try:
            return cls(
                allocation_ordinal=int(value["allocation_ordinal"]),
                operation=TaskAllocationOperation(str(value["operation"])),
                requested_bytes=int(value["requested_bytes"]),
                charged_bytes=int(value["charged_bytes"]),
                output_leaf_indices=tuple(
                    int(item) for item in value["output_leaf_indices"]
                ),
                output_view_offsets=tuple(
                    int(item) for item in value["output_view_offsets"]
                ),
                reuses_ordinal=(
                    None
                    if value["reuses_ordinal"] is None
                    else int(value["reuses_ordinal"])
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("cached allocation event has an invalid schema") from exc


class ProfilableArtifact(Protocol):
    """Minimal identity shared by compiled graphs and bounded eager tasks."""

    @property
    def compatibility_digest(self) -> str: ...


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
    """Calibrated task time and allocator behavior for one structural ABI.

    ``persistent_extent_bytes`` is a conservative slab reserve for bounded
    provider or custom-operation state discovered by repeated-task auditing.
    It is not part of the task-local workspace timeline.
    """

    runtime_ns: int
    workspace_requested_bytes: int
    workspace_charged_bytes: int
    workspace_extent_bytes: tuple[int, ...]
    samples_ns: tuple[int, ...]
    provenance: str
    allocation_trace: tuple[TaskAllocationEvent, ...] = ()
    persistent_extent_bytes: tuple[int, ...] = ()
    profiling_wall_time_ns: int = 0

    def __post_init__(self) -> None:
        values = (
            self.runtime_ns,
            self.workspace_requested_bytes,
            self.workspace_charged_bytes,
            *self.workspace_extent_bytes,
            *self.samples_ns,
            *self.persistent_extent_bytes,
            self.profiling_wall_time_ns,
        )
        if any(value < 0 for value in values):
            raise ValueError("profile measurements must be non-negative")
        if not self.samples_ns:
            raise ValueError("profile measurement requires at least one sample")
        if not self.provenance:
            raise ValueError("profile provenance must be non-empty")
        self._validate_allocation_trace()

    def _validate_allocation_trace(self) -> None:
        live: dict[int, tuple[int, int]] = {}
        retired: dict[int, tuple[int, int]] = {}
        reused: set[int] = set()
        output_leaves: set[int] = set()
        for event in self.allocation_trace:
            if event.operation is TaskAllocationOperation.ALLOCATE:
                if event.allocation_ordinal in live:
                    raise ValueError(
                        "task allocation trace creates an allocation twice"
                    )
                if output_leaves.intersection(event.output_leaf_indices):
                    raise ValueError("task allocation trace binds an output leaf twice")
                if event.reuses_ordinal is not None:
                    sizes = retired.get(event.reuses_ordinal)
                    if sizes is None or event.reuses_ordinal in reused:
                        raise ValueError(
                            "task allocation trace reuses an unavailable extent"
                        )
                    if sizes[1] != event.charged_bytes:
                        raise ValueError(
                            "task allocation trace changes a reused physical extent"
                        )
                    reused.add(event.reuses_ordinal)
                live[event.allocation_ordinal] = (
                    event.requested_bytes,
                    event.charged_bytes,
                )
                output_leaves.update(event.output_leaf_indices)
                continue
            sizes = live.pop(event.allocation_ordinal, None)
            if sizes is None:
                raise ValueError("task allocation trace frees an unknown allocation")
            if sizes != (event.requested_bytes, event.charged_bytes):
                raise ValueError(
                    "task allocation trace changes allocation size on free"
                )
            retired[event.allocation_ordinal] = sizes

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_ns": self.runtime_ns,
            "workspace_requested_bytes": self.workspace_requested_bytes,
            "workspace_charged_bytes": self.workspace_charged_bytes,
            "workspace_extent_bytes": list(self.workspace_extent_bytes),
            "samples_ns": list(self.samples_ns),
            "provenance": self.provenance,
            "allocation_trace": [event.to_dict() for event in self.allocation_trace],
            "persistent_extent_bytes": list(self.persistent_extent_bytes),
            "profiling_wall_time_ns": self.profiling_wall_time_ns,
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
                allocation_trace=tuple(
                    TaskAllocationEvent.from_dict(item)
                    for item in value["allocation_trace"]
                ),
                persistent_extent_bytes=tuple(
                    int(item) for item in value["persistent_extent_bytes"]
                ),
                profiling_wall_time_ns=int(value["profiling_wall_time_ns"]),
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
    fixed_slab_bytes: int


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
    artifacts: Iterable[ProfilableArtifact],
    *,
    environment: ProfileEnvironment,
    measure: Callable[[ProfilableArtifact], TaskMeasurement],
    cache: ProfileCache,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> ProfilingResult:
    """Measure each structural key once and scatter it to every occurrence."""

    sequence = tuple(artifacts)
    by_key: dict[str, list[int]] = {}
    key_objects: dict[str, ProfileKey] = {}
    representatives: dict[str, ProfilableArtifact] = {}
    for position, artifact in enumerate(sequence):
        key = ProfileKey(artifact.compatibility_digest, environment)
        by_key.setdefault(key.digest, []).append(position)
        key_objects[key.digest] = key
        representatives.setdefault(key.digest, artifact)
    results: list[TaskMeasurement | None] = [None] * len(sequence)
    hits = 0
    misses = 0
    fixed_slab_bytes = 0
    ordered_digests = sorted(by_key)
    for index, digest in enumerate(ordered_digests, start=1):
        key = key_objects[digest]
        measurement = cache.read(key)
        if measurement is None:
            if progress is not None:
                progress(index, len(ordered_digests), "measuring", digest)
            measurement = measure(representatives[digest])
            cache.write(key, measurement)
            misses += 1
        else:
            if progress is not None:
                progress(index, len(ordered_digests), "cache-hit", digest)
            hits += 1
        fixed_slab_bytes += sum(measurement.persistent_extent_bytes)
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
        fixed_slab_bytes=fixed_slab_bytes,
    )
