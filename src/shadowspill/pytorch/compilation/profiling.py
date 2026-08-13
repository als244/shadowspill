"""Content-addressed profiling over unique structural graph ABIs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from shadowspill.pytorch.compilation.representative import (
    REPRESENTATIVE_VALUE_POLICY,
    RepresentativeInputSummary,
)
from shadowspill.pytorch.contracts import CaptureError

# v10 records when a compiled output is physically served by a task input.
# The optimized Inductor storage contract must already describe that alias;
# profiling validates the contract and measures layout, workspace, and timing.
PROFILE_SCHEMA = "shadowspill.pytorch.profile/v12"


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


@dataclass(frozen=True, slots=True)
class TaskOutputInputBinding:
    """Physical evidence that one output is served by a task input."""

    output_leaf_index: int
    input_position: int
    output_offset_bytes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.output_leaf_index,
                self.input_position,
                self.output_offset_bytes,
            )
            < 0
        ):
            raise ValueError("task output/input binding fields must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "output_leaf_index": self.output_leaf_index,
            "input_position": self.input_position,
            "output_offset_bytes": self.output_offset_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskOutputInputBinding:
        if not isinstance(value, dict):
            raise ValueError("cached output/input binding must be an object")
        try:
            return cls(
                output_leaf_index=int(value["output_leaf_index"]),
                input_position=int(value["input_position"]),
                output_offset_bytes=int(value["output_offset_bytes"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "cached output/input binding has an invalid schema"
            ) from exc


class ProfilableArtifact(Protocol):
    """Minimal identity shared by compiled graphs and bounded eager tasks."""

    @property
    def compatibility_digest(self) -> str: ...


class PlanningArtifactRecorder(Protocol):
    """Minimal callback used to publish persistent-cache evidence."""

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


@dataclass(frozen=True, slots=True)
class ProfileEnvironment:
    """Implementation identity that can change executable task cost."""

    torch_version: str
    cuda_version: str | None
    device_name: str
    compute_capability: tuple[int, int]
    compiler_id: str
    provider_id: str
    implementation_revision: str | None = None

    def identity(self) -> dict[str, object]:
        return {
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "compiler_id": self.compiler_id,
            "provider_id": self.provider_id,
            "implementation_revision": self.implementation_revision,
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
    output_input_bindings: tuple[TaskOutputInputBinding, ...] = ()
    persistent_extent_bytes: tuple[int, ...] = ()
    profiling_wall_time_ns: int = 0
    representative_inputs: tuple[RepresentativeInputSummary, ...] = ()
    phase_timings_ns: tuple[tuple[str, int], ...] = ()
    timing_relative_mad: float = 0.0
    timing_half_drift: float = 0.0
    timing_unstable: bool = False

    def __post_init__(self) -> None:
        values = (
            self.runtime_ns,
            self.workspace_requested_bytes,
            self.workspace_charged_bytes,
            *self.workspace_extent_bytes,
            *self.samples_ns,
            *self.persistent_extent_bytes,
            self.profiling_wall_time_ns,
            *(duration for _name, duration in self.phase_timings_ns),
        )
        if any(value < 0 for value in values):
            raise ValueError("profile measurements must be non-negative")
        if not self.samples_ns:
            raise ValueError("profile measurement requires at least one sample")
        if not self.provenance:
            raise ValueError("profile provenance must be non-empty")
        if any(not name for name, _duration in self.phase_timings_ns):
            raise ValueError("profile phase names must be non-empty")
        if not math.isfinite(self.timing_relative_mad) or self.timing_relative_mad < 0:
            raise ValueError("profile relative MAD must be finite and non-negative")
        if not math.isfinite(self.timing_half_drift) or self.timing_half_drift < 0:
            raise ValueError("profile half drift must be finite and non-negative")
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
        donated_leaves = [item.output_leaf_index for item in self.output_input_bindings]
        if len(set(donated_leaves)) != len(donated_leaves):
            raise ValueError("task output/input binding names one leaf twice")
        if output_leaves.intersection(donated_leaves):
            raise ValueError("task output has both allocated and donated storage")

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_ns": self.runtime_ns,
            "workspace_requested_bytes": self.workspace_requested_bytes,
            "workspace_charged_bytes": self.workspace_charged_bytes,
            "workspace_extent_bytes": list(self.workspace_extent_bytes),
            "samples_ns": list(self.samples_ns),
            "provenance": self.provenance,
            "allocation_trace": [event.to_dict() for event in self.allocation_trace],
            "output_input_bindings": [
                item.to_dict() for item in self.output_input_bindings
            ],
            "persistent_extent_bytes": list(self.persistent_extent_bytes),
            "profiling_wall_time_ns": self.profiling_wall_time_ns,
            "representative_inputs": [
                item.to_dict() for item in self.representative_inputs
            ],
            "phase_timings_ns": [list(item) for item in self.phase_timings_ns],
            "timing_relative_mad": self.timing_relative_mad,
            "timing_half_drift": self.timing_half_drift,
            "timing_unstable": self.timing_unstable,
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
                output_input_bindings=tuple(
                    TaskOutputInputBinding.from_dict(item)
                    for item in value["output_input_bindings"]
                ),
                persistent_extent_bytes=tuple(
                    int(item) for item in value["persistent_extent_bytes"]
                ),
                profiling_wall_time_ns=int(value["profiling_wall_time_ns"]),
                representative_inputs=tuple(
                    RepresentativeInputSummary.from_dict(item)
                    for item in value["representative_inputs"]
                ),
                phase_timings_ns=tuple(
                    (str(item[0]), int(item[1])) for item in value["phase_timings_ns"]
                ),
                timing_relative_mad=float(value["timing_relative_mad"]),
                timing_half_drift=float(value["timing_half_drift"]),
                timing_unstable=bool(value["timing_unstable"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("cached task measurement has an invalid schema") from exc


@dataclass(frozen=True, slots=True)
class ProfileKey:
    graph_digest: str
    environment: ProfileEnvironment
    profiling_metadata_digest: str | None = None

    @property
    def digest(self) -> str:
        payload = {
            "schema": PROFILE_SCHEMA,
            "representative_value_policy": REPRESENTATIVE_VALUE_POLICY,
            "graph_digest": self.graph_digest,
            "environment": self.environment.identity(),
            "profiling_metadata_digest": self.profiling_metadata_digest,
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
    key_digests: tuple[str, ...] = ()
    profiling_metadata_digests: tuple[str | None, ...] = ()


class ProfileCache:
    """Atomic per-key JSON cache with no dependency on planning task identity."""

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
        configured = os.environ.get("SHADOWSPILL_PROFILE_CACHE")
        selected = root if root is not None else configured
        self.root = (
            Path(selected).expanduser()
            if selected is not None
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
        measurement = TaskMeasurement.from_dict(value.get("measurement"))
        self._record(key, path, "read")
        return measurement

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
        payload = {
            "schema": PROFILE_SCHEMA,
            "key_digest": key.digest,
            "measurement": measurement.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if path.exists() and not self.overwrite and not replace_invalid:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(f"profile cache entry {path} cannot be read") from exc
            if existing != encoded:
                raise ValueError(
                    "fresh profiling differs from an existing cache entry; "
                    "use overwrite_plan=True or a new implementation_revision: "
                    f"{path}"
                )
            self._record(key, path, "matched")
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{key.digest}.", suffix=".tmp", dir=path.parent
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
        self._record(key, path, "write")

    def _record(self, key: ProfileKey, path: Path, access: str) -> None:
        if self.artifact_recorder is None:
            return
        self.artifact_recorder(
            category="profiling",
            kind="task_measurement",
            digest=key.digest,
            path=path,
            access=access,
            schema=PROFILE_SCHEMA,
            dependencies=(key.graph_digest,),
        )


def profile_unique_artifacts(
    artifacts: Iterable[ProfilableArtifact],
    *,
    environment: ProfileEnvironment,
    measure: Callable[[ProfilableArtifact], TaskMeasurement],
    cache: ProfileCache,
    validate: Callable[[ProfilableArtifact, TaskMeasurement], None] | None = None,
    progress: Callable[[int, int, str, str], None] | None = None,
    profiling_metadata_digests: Sequence[str | None] | None = None,
) -> ProfilingResult:
    """Measure each structural key once and scatter it to every occurrence."""

    sequence = tuple(artifacts)
    metadata = (
        (None,) * len(sequence)
        if profiling_metadata_digests is None
        else tuple(profiling_metadata_digests)
    )
    if len(metadata) != len(sequence):
        raise ValueError("profiling metadata must align with task artifacts")
    by_key: dict[str, list[int]] = {}
    key_objects: dict[str, ProfileKey] = {}
    representatives: dict[str, ProfilableArtifact] = {}
    position_key_digests: list[str] = []
    for position, (artifact, metadata_digest) in enumerate(
        zip(sequence, metadata, strict=True)
    ):
        key = ProfileKey(
            artifact.compatibility_digest,
            environment,
            metadata_digest,
        )
        position_key_digests.append(key.digest)
        by_key.setdefault(key.digest, []).append(position)
        key_objects[key.digest] = key
        representatives.setdefault(key.digest, artifact)
    results: list[TaskMeasurement | None] = [None] * len(sequence)
    hits = 0
    misses = 0
    persistent_bytes_by_graph: dict[str, int] = {}
    ordered_digests = sorted(by_key)
    for index, digest in enumerate(ordered_digests, start=1):
        key = key_objects[digest]
        measurement = cache.read(key)
        replace_invalid = False
        if measurement is not None and validate is not None:
            try:
                validate(representatives[digest], measurement)
            except CaptureError:
                if progress is not None:
                    progress(index, len(ordered_digests), "cache-invalid", digest)
                measurement = None
                replace_invalid = True
        if measurement is None:
            if progress is not None:
                progress(index, len(ordered_digests), "measuring", digest)
            measurement = measure(representatives[digest])
            if validate is not None:
                validate(representatives[digest], measurement)
            cache.write(key, measurement, replace_invalid=replace_invalid)
            misses += 1
        else:
            if progress is not None:
                progress(index, len(ordered_digests), "cache-hit", digest)
            hits += 1
        persistent_bytes = sum(measurement.persistent_extent_bytes)
        persistent_bytes_by_graph[key.graph_digest] = max(
            persistent_bytes_by_graph.get(key.graph_digest, 0),
            persistent_bytes,
        )
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
        fixed_slab_bytes=sum(persistent_bytes_by_graph.values()),
        key_digests=tuple(position_key_digests),
        profiling_metadata_digests=metadata,
    )
