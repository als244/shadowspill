"""Immutable measurements and structural profile identities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from shadowspill.pytorch.profiling.allocation_abi import (
    TaskAllocationABI,
    TaskAllocationEvent,
    TaskAllocationOperation,
)
from shadowspill.pytorch.profiling.inputs import (
    REPRESENTATIVE_VALUE_POLICY,
    RepresentativeInputSummary,
)

# The opaque-optimizer artifact versions its own representative-gradient
# construction contract.  That targeted identity change avoids invalidating
# unrelated compiled-graph measurements.
PROFILE_SCHEMA = "shadowspill.pytorch.profile/v13"


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
    allocation_abi: TaskAllocationABI | None = None

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
        if self.allocation_abi is not None:
            trace_geometry = tuple(
                (
                    event.allocation_ordinal,
                    event.operation,
                    event.requested_bytes,
                    event.charged_bytes,
                    event.alignment_bytes,
                    event.output_leaf_indices,
                )
                for event in self.allocation_trace
                if event.operation is TaskAllocationOperation.ALLOCATE
            )
            abi_geometry = tuple(
                (
                    step.allocation_ordinal,
                    step.operation,
                    step.requested_bytes,
                    step.charged_bytes,
                    step.alignment_bytes,
                    step.output_leaf_indices,
                )
                for step in self.allocation_abi.steps
                if step.operation is TaskAllocationOperation.ALLOCATE
            )
            if trace_geometry != abi_geometry:
                raise ValueError(
                    "task allocation ABI allocations disagree with its trace"
                )
            output_ordinals = {
                event.allocation_ordinal
                for event in self.allocation_trace
                if event.operation is TaskAllocationOperation.ALLOCATE
                and event.output_leaf_indices
            }
            trace_frees = tuple(
                event.allocation_ordinal
                for event in self.allocation_trace
                if event.operation is TaskAllocationOperation.FREE
            )
            abi_frees = tuple(
                step.allocation_ordinal
                for step in self.allocation_abi.steps
                if step.operation is TaskAllocationOperation.FREE
                and step.allocation_ordinal not in output_ordinals
            )
            if trace_frees != abi_frees:
                raise ValueError(
                    "task allocation ABI temporary frees disagree with its trace"
                )

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
            "allocation_abi": (
                None if self.allocation_abi is None else self.allocation_abi.to_dict()
            ),
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
                allocation_abi=(
                    None
                    if value.get("allocation_abi") is None
                    else TaskAllocationABI.from_dict(value["allocation_abi"])
                ),
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


__all__ = [
    "PROFILE_SCHEMA",
    "ProfileEnvironment",
    "ProfileKey",
    "ProfilingResult",
    "TaskAllocationABI",
    "TaskAllocationEvent",
    "TaskAllocationOperation",
    "TaskMeasurement",
    "TaskOutputInputBinding",
]
