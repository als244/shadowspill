"""Validated task measurements and output/input storage bindings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shadowspill.ir.program.timing import (
    TaskOverlapCost,
    TaskResourcePhase,
    validate_task_timing,
)

from .allocations.contract import TaskAllocationContract
from .allocations.events import TaskAllocationEvent, TaskAllocationOperation
from .inputs import (
    RepresentativeInputSummary,
)
from .profiling import ProfilingOptions, TimingWindow


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
class TaskMeasurement:
    """Calibrated task time and allocator behavior for one structural contract.

    ``persistent_extent_bytes`` is the incremental slab growth attributable to
    bounded provider or custom-operation state first observed by this profile.
    It excludes profiler-owned task arguments and task-local workspace.  Since
    profiles run serially against one live provider inventory, a shared cache
    allocation is charged only by the first structural contract that creates it.
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
    profiling_options: ProfilingOptions = field(default_factory=ProfilingOptions)
    conditioning: TimingWindow = field(default_factory=TimingWindow)
    sampling: TimingWindow = field(default_factory=TimingWindow)
    allocation_contract: TaskAllocationContract | None = None
    #: Output leaves this task was observed to produce off the execution
    #: device. A storage contract is traced, and a trace can claim device
    #: memory for a result an operator really returns as a host scalar. No
    #: device allocation accounts for such a leaf, so it is named here and
    #: the contract gives up its span rather than an allocation being
    #: looked for that was never made.
    off_device_output_leaves: tuple[int, ...] = ()
    resource_phases: tuple[TaskResourcePhase, ...] = ()
    overlap_costs: tuple[TaskOverlapCost, ...] = ()

    def __post_init__(self) -> None:
        validate_task_timing(self.runtime_ns, self.resource_phases, self.overlap_costs)
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
        if any(index < 0 for index in self.off_device_output_leaves):
            raise ValueError("profile host output leaves must be non-negative")
        if len(set(self.off_device_output_leaves)) != len(
            self.off_device_output_leaves
        ):
            raise ValueError("profile names one host output leaf twice")
        if not math.isfinite(self.timing_relative_mad) or self.timing_relative_mad < 0:
            raise ValueError("profile relative MAD must be finite and non-negative")
        if not math.isfinite(self.timing_half_drift) or self.timing_half_drift < 0:
            raise ValueError("profile half drift must be finite and non-negative")
        self._validate_allocation_trace()
        if self.allocation_contract is not None:
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
            contract_geometry = tuple(
                (
                    step.allocation_ordinal,
                    step.operation,
                    step.requested_bytes,
                    step.charged_bytes,
                    step.alignment_bytes,
                    step.output_leaf_indices,
                )
                for step in self.allocation_contract.steps
                if step.operation is TaskAllocationOperation.ALLOCATE
                and not (step.persistent_after_task and not step.output_leaf_indices)
            )
            if trace_geometry != contract_geometry:
                raise ValueError(
                    "task allocation contract allocations disagree with its trace"
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
            contract_frees = tuple(
                step.allocation_ordinal
                for step in self.allocation_contract.steps
                if step.operation is TaskAllocationOperation.FREE
                and step.allocation_ordinal not in output_ordinals
            )
            if trace_frees != contract_frees:
                raise ValueError(
                    "task allocation contract temporary frees disagree with its trace"
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


__all__ = [
    "TaskMeasurement",
    "TaskOutputInputBinding",
]
