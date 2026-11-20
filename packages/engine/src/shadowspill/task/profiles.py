"""Validated task measurements and output/input storage bindings."""

from __future__ import annotations

from dataclasses import dataclass

from .allocations.contract import TaskAllocationContract
from .allocations.events import TaskAllocationEvent, TaskAllocationOperation


@dataclass(frozen=True, slots=True)
class TaskOutputInputBinding:
    """Evidence that one output of a task is stored in one of its inputs.

    Parameters
    ----------
    output_leaf_index : int
        The output's position among the task's flattened outputs.
    input_position : int
        The position of the input whose storage holds it.
    output_offset_bytes : int
        Where the output starts in that storage, in bytes.

    Raises
    ------
    ValueError
        If a field is negative.
    """

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
        """Return the binding as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "output_leaf_index": self.output_leaf_index,
            "input_position": self.input_position,
            "output_offset_bytes": self.output_offset_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskOutputInputBinding:
        """Read a binding from its `to_dict` form.

        Parameters
        ----------
        value : dict
            The decoded JSON value.

        Returns
        -------
        TaskOutputInputBinding
            The binding.

        Raises
        ------
        ValueError
            If ``value`` is not a dict, lacks a field, or holds a negative one.
        """
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
    """Calibrated time and allocator behavior of one task.

    Profiling produces it, and the allocation evidence in it is what physical
    admission checks a task against.

    Parameters
    ----------
    runtime_ns : int
        The task's duration, in nanoseconds; the median sample when profiled.
    workspace_requested_bytes : int
        Peak bytes the task's temporary allocations requested at once.
    workspace_charged_bytes : int
        Peak bytes the allocator charged for them.
    workspace_extent_bytes : tuple of int
        Charged size of each temporary allocation live at that peak; empty when
        not recorded.
    samples_ns : tuple of int
        Each timed run, in nanoseconds. At least one.
    provenance : str
        What took the measurement. Not empty.
    allocation_trace : tuple of TaskAllocationEvent, default ()
        The task's allocations and frees, in order, without the frees of its
        outputs' storage.
    output_input_bindings : tuple of TaskOutputInputBinding, default ()
        The outputs stored in inputs, at most one binding per output.
    persistent_extent_bytes : tuple of int, default ()
        Sizes of the allocations this profile first observed outliving the
        task as bounded state, excluding task arguments and workspace. Profiles
        run one at a time, so a shared allocation is charged to the first.
    allocation_contract : TaskAllocationContract, optional
        The contract the trace satisfies: its allocations and temporary frees
        match the trace's.
    off_device_output_leaves : tuple of int, default ()
        Outputs observed off the execution device, such as a scalar the task
        returns to the caller's process. No device allocation accounts for
        them.

    Raises
    ------
    ValueError
        If a measurement is negative, there is no sample, ``provenance`` is
        empty, an off-device leaf repeats, or the allocation trace is
        inconsistent or disagrees with ``allocation_contract``.
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
    allocation_contract: TaskAllocationContract | None = None
    #: Output leaves this task was observed to produce off the execution
    #: device. A storage contract is traced, and a trace can claim device
    #: memory for a result an operator really returns as a host scalar. No
    #: device allocation accounts for such a leaf, so it is named here and
    #: the contract gives up its span rather than an allocation being
    #: looked for that was never made.
    off_device_output_leaves: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.runtime_ns,
            self.workspace_requested_bytes,
            self.workspace_charged_bytes,
            *self.workspace_extent_bytes,
            *self.samples_ns,
            *self.persistent_extent_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("profile measurements must be non-negative")
        if not self.samples_ns:
            raise ValueError("profile measurement requires at least one sample")
        if not self.provenance:
            raise ValueError("profile provenance must be non-empty")
        if any(index < 0 for index in self.off_device_output_leaves):
            raise ValueError("profile host output leaves must be non-negative")
        if len(set(self.off_device_output_leaves)) != len(
            self.off_device_output_leaves
        ):
            raise ValueError("profile names one host output leaf twice")
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
        roots: dict[int, int] = {}
        slot_sizes: dict[int, set[int]] = {}
        slot_alignments: dict[int, int] = {}
        output_slots: set[int] = set()
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
                    reused.add(event.reuses_ordinal)
                root = (
                    event.allocation_ordinal
                    if event.reuses_ordinal is None
                    else roots[event.reuses_ordinal]
                )
                roots[event.allocation_ordinal] = root
                slot_sizes.setdefault(root, set()).add(event.charged_bytes)
                if (
                    slot_alignments.setdefault(root, event.alignment_bytes)
                    != event.alignment_bytes
                ):
                    raise ValueError("task allocation slot changes alignment")
                if event.output_leaf_indices:
                    output_slots.add(root)
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
        if any(len(slot_sizes[root]) != 1 for root in output_slots):
            raise ValueError(
                "different-size task slot reuse is restricted to anonymous storage"
            )
        donated_leaves = [item.output_leaf_index for item in self.output_input_bindings]
        if len(set(donated_leaves)) != len(donated_leaves):
            raise ValueError("task output/input binding names one leaf twice")
        if output_leaves.intersection(donated_leaves):
            raise ValueError("task output has both allocated and donated storage")


__all__ = [
    "TaskMeasurement",
    "TaskOutputInputBinding",
]
