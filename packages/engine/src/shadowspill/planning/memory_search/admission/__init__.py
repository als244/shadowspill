"""Schedule-invariant physical facts consumed by admission-aware planning.

The logical :class:`~shadowspill.ir.ShadowSpillProgram` deliberately does not encode how
a compiled task returns storage.  This module carries the small additional
physical contract needed to evaluate dynamic slab admission without importing
the framework or consulting runtime allocator telemetry in the candidate loop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from shadowspill.ir import ShadowSpillProgram
from shadowspill.planning.memory_search.strict import (
    _integer,
    _integer_tuple,
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _string,
    _string_tuple,
)
from shadowspill.schema import artifact_schema
from shadowspill.serialization.sharing import read_record, reading_shared, shared_writer

if TYPE_CHECKING:
    from shadowspill.planning.memory_search.admission.task_columns import TaskColumns

_SCHEMA = artifact_schema("admission_facts")


class TaskAllocationStepKind(StrEnum):
    """Whether a task's allocation step allocates storage or releases it.

    Attributes
    ----------
    ALLOCATE : str
        ``"allocate"``: the task allocates storage.
    RELEASE : str
        ``"release"``: the task frees storage it allocated.
    """

    ALLOCATE = "allocate"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class TaskAllocationStep:
    """One allocation or release a task makes while it runs, as profiled.

    A `TaskAdmissionSpec` lists a task's steps in the order the task made
    them. They are evidence for admission, which places them; they name no
    address, and the runtime's allocations need not follow their order.

    Parameters
    ----------
    allocation_ordinal : int
        The allocation's identity within the task. At least zero. A release
        names the ordinal of the allocation it frees.
    kind : TaskAllocationStepKind
        Whether the step allocates or releases.
    charged_bytes : int, default 0
        Bytes the allocation is charged: positive for an allocation, zero
        for a release.
    output_alias_group_id : str, optional
        The alias group the allocation becomes when it outlives the task as
        a program object. None for workspace, and for a release.
    reuses_allocation_ordinal : int, optional
        An earlier, released allocation of the same task whose storage this
        one reuses. None for new storage, and for a release.

    Raises
    ------
    ValueError
        If an ordinal is negative or reuses itself, an allocation's bytes are
        not positive, an output alias is empty, or a release carries an
        allocation's fields.
    TypeError
        If ``kind`` is not a `TaskAllocationStepKind`.
    """

    allocation_ordinal: int
    kind: TaskAllocationStepKind
    charged_bytes: int = 0
    output_alias_group_id: str | None = None
    reuses_allocation_ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.allocation_ordinal < 0:
            raise ValueError("task allocation ordinal must be non-negative")
        if not isinstance(self.kind, TaskAllocationStepKind):
            raise TypeError("task allocation step kind is invalid")
        if self.kind is TaskAllocationStepKind.ALLOCATE:
            if self.charged_bytes <= 0:
                raise ValueError("task allocation bytes must be positive")
            if self.output_alias_group_id == "":
                raise ValueError("task allocation output alias must be non-empty")
            if self.reuses_allocation_ordinal is not None and (
                self.reuses_allocation_ordinal < 0
                or self.reuses_allocation_ordinal == self.allocation_ordinal
            ):
                raise ValueError("task allocation reuse ordinal is invalid")
        elif (
            self.charged_bytes != 0
            or self.output_alias_group_id is not None
            or self.reuses_allocation_ordinal is not None
        ):
            raise ValueError("task release carries allocation-only fields")

    @shared_writer
    def to_dict(self) -> dict[str, object]:
        """Write the step as a JSON-compatible mapping.

        Returns
        -------
        dict
            Every field by name, ``kind`` as its value.
        """
        return {
            "allocation_ordinal": self.allocation_ordinal,
            "charged_bytes": self.charged_bytes,
            "kind": self.kind.value,
            "output_alias_group_id": self.output_alias_group_id,
            "reuses_allocation_ordinal": self.reuses_allocation_ordinal,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAllocationStep:
        """Read a step back from what `TaskAllocationStep.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        TaskAllocationStep
            The step.

        Raises
        ------
        ValueError
            If a field is missing or malformed.
        """
        data = _mapping(value, path)
        return cls(
            allocation_ordinal=_integer(
                data.get("allocation_ordinal"), f"{path}.allocation_ordinal"
            ),
            kind=_enum(
                TaskAllocationStepKind,
                data.get("kind"),
                f"{path}.kind",
            ),
            charged_bytes=_integer(data.get("charged_bytes"), f"{path}.charged_bytes"),
            output_alias_group_id=_optional_string(
                data.get("output_alias_group_id"),
                f"{path}.output_alias_group_id",
            ),
            reuses_allocation_ordinal=_optional_integer(
                data.get("reuses_allocation_ordinal"),
                f"{path}.reuses_allocation_ordinal",
            ),
        )


class _AllocationSteps:
    """Share repeated immutable allocator transitions within one decoded artifact.

    Large models repeat the same temporary allocation/release trace for many
    tasks. Decode and validate each distinct transition once, retaining a bounded
    table only for this load. Include scalar types so ``True`` cannot reuse a
    previously validated integer, and send malformed values to the strict parser.
    """

    def __init__(self) -> None:
        self.values: dict[tuple[tuple[type, Any], ...], TaskAllocationStep] = {}

    def read(self, value: object, path: str) -> TaskAllocationStep:
        if reading_shared(value):
            return read_record(TaskAllocationStep.from_value, value, path)
        data = _mapping(value, path)
        values = tuple(
            data.get(name)
            for name in (
                "allocation_ordinal",
                "kind",
                "charged_bytes",
                "output_alias_group_id",
                "reuses_allocation_ordinal",
            )
        )
        if any(type(item) not in (int, str, type(None)) for item in values):
            return TaskAllocationStep.from_value(value, path)
        key = tuple((type(item), item) for item in values)
        previous = self.values.get(key)
        if previous is not None:
            return previous
        result = TaskAllocationStep.from_value(value, path)
        if len(self.values) < 16384:
            self.values[key] = result
        return result


@dataclass(frozen=True, slots=True)
class StorageHandoff:
    """A live lease a task passes from one alias group to another, uncopied.

    Parameters
    ----------
    source_alias_group_id : str
        The alias group that gives up the lease.
    destination_alias_group_id : str
        The alias group that takes it over. Differs from the source.

    Raises
    ------
    ValueError
        If an identifier is empty, or the two are the same.
    """

    source_alias_group_id: str
    destination_alias_group_id: str

    def __post_init__(self) -> None:
        if not self.source_alias_group_id or not self.destination_alias_group_id:
            raise ValueError("storage-handoff alias IDs must be non-empty")
        if self.source_alias_group_id == self.destination_alias_group_id:
            raise ValueError("storage-handoff source and destination must differ")

    @shared_writer
    def to_dict(self) -> dict[str, str]:
        """Write the handoff as a JSON-compatible mapping.

        Returns
        -------
        dict
            Both alias group identifiers, by field name.
        """
        return {
            "destination_alias_group_id": self.destination_alias_group_id,
            "source_alias_group_id": self.source_alias_group_id,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> StorageHandoff:
        """Read a handoff back from what `StorageHandoff.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        StorageHandoff
            The handoff.

        Raises
        ------
        ValueError
            If a field is missing or malformed.
        """
        data = _mapping(value, path)
        return cls(
            source_alias_group_id=_string(
                data.get("source_alias_group_id"),
                f"{path}.source_alias_group_id",
            ),
            destination_alias_group_id=_string(
                data.get("destination_alias_group_id"),
                f"{path}.destination_alias_group_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskAdmissionSpec:
    """What one task allocates, produces and hands over, for admission.

    Parameters
    ----------
    task_id : str
        The task. Non-empty.
    workspace_extents : tuple of int, default ()
        Byte sizes of the task's anonymous allocations that are live
        together; their sum is ``workspace_bytes``. Each positive.
    fresh_output_aliases : tuple of str, default ()
        Alias groups the task produces into storage it allocates, each once.
    replacement_aliases : tuple of str, default ()
        Alias groups the task mutates by allocating their replacement, each
        once.
    storage_handoffs : tuple of StorageHandoff, default ()
        Leases the task passes between alias groups, each source and each
        destination at most once.
    allocation_steps : tuple of TaskAllocationStep, default ()
        Every allocation and release the task makes, in order. Required when
        the task has workspace, fresh outputs or replacements. Every
        allocation not bound to an output is released, and the outputs bound
        are exactly the fresh outputs and replacements.

    Raises
    ------
    ValueError
        If an identifier is empty or repeated, an extent is not positive, fresh
        outputs, replacements and handoff destinations overlap, or the
        allocation steps contradict each other or the outputs.
    """

    task_id: str
    workspace_extents: tuple[int, ...] = ()
    fresh_output_aliases: tuple[str, ...] = ()
    replacement_aliases: tuple[str, ...] = ()
    storage_handoffs: tuple[StorageHandoff, ...] = ()
    allocation_steps: tuple[TaskAllocationStep, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task admission ID must be non-empty")
        if any(value <= 0 for value in self.workspace_extents):
            raise ValueError("task admission workspace extents must be positive")
        for field, values in (
            ("fresh_output_aliases", self.fresh_output_aliases),
            ("replacement_aliases", self.replacement_aliases),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"task admission {field} must be unique")
            if any(not value for value in values):
                raise ValueError(f"task admission {field} must be non-empty")
        destinations = tuple(
            item.destination_alias_group_id for item in self.storage_handoffs
        )
        sources = tuple(item.source_alias_group_id for item in self.storage_handoffs)
        if len(destinations) != len(set(destinations)):
            raise ValueError("task storage-handoff destinations must be unique")
        if len(sources) != len(set(sources)):
            raise ValueError("task storage-handoff sources must be unique")
        output_kinds = (
            set(self.fresh_output_aliases),
            set(self.replacement_aliases),
            set(destinations),
        )
        if any(
            left & right
            for index, left in enumerate(output_kinds)
            for right in output_kinds[index + 1 :]
        ):
            raise ValueError(
                "fresh, replacement, and handoff destination aliases must be disjoint"
            )
        self._validate_allocation_steps()

    def _validate_allocation_steps(self) -> None:
        if not self.allocation_steps:
            if (
                self.workspace_extents
                or self.fresh_output_aliases
                or self.replacement_aliases
            ):
                raise ValueError(
                    f"task {self.task_id} physical admission requires explicit "
                    "allocation steps for workspace and allocated outputs"
                )
            return
        live: set[int] = set()
        allocations: dict[int, TaskAllocationStep] = {}
        retired: set[int] = set()
        reused: set[int] = set()
        output_aliases: set[str] = set()
        slot_roots: dict[int, int] = {}
        slot_sizes: dict[int, set[int]] = {}
        output_slots: set[int] = set()
        for step in self.allocation_steps:
            ordinal = step.allocation_ordinal
            if step.kind is TaskAllocationStepKind.ALLOCATE:
                if ordinal in allocations:
                    raise ValueError(
                        "task allocation trace allocates one ordinal twice"
                    )
                source = step.reuses_allocation_ordinal
                if source is not None:
                    if source not in retired:
                        raise ValueError(
                            "task allocation trace reuses a non-retired ordinal"
                        )
                    if source in reused:
                        raise ValueError(
                            "task allocation trace reuses one ordinal twice"
                        )
                    reused.add(source)
                root = ordinal if source is None else slot_roots[source]
                slot_roots[ordinal] = root
                slot_sizes.setdefault(root, set()).add(step.charged_bytes)
                allocations[ordinal] = step
                live.add(ordinal)
                if step.output_alias_group_id is not None:
                    output_slots.add(root)
                    if step.output_alias_group_id in output_aliases:
                        raise ValueError(
                            "task allocation trace binds one output alias twice"
                        )
                    output_aliases.add(step.output_alias_group_id)
                continue
            if ordinal not in live:
                raise ValueError("task allocation trace releases an unknown ordinal")
            live.remove(ordinal)
            retired.add(ordinal)
        if any(len(slot_sizes[root]) != 1 for root in output_slots):
            raise ValueError(
                "different-size task slot reuse is restricted to anonymous storage"
            )
        expected_outputs = set(self.fresh_output_aliases) | set(
            self.replacement_aliases
        )
        if output_aliases != expected_outputs:
            raise ValueError(
                "task allocation trace output aliases disagree with task ownership: "
                f"trace={sorted(output_aliases)!r}, "
                f"expected={sorted(expected_outputs)!r}"
            )
        persistent_ordinals = {
            step.allocation_ordinal
            for step in self.allocation_steps
            if step.output_alias_group_id is not None
        }
        if live != persistent_ordinals:
            raise ValueError(
                "task allocation trace must release every anonymous allocation"
            )

    @property
    def workspace_bytes(self) -> int:
        """Bytes of anonymous workspace live together: the sum of the extents."""

        return sum(self.workspace_extents)

    @shared_writer
    def to_dict(self) -> dict[str, object]:
        """Write the task's facts as a JSON-compatible mapping.

        Returns
        -------
        dict
            Every field by name, handoffs and steps as their own mappings.
        """
        return {
            "fresh_output_aliases": list(self.fresh_output_aliases),
            "replacement_aliases": list(self.replacement_aliases),
            "storage_handoffs": [item.to_dict() for item in self.storage_handoffs],
            "task_id": self.task_id,
            "workspace_extents": list(self.workspace_extents),
            "allocation_steps": [item.to_dict() for item in self.allocation_steps],
        }

    @classmethod
    def from_value(
        cls, value: object, path: str, *, _steps: _AllocationSteps | None = None
    ) -> TaskAdmissionSpec:
        """Read a task's facts back from what `TaskAdmissionSpec.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        TaskAdmissionSpec
            The task's facts.

        Raises
        ------
        ValueError
            If a field is missing or malformed.
        """
        data = _mapping(value, path)
        read_step = TaskAllocationStep.from_value if _steps is None else _steps.read
        return cls(
            task_id=_string(data.get("task_id"), f"{path}.task_id"),
            workspace_extents=_integer_tuple(
                data.get("workspace_extents"), f"{path}.workspace_extents"
            ),
            fresh_output_aliases=_string_tuple(
                data.get("fresh_output_aliases"),
                f"{path}.fresh_output_aliases",
            ),
            replacement_aliases=_string_tuple(
                data.get("replacement_aliases"),
                f"{path}.replacement_aliases",
            ),
            storage_handoffs=tuple(
                StorageHandoff.from_value(item, f"{path}.storage_handoffs[{index}]")
                for index, item in enumerate(
                    _list(data.get("storage_handoffs"), f"{path}.storage_handoffs")
                )
            ),
            allocation_steps=tuple(
                read_step(item, f"{path}.allocation_steps[{index}]")
                for index, item in enumerate(
                    _list(data.get("allocation_steps"), f"{path}.allocation_steps")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AdmissionFacts:
    """The physical facts every candidate schedule is admitted against.

    They describe one device's execution pool and how each task allocates
    within it, and do not depend on the schedule. A search measures its
    candidates against them, and admission places a plan's leases with them.

    Parameters
    ----------
    device_id : str
        The device whose execution pool this describes. Non-empty.
    pool_capacity_bytes : int
        The execution pool's capacity, in bytes. Positive.
    object_capacity_bytes : int
        The capacity, in bytes, a search plans object residency against
        before exact task and transfer deltas are evaluated. Positive and at
        most ``pool_capacity_bytes``.
    minimum_alignment : int
        The alignment, in bytes, every offset in the pool satisfies.
        Positive.
    tasks : tuple of TaskAdmissionSpec
        One entry per program task, each task once; `AdmissionFacts.validate`
        requires the program's task order.

    Raises
    ------
    ValueError
        If a field is empty or out of range, or a task is listed twice.
    """

    device_id: str
    pool_capacity_bytes: int
    object_capacity_bytes: int
    minimum_alignment: int
    tasks: tuple[TaskAdmissionSpec, ...]

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("admission device ID must be non-empty")
        if self.pool_capacity_bytes <= 0:
            raise ValueError("admission pool capacity must be positive")
        if not 0 < self.object_capacity_bytes <= self.pool_capacity_bytes:
            raise ValueError(
                "admission object capacity must be positive and no larger than "
                "the pool capacity"
            )
        if self.minimum_alignment <= 0:
            raise ValueError("admission minimum alignment must be positive")
        task_ids = tuple(item.task_id for item in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("admission task IDs must be unique")

    #: `tasks` as columns, and their digest, shared with every copy `replace`
    #: makes of these facts. The tasks are nearly all of the facts and the
    #: same in each capacity variant of them, so a variant's digest hashes its
    #: own fields and this one, and every variant compiles from the same
    #: columns; a copy uses them only if it was taken of the same tasks.
    _tasks_derived: list[tuple[tuple[TaskAdmissionSpec, ...], TaskColumns, str]] = (
        dataclass_field(default_factory=list, repr=False, compare=False)
    )

    #: The digest, once computed: the facts are immutable and every plan key
    #: and certificate names them.
    _digest_cache: list[str] = dataclass_field(
        default_factory=list, init=False, repr=False, compare=False
    )

    @property
    def digest(self) -> str:
        """SHA-256 hex digest naming these facts, computed once.

        It covers every field, the tasks through the digest of their
        columns; plan keys and layout certificates name the facts by it.
        """
        if not self._digest_cache:
            encoded = json.dumps(
                {
                    "device_id": self.device_id,
                    "minimum_alignment": self.minimum_alignment,
                    "object_capacity_bytes": self.object_capacity_bytes,
                    "pool_capacity_bytes": self.pool_capacity_bytes,
                    "schema": _SCHEMA,
                    "tasks_digest": self._digest_of_tasks(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self._digest_cache.append(hashlib.sha256(encoded.encode()).hexdigest())
        return self._digest_cache[0]

    @property
    def task_columns(self) -> TaskColumns:
        """The tasks as arrays, one per field, as the library reads them.

        Extracted once and shared by every copy of these facts with the same
        tasks, such as a copy at another capacity.
        """

        return self._derived_tasks()[0]

    def _digest_of_tasks(self) -> str:
        return self._derived_tasks()[1]

    def _derived_tasks(self) -> tuple[TaskColumns, str]:
        shared = self._tasks_derived
        if shared and shared[0][0] is self.tasks:
            return shared[0][1], shared[0][2]
        from shadowspill.planning.memory_search.admission.task_columns import (
            task_columns,
        )

        columns = task_columns(self.tasks)
        digest = columns.digest()
        shared[:] = [(self.tasks, columns, digest)]
        return columns, digest

    @shared_writer
    def to_dict(self) -> dict[str, object]:
        """Write the facts as a JSON-compatible mapping.

        Returns
        -------
        dict
            Every field by name, the tasks as their own mappings, and the
            record's ``schema``.
        """
        return {
            "device_id": self.device_id,
            "minimum_alignment": self.minimum_alignment,
            "object_capacity_bytes": self.object_capacity_bytes,
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "schema": _SCHEMA,
            "tasks": [item.to_dict() for item in self.tasks],
        }

    def to_json(self) -> str:
        """Write the facts as compact JSON with sorted keys.

        Returns
        -------
        str
            `AdmissionFacts.to_dict`, encoded.
        """
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: object) -> AdmissionFacts:
        """Read facts back from what `AdmissionFacts.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.

        Returns
        -------
        AdmissionFacts
            The facts.

        Raises
        ------
        ValueError
            If the schema is another one, or a field is missing or malformed.
        """
        data = _mapping(value, "admission")
        schema = _string(data.get("schema"), "admission.schema")
        if schema != _SCHEMA:
            raise ValueError(f"admission.schema: unsupported schema {schema!r}")
        steps = _AllocationSteps()
        return cls(
            device_id=_string(data.get("device_id"), "admission.device_id"),
            pool_capacity_bytes=_integer(
                data.get("pool_capacity_bytes"), "admission.pool_capacity_bytes"
            ),
            object_capacity_bytes=_integer(
                data.get("object_capacity_bytes"),
                "admission.object_capacity_bytes",
            ),
            minimum_alignment=_integer(
                data.get("minimum_alignment"), "admission.minimum_alignment"
            ),
            tasks=tuple(
                TaskAdmissionSpec.from_value(
                    item, f"admission.tasks[{index}]", _steps=steps
                )
                for index, item in enumerate(
                    _list(data.get("tasks"), "admission.tasks")
                )
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> AdmissionFacts:
        """Read facts back from what `AdmissionFacts.to_json` wrote.

        Parameters
        ----------
        payload : str
            The JSON text.

        Returns
        -------
        AdmissionFacts
            The facts.

        Raises
        ------
        ValueError
            If the text is not JSON or `AdmissionFacts.from_dict` refuses its
            record.
        """
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("admission JSON is invalid") from exc
        return cls.from_dict(value)

    def validate(self, program: ShadowSpillProgram) -> None:
        """Check that these facts describe exactly ``program``.

        Parameters
        ----------
        program : ShadowSpillProgram
            The program the facts are for.

        Raises
        ------
        ValueError
            If the program has more than one device or lacks the facts' device,
            the tasks do not follow the program's task order exactly, or a task
            names an alias group the program lacks.
        """

        devices = {item.device_id for item in program.devices}
        if self.device_id not in devices:
            raise ValueError(f"admission facts names unknown device {self.device_id!r}")
        if len(program.devices) != 1:
            raise ValueError(
                "one AdmissionFacts describes exactly one execution "
                f"pool; ShadowSpillProgram has {len(program.devices)} devices"
            )
        expected_tasks = tuple(item.task_id for item in program.tasks)
        actual_tasks = tuple(item.task_id for item in self.tasks)
        if actual_tasks != expected_tasks:
            raise ValueError(
                "admission tasks must exactly follow ShadowSpillProgram task order: "
                f"expected={expected_tasks!r}, actual={actual_tasks!r}"
            )
        aliases = {item.alias_group_id for item in program.alias_groups}
        for task in self.tasks:
            referenced = (
                *task.fresh_output_aliases,
                *task.replacement_aliases,
                *(item.source_alias_group_id for item in task.storage_handoffs),
                *(item.destination_alias_group_id for item in task.storage_handoffs),
            )
            unknown = sorted(set(referenced) - aliases)
            if unknown:
                raise ValueError(
                    f"task admission {task.task_id!r} references unknown aliases "
                    f"{unknown}"
                )


def _enum(
    enum_type: type[TaskAllocationStepKind], value: object, path: str
) -> TaskAllocationStepKind:
    raw = _string(value, path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: unknown value {raw!r}") from exc


__all__ = [
    "AdmissionFacts",
    "StorageHandoff",
    "TaskAdmissionSpec",
    "TaskAllocationStep",
    "TaskAllocationStepKind",
]
