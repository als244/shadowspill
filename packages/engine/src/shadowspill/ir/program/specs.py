"""What a program is made of: devices, alias groups, objects, profiles, tasks."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.schema import artifact_schema
from shadowspill.serialization.sharing import shared_writer

from ..serialization import (
    JsonValue,
    string_list,
)
from ..validation import (
    expect_boolean,
    expect_integer,
    expect_list,
    expect_mapping,
    expect_string,
    fail,
    field,
    index_unique,
    require,
    require_identifier,
    require_non_negative,
    require_positive,
    require_tuple,
)
from .enums import ObjectRole, Persistence, ResourceKind, SharedResidencyPolicy
from .timing import TaskOverlapCost, TaskResourcePhase, validate_task_timing
from .transfers import TransferSpec

PROGRAM_SCHEMA = artifact_schema("program")


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """One device the program's tasks run on.

    Parameters
    ----------
    device_id : str
        The device's identity, unique in the program.

    Raises
    ------
    ValidationError
        If ``device_id`` is empty or padded with whitespace.
    """

    device_id: str

    def __post_init__(self) -> None:
        require_identifier(self.device_id, "device.device_id")

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {"device_id": self.device_id}

    @classmethod
    def from_value(cls, value: object, path: str) -> DeviceSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.devices[0]"``.
            Error paths start with it.

        Returns
        -------
        DeviceSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        return cls(
            device_id=expect_string(
                field(data, "device_id", path), f"{path}.device_id"
            ),
        )


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """The queue a task runs on: one lane of one kind on one device.

    Tasks on the same device, kind and lane run one at a time, in program order.

    Parameters
    ----------
    device_id : str
        The device, by `DeviceSpec.device_id`.
    kind : ResourceKind
        The kind of queue.
    lane : int, default 0
        Which queue of that kind on the device. At least zero.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace, ``kind`` is not a
        `ResourceKind`, or ``lane`` is negative.
    """

    device_id: str
    kind: ResourceKind
    lane: int = 0

    def __post_init__(self) -> None:
        require_identifier(self.device_id, "resource.device_id")
        require(isinstance(self.kind, ResourceKind), "resource.kind", "invalid kind")
        require_non_negative(self.lane, "resource.lane")

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {"device_id": self.device_id, "kind": self.kind.value, "lane": self.lane}

    @classmethod
    def from_value(cls, value: object, path: str) -> ResourceSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.tasks[0].resource"``.
            Error paths start with it.

        Returns
        -------
        ResourceSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        kind_value = expect_string(field(data, "kind", path), f"{path}.kind")
        try:
            kind = ResourceKind(kind_value)
        except ValueError:
            fail(f"{path}.kind", f"unknown resource kind {kind_value!r}")
        return cls(
            device_id=expect_string(
                field(data, "device_id", path), f"{path}.device_id"
            ),
            kind=kind,
            lane=expect_integer(field(data, "lane", path), f"{path}.lane"),
        )


@dataclass(frozen=True, slots=True)
class AliasGroupSpec:
    """One storage on one device, which objects occupy and memory actions move.

    A group has at most one device copy and one spill copy, and every memory
    action moves or drops a whole group.

    Parameters
    ----------
    alias_group_id : str
        The group's identity, unique in the program.
    device_id : str
        The device that holds the device copy.
    size_bytes : int
        Size of the device copy, in bytes. Zero means the group needs no device
        storage: a schedule gives it no actions and never requires it resident.
    initial_version : int, default 0
        Version of the group's value before the first task. Each mutation of an
        object in the group adds its `MutationSpec.version_delta`.
    retain_spill_copy : bool, default False
        Whether the spill copy survives a fetch or a release. A schedule starts
        such a group with a current spill copy.
    shared_residency : SharedResidencyPolicy, optional
        How the runtime shares the group across plans; None for a group the
        schedule moves.
    transfer : TransferSpec, optional
        How the group moves when a move is more than one copy of
        ``size_bytes``; None for one copy each way.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace, a size or version is
        negative, or a field has the wrong type.
    """

    alias_group_id: str
    device_id: str
    size_bytes: int
    initial_version: int = 0
    retain_spill_copy: bool = False
    shared_residency: SharedResidencyPolicy | None = None
    transfer: TransferSpec | None = None

    def __post_init__(self) -> None:
        require_identifier(self.alias_group_id, "alias_group.alias_group_id")
        require_identifier(self.device_id, "alias_group.device_id")
        require_non_negative(self.size_bytes, "alias_group.size_bytes")
        require_non_negative(self.initial_version, "alias_group.initial_version")
        require(
            self.transfer is None or isinstance(self.transfer, TransferSpec),
            "alias_group.transfer",
            "invalid transfer specification",
        )
        require(
            isinstance(self.retain_spill_copy, bool),
            "alias_group.retain_spill_copy",
            "must be a boolean",
        )
        require(
            self.shared_residency is None
            or isinstance(self.shared_residency, SharedResidencyPolicy),
            "alias_group.shared_residency",
            "invalid shared residency policy",
        )

    @property
    def spill_size_bytes(self) -> int:
        """Size of the group's spill copy, in bytes.

        ``transfer.spill_size_bytes`` with a transfer recipe, ``size_bytes`` without.
        """
        return (
            self.size_bytes if self.transfer is None else self.transfer.spill_size_bytes
        )

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "alias_group_id": self.alias_group_id,
            "device_id": self.device_id,
            "initial_version": self.initial_version,
            "retain_spill_copy": self.retain_spill_copy,
            "shared_residency": (
                None if self.shared_residency is None else self.shared_residency.value
            ),
            "size_bytes": self.size_bytes,
            **({} if self.transfer is None else {"transfer": self.transfer.to_dict()}),
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> AliasGroupSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.alias_groups[0]"``.
            Error paths start with it.

        Returns
        -------
        AliasGroupSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        shared_value = field(data, "shared_residency", path)
        if shared_value is None:
            shared_residency = None
        else:
            shared_name = expect_string(shared_value, f"{path}.shared_residency")
            try:
                shared_residency = SharedResidencyPolicy(shared_name)
            except ValueError:
                fail(
                    f"{path}.shared_residency",
                    f"unknown shared residency policy {shared_name!r}",
                )
        return cls(
            alias_group_id=expect_string(
                field(data, "alias_group_id", path), f"{path}.alias_group_id"
            ),
            device_id=expect_string(
                field(data, "device_id", path), f"{path}.device_id"
            ),
            size_bytes=expect_integer(
                field(data, "size_bytes", path), f"{path}.size_bytes"
            ),
            initial_version=expect_integer(
                field(data, "initial_version", path), f"{path}.initial_version"
            ),
            retain_spill_copy=expect_boolean(
                field(data, "retain_spill_copy", path),
                f"{path}.retain_spill_copy",
            ),
            shared_residency=shared_residency,
            transfer=(
                None
                if data.get("transfer") is None
                else TransferSpec.from_value(data["transfer"], f"{path}.transfer")
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """One value tasks read or write: a byte range of one alias group.

    Several objects can share a group, as views of one storage.

    Parameters
    ----------
    object_id : str
        The object's identity, unique in the program.
    alias_group_id : str
        The group whose storage holds it.
    offset_bytes : int
        Where the object starts in the group's storage, in bytes.
    size_bytes : int
        Bytes of the group's storage the object reaches from ``offset_bytes``.
        The range lies inside the group.
    role : ObjectRole, default ObjectRole.OTHER
        What the object holds.
    persistence : Persistence, default Persistence.STEP
        How long its value lasts.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace, an offset or size is
        negative, or ``role`` or ``persistence`` has the wrong type.
    """

    object_id: str
    alias_group_id: str
    offset_bytes: int
    size_bytes: int
    role: ObjectRole = ObjectRole.OTHER
    persistence: Persistence = Persistence.STEP

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "object.object_id")
        require_identifier(self.alias_group_id, "object.alias_group_id")
        require_non_negative(self.offset_bytes, "object.offset_bytes")
        require_non_negative(self.size_bytes, "object.size_bytes")
        require(isinstance(self.role, ObjectRole), "object.role", "invalid role")
        require(
            isinstance(self.persistence, Persistence),
            "object.persistence",
            "invalid persistence",
        )

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "alias_group_id": self.alias_group_id,
            "object_id": self.object_id,
            "offset_bytes": self.offset_bytes,
            "persistence": self.persistence.value,
            "role": self.role.value,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> ObjectSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.objects[0]"``.
            Error paths start with it.

        Returns
        -------
        ObjectSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        role_value = expect_string(field(data, "role", path), f"{path}.role")
        persistence_value = expect_string(
            field(data, "persistence", path), f"{path}.persistence"
        )
        try:
            role = ObjectRole(role_value)
        except ValueError:
            fail(f"{path}.role", f"unknown object role {role_value!r}")
        try:
            persistence = Persistence(persistence_value)
        except ValueError:
            fail(
                f"{path}.persistence",
                f"unknown persistence {persistence_value!r}",
            )
        return cls(
            object_id=expect_string(
                field(data, "object_id", path), f"{path}.object_id"
            ),
            alias_group_id=expect_string(
                field(data, "alias_group_id", path), f"{path}.alias_group_id"
            ),
            offset_bytes=expect_integer(
                field(data, "offset_bytes", path), f"{path}.offset_bytes"
            ),
            size_bytes=expect_integer(
                field(data, "size_bytes", path), f"{path}.size_bytes"
            ),
            role=role,
            persistence=persistence,
        )


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """One task measurement, which tasks refer to by ID.

    Several tasks may share a profile.

    Parameters
    ----------
    profile_id : str
        The profile's identity, unique in the program.
    runtime_ns : int
        How long the task runs alone, in nanoseconds.
    workspace_bytes : int
        Peak bytes of temporary storage the task holds while it runs, besides
        its inputs and outputs.
    compatibility_digest : str
        Identity of the task contract the measurement was taken for.
    resource_phases : tuple of TaskResourcePhase, default ()
        The run split into sequential parts, whose durations sum to
        ``runtime_ns``; empty for one part of computation.
    overlap_costs : tuple of TaskOverlapCost, default ()
        How fast the task and timed transfer phases progress while they
        overlap, at most one sample per timing class and size; empty when not
        measured, which leaves both at full rate.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace, a measurement is
        negative, the phase durations do not sum to ``runtime_ns``, or two
        overlap samples share a class and size.
    """

    profile_id: str
    runtime_ns: int
    workspace_bytes: int
    compatibility_digest: str
    resource_phases: tuple[TaskResourcePhase, ...] = ()
    overlap_costs: tuple[TaskOverlapCost, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.profile_id, "profile.profile_id")
        require_non_negative(self.runtime_ns, "profile.runtime_ns")
        require_non_negative(self.workspace_bytes, "profile.workspace_bytes")
        require_identifier(self.compatibility_digest, "profile.compatibility_digest")
        validate_task_timing(self.runtime_ns, self.resource_phases, self.overlap_costs)

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the profile as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "compatibility_digest": self.compatibility_digest,
            "profile_id": self.profile_id,
            "runtime_ns": self.runtime_ns,
            "workspace_bytes": self.workspace_bytes,
            "resource_phases": [item.to_dict() for item in self.resource_phases],
            "overlap_costs": [item.to_dict() for item in self.overlap_costs],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskProfile:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.profiles[0]"``.
            Error paths start with it.

        Returns
        -------
        TaskProfile
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        return cls(
            profile_id=expect_string(
                field(data, "profile_id", path), f"{path}.profile_id"
            ),
            runtime_ns=expect_integer(
                field(data, "runtime_ns", path), f"{path}.runtime_ns"
            ),
            workspace_bytes=expect_integer(
                field(data, "workspace_bytes", path), f"{path}.workspace_bytes"
            ),
            compatibility_digest=expect_string(
                field(data, "compatibility_digest", path),
                f"{path}.compatibility_digest",
            ),
            resource_phases=tuple(
                TaskResourcePhase.from_value(item, f"{path}.resource_phases[{index}]")
                for index, item in enumerate(
                    expect_list(
                        data.get("resource_phases", []), f"{path}.resource_phases"
                    )
                )
            ),
            overlap_costs=tuple(
                TaskOverlapCost.from_value(item, f"{path}.overlap_costs[{index}]")
                for index, item in enumerate(
                    expect_list(data.get("overlap_costs", []), f"{path}.overlap_costs")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class MutationSpec:
    """An update a task makes to one of its inputs in place.

    Parameters
    ----------
    object_id : str
        The object updated, which is one of the task's inputs.
    version_delta : int, default 1
        How far the update advances the version of the object's alias group.
        Positive.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace, or ``version_delta``
        is not positive.
    """

    object_id: str
    version_delta: int = 1

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "mutation.object_id")
        require_positive(self.version_delta, "mutation.version_delta")

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {"object_id": self.object_id, "version_delta": self.version_delta}

    @classmethod
    def from_value(cls, value: object, path: str) -> MutationSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.tasks[0].mutations[0]"``.
            Error paths start with it.

        Returns
        -------
        MutationSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        return cls(
            object_id=expect_string(
                field(data, "object_id", path), f"{path}.object_id"
            ),
            version_delta=expect_integer(
                field(data, "version_delta", path), f"{path}.version_delta"
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One unit of work: where it runs, what it costs, what it reads and writes.

    A program lists its tasks in execution order. A task depends only on
    earlier tasks, among them the producer of every input an earlier task
    produces.

    Parameters
    ----------
    task_id : str
        The task's identity, unique in the program.
    resource : ResourceSpec
        The queue the task runs on.
    profile_id : str
        The `TaskProfile` that costs the task.
    dependencies : tuple of str, default ()
        Earlier tasks this one waits for: the producers of what it reads, the
        earlier writers of what it updates, and any ordering the frontend adds.
    inputs : tuple of str, default ()
        Objects the task reads.
    outputs : tuple of str, default ()
        Objects the task produces.
    mutations : tuple of MutationSpec, default ()
        Inputs the task updates in place, at most one each.
    phase : str, default "compute"
        Label of the part of the work the task belongs to, such as
        ``"forward"``. The IR carries it without interpreting it.
    requires_entrypoint : bool, default True
        Whether running the task calls a function bound to it. False for a
        boundary that does no work.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace, a collection is not a
        tuple or repeats an entry, or ``resource`` is not a `ResourceSpec`.
    """

    task_id: str
    resource: ResourceSpec
    profile_id: str
    dependencies: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    mutations: tuple[MutationSpec, ...] = ()
    phase: str = "compute"
    requires_entrypoint: bool = True

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "task.task_id")
        require(
            isinstance(self.resource, ResourceSpec), "task.resource", "invalid resource"
        )
        require_identifier(self.profile_id, "task.profile_id")
        require_tuple(self.dependencies, "task.dependencies")
        require_tuple(self.inputs, "task.inputs")
        require_tuple(self.outputs, "task.outputs")
        require_tuple(self.mutations, "task.mutations")
        index_unique(self.dependencies, "task.dependencies")
        index_unique(self.inputs, "task.inputs")
        index_unique(self.outputs, "task.outputs")
        index_unique(
            (mutation.object_id for mutation in self.mutations),
            "task.mutations",
        )
        require_identifier(self.phase, "task.phase")
        require(
            isinstance(self.requires_entrypoint, bool),
            "task.requires_entrypoint",
            "must be a boolean",
        )

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "dependencies": string_list(self.dependencies),
            "inputs": string_list(self.inputs),
            "mutations": [mutation.to_dict() for mutation in self.mutations],
            "outputs": string_list(self.outputs),
            "phase": self.phase,
            "profile_id": self.profile_id,
            "requires_entrypoint": self.requires_entrypoint,
            "resource": self.resource.to_dict(),
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.tasks[0]"``.
            Error paths start with it.

        Returns
        -------
        TaskSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        dependencies = expect_list(
            field(data, "dependencies", path), f"{path}.dependencies"
        )
        inputs = expect_list(field(data, "inputs", path), f"{path}.inputs")
        outputs = expect_list(field(data, "outputs", path), f"{path}.outputs")
        mutations = expect_list(field(data, "mutations", path), f"{path}.mutations")
        return cls(
            task_id=expect_string(field(data, "task_id", path), f"{path}.task_id"),
            resource=ResourceSpec.from_value(
                field(data, "resource", path), f"{path}.resource"
            ),
            profile_id=expect_string(
                field(data, "profile_id", path), f"{path}.profile_id"
            ),
            dependencies=tuple(
                expect_string(item, f"{path}.dependencies[{index}]")
                for index, item in enumerate(dependencies)
            ),
            inputs=tuple(
                expect_string(item, f"{path}.inputs[{index}]")
                for index, item in enumerate(inputs)
            ),
            outputs=tuple(
                expect_string(item, f"{path}.outputs[{index}]")
                for index, item in enumerate(outputs)
            ),
            mutations=tuple(
                MutationSpec.from_value(item, f"{path}.mutations[{index}]")
                for index, item in enumerate(mutations)
            ),
            phase=expect_string(field(data, "phase", path), f"{path}.phase"),
            requires_entrypoint=expect_boolean(
                field(data, "requires_entrypoint", path),
                f"{path}.requires_entrypoint",
            ),
        )
