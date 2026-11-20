"""The static, unprofiled result of building one concrete computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from functools import cached_property
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    ObjectSpec,
    ResidencySpec,
    TaskAlternativeGroup,
    TaskSpec,
)
from shadowspill.ir.program.checks import (
    check_alias_groups,
    check_alternative_groups,
    check_objects,
    check_tasks,
    index_identities,
    require_shapes,
)
from shadowspill.parallelism import Parallelism
from shadowspill.serialization.sharing import (
    read_record,
    shared_writer,
    writing_metadata,
)

from ._values import digest, freeze, nonnegative_int, plain
from .archive import Artifact

if TYPE_CHECKING:
    from shadowspill.resources import Resource


@dataclass(frozen=True)
class TaskDefinition:
    """One task form of a Program, shared by every task that runs it.

    Parameters
    ----------
    profile_key : str
        The form's key: tasks name it, and profiles are looked up by it.
        Nonempty.
    compatibility_digest : str
        Identity of the form's finalized implementation. A profile measured
        for another implementation does not apply. Nonempty.
    workspace_bytes : int, optional
        Temporary memory the build knows the form needs, in bytes. None means
        unknown, never zero.
    name : str, default ""
        What the form does, as progress messages and reports name it.
    module_paths : sequence of str, default ()
        The model modules covered by the stages that use the form.

    Raises
    ------
    ValueError
        If a key is empty, or ``workspace_bytes`` is not a nonnegative integer.
    """

    profile_key: str
    compatibility_digest: str
    workspace_bytes: int | None = None
    name: str = ""
    module_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_key or not self.compatibility_digest:
            raise ValueError("task key and compatibility digest must be nonempty")
        if self.workspace_bytes is not None:
            nonnegative_int(self.workspace_bytes, "workspace_bytes")
        object.__setattr__(self, "module_paths", tuple(self.module_paths))


@dataclass(frozen=True)
class Program(Artifact):
    """A captured computation: its tasks, the storage they use, and their code.

    `build_step`, `build_forward_backward` and `build_forward` return one, as
    do `optimizer_program`, `backward_program` and `instantiate_examples`;
    ``Program.load`` reads a saved one. It holds no costs or pools and names no
    real device: `measure_program` profiles its tasks and `apply_profiles`
    attaches the profiles. Profile keys name task forms; task IDs name their
    occurrences. It is read-only.

    Attributes
    ----------
    devices : tuple of DeviceSpec
        The logical devices its tasks run on.
    alias_groups : tuple of AliasGroupSpec
        Its storage allocations, each holding one or more objects.
    objects : tuple of ObjectSpec
        The values its tasks read and write.
    tasks : tuple of TaskSpec
        The tasks of one invocation.
    task_definitions : tuple of TaskDefinition
        The task forms, one per profile key.
    initial_residency : tuple of ResidencySpec
        Where storage is when an invocation starts.
    final_residency : tuple of ResidencySpec
        Where storage must be when an invocation ends.
    task_alternative_groups : tuple of TaskAlternativeGroup
        Interchangeable task sequences, such as a stage's save and recompute
        forms; planning selects one of each.
    name : str
        A readable name, such as the model's class name.
    setup : mapping
        The device and software builds its code was finalized for. Profiles
        and execution must match it.
    sources : mapping of str to sequence of str
        Model names, such as parameter and input names, by storage ID.
    executables : mapping of str to bytes
        Finalized task code, by profile key.
    bindings : mapping
        How inputs, outputs, model state and controls map onto objects, with
        the build's other metadata. ``bindings["kind"]`` is ``"step"``,
        ``"forward_backward"``, ``"forward"`` or ``"optim_step"``.
    parallelism : Parallelism or None
        The ownership it was built with: one rank when the builder was given
        none.
    resources : mapping of str to Resource
        Shared resources its operators use, by name.
    operator_binaries : mapping of str to bytes
        Kernel binaries saved by `OperatorPreparation` adapters, by SHA-256
        hex digest.
    """

    artifact_kind: ClassVar[str] = "program"
    devices: tuple[DeviceSpec, ...]
    alias_groups: tuple[AliasGroupSpec, ...]
    objects: tuple[ObjectSpec, ...]
    tasks: tuple[TaskSpec, ...]
    task_definitions: tuple[TaskDefinition, ...]
    initial_residency: tuple[ResidencySpec, ...] = ()
    final_residency: tuple[ResidencySpec, ...] = ()
    task_alternative_groups: tuple[TaskAlternativeGroup, ...] = ()
    name: str = ""
    setup: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, Any] = field(default_factory=dict)
    executables: Mapping[str, bytes] = field(default_factory=dict, repr=False)
    bindings: Mapping[str, Any] = field(default_factory=dict)
    parallelism: Parallelism | None = None
    resources: Mapping[str, Resource] = field(default_factory=dict)
    operator_binaries: Mapping[str, bytes] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        from shadowspill.resources.description import validate_resources

        if self.parallelism is not None and not isinstance(
            self.parallelism, Parallelism
        ):
            raise TypeError("parallelism must be Parallelism")
        validate_resources(self.resources, self.parallelism)
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))
        require_shapes(self)
        if not isinstance(self.task_definitions, tuple):
            raise TypeError("task_definitions must be a tuple")
        identities = index_identities(
            self, tuple(item.profile_key for item in self.task_definitions)
        )
        check_alias_groups(self, identities)
        check_objects(self, identities)
        check_tasks(self, identities, check_alternative_groups(self, identities))
        for name in ("initial_residency", "final_residency"):
            entries = getattr(self, name)
            if not isinstance(entries, tuple):
                raise TypeError(f"{name} must be a tuple")
            keys = [item.alias_group_id for item in entries]
            if len(set(keys)) != len(keys) or set(keys) - identities.alias_ids.keys():
                raise ValueError(f"{name} contains duplicate or unknown storage IDs")
        object.__setattr__(self, "setup", freeze(self.setup))
        object.__setattr__(self, "sources", freeze(self.sources))
        if any(not isinstance(value, bytes) for value in self.executables.values()):
            raise TypeError("executable payloads must be immutable bytes")
        object.__setattr__(
            self, "executables", MappingProxyType(dict(self.executables))
        )
        object.__setattr__(self, "bindings", freeze(self.bindings))
        for checksum, value in self.operator_binaries.items():
            if not isinstance(value, bytes) or sha256(value).hexdigest() != checksum:
                raise ValueError("invalid prepared operator binary digest")
        object.__setattr__(
            self, "operator_binaries", MappingProxyType(dict(self.operator_binaries))
        )
        if self.bindings.get("collective_tasks"):
            from shadowspill.communication.task_costs import task_collectives

            task_collectives(self)

    @cached_property
    def digest(self) -> str:
        """SHA-256 hex digest of the Program, with its code by checksum."""
        # Programs and their nested metadata are immutable. Runtime setup and
        # repeated checkpoints can share this identity without reserializing IR.
        with writing_metadata():
            return digest(self._to_dict(copy_metadata=False))

    @shared_writer
    def to_dict(self) -> dict[str, Any]:
        """Return the Program as plain data.

        Returns
        -------
        dict
            Every field, with ``executables`` and ``operator_binaries`` named
            by the SHA-256 hex digests of their bytes.
        """
        return self._to_dict(copy_metadata=True)

    def _to_dict(self, *, copy_metadata: bool) -> dict[str, Any]:
        # Integrity hashing consumes the same canonical values without making
        # editable copies of large, already owned allocation/capture metadata.
        value = {
            **{
                name: [item.to_dict() for item in getattr(self, name)]
                for name in (
                    "devices",
                    "alias_groups",
                    "objects",
                    "tasks",
                    "initial_residency",
                    "final_residency",
                    "task_alternative_groups",
                )
            },
            "task_definitions": [asdict(item) for item in self.task_definitions],
            "name": self.name,
            "setup": plain(self.setup) if copy_metadata else self.setup,
            "sources": plain(self.sources) if copy_metadata else self.sources,
            "executables": {
                key: sha256(value).hexdigest()
                for key, value in self.executables.items()
            },
            "bindings": plain(self.bindings) if copy_metadata else self.bindings,
            "parallelism": None
            if self.parallelism is None
            else self.parallelism.to_dict(),
        }
        if self.resources:
            value["resources"] = {
                name: resource.to_dict() for name, resource in self.resources.items()
            }
        if self.operator_binaries:
            value["operator_binaries"] = sorted(self.operator_binaries)
        return value

    def _archive_payloads(self) -> Mapping[str, bytes]:
        return {
            **self.operator_binaries,
            **{sha256(value).hexdigest(): value for value in self.executables.values()},
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Program:
        """Rebuild a Program from `to_dict` data and its binary payloads.

        Parameters
        ----------
        value : dict
            The Program, in `to_dict` form.
        payloads : mapping of str to bytes, optional
            Executables and operator binaries, by SHA-256 hex digest.

        Returns
        -------
        Program
            The Program.

        Raises
        ------
        ValueError
            If a payload is missing, or the Program is invalid.
        """
        records: dict[str, Callable[[object, str], Any]] = {
            "devices": DeviceSpec.from_value,
            "alias_groups": AliasGroupSpec.from_value,
            "objects": ObjectSpec.from_value,
            "tasks": TaskSpec.from_value,
            "initial_residency": ResidencySpec.from_value,
            "final_residency": ResidencySpec.from_value,
            "task_alternative_groups": TaskAlternativeGroup.from_value,
        }
        from shadowspill.resources import Resource

        data = dict(value)
        data["resources"] = {
            name: Resource.from_dict(item)
            for name, item in data.get("resources", {}).items()
        }
        if data.get("parallelism") is not None:
            data["parallelism"] = Parallelism.from_dict(data["parallelism"])
        for name, record in records.items():
            data[name] = tuple(
                read_record(record, item, f"program.{name}[{index}]")
                for index, item in enumerate(data[name])
            )
        data["task_definitions"] = tuple(
            TaskDefinition(**item) for item in data["task_definitions"]
        )
        try:
            data["executables"] = {
                key: (payloads or {})[checksum]
                for key, checksum in data.get("executables", {}).items()
            }
        except KeyError as error:
            raise ValueError("program executable payload is missing") from error
        try:
            data["operator_binaries"] = {
                checksum: (payloads or {})[checksum]
                for checksum in data.get("operator_binaries", ())
            }
        except KeyError as error:
            raise ValueError("program operator binary payload is missing") from error
        return cls(**data)
