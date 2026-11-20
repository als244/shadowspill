"""Editable profile collections with plain numeric per-task records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from shadowspill.ir import TaskOverlapCost, TaskResourcePhase
from shadowspill.serialization.sharing import shared_writer

from ._values import freeze, nonnegative_int, plain, seconds
from .archive import Artifact


@dataclass(frozen=True)
class TaskProfile:
    """The measured or supplied costs of one task form.

    A field left None is unspecified: `Profiles.merge` keeps the earlier value,
    and `apply_profiles` takes a missing workspace size from the build. A value
    marked ``"estimated"`` in ``metadata``, or unmarked, is an estimate, and
    estimated memory leaves a Plan simulation-only. A workspace size alone,
    without measured allocation evidence, does not qualify a Plan to run.

    Parameters
    ----------
    duration_seconds : float, optional
        Time the task takes alone, in seconds. Finite and at least zero.
    workspace_bytes : int, optional
        Peak temporary memory the task allocates besides the objects it reads
        and writes, in bytes. At least zero.
    compatibility_digest : str, optional
        The implementation the costs were measured for. When set, it must
        equal the Program's for the form.
    resource_phases : sequence of TaskResourcePhase, optional
        The task's duration as sequential phases, each on a named resource or
        on ordinary computation.
    overlap_costs : sequence of TaskOverlapCost, optional
        The task's progress while each kind of collective transfer overlaps
        it, as `measure_overlap` measures it.
    metadata : mapping, default {}
        Provenance and measurement details. ``duration_kind`` and
        ``memory_kind`` classify the duration and the memory as
        ``"measured"``, ``"estimated"`` (the default) or ``"static"``.

    Raises
    ------
    ValueError
        If a duration or size is negative or not finite, or ``duration_kind``
        or ``memory_kind`` has another value.
    TypeError
        If ``resource_phases`` or ``overlap_costs`` holds another type.
    """

    duration_seconds: float | None = None
    workspace_bytes: int | None = None
    compatibility_digest: str | None = None
    resource_phases: tuple[TaskResourcePhase, ...] | None = None
    overlap_costs: tuple[TaskOverlapCost, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration_seconds is not None:
            seconds(self.duration_seconds, "duration_seconds")
        if self.workspace_bytes is not None:
            nonnegative_int(self.workspace_bytes, "workspace_bytes")
        for name, kind in (
            ("resource_phases", TaskResourcePhase),
            ("overlap_costs", TaskOverlapCost),
        ):
            values = getattr(self, name)
            if values is not None:
                values = tuple(values)
                if not all(isinstance(value, kind) for value in values):
                    raise TypeError(f"{name} must contain {kind.__name__} records")
                object.__setattr__(self, name, values)
        object.__setattr__(self, "metadata", freeze(self.metadata))
        for key in ("duration_kind", "memory_kind"):
            if self.metadata.get(key, "estimated") not in (
                "measured",
                "estimated",
                "static",
            ):
                raise ValueError(f"{key} must be measured, estimated or static")

    @shared_writer
    def to_dict(self) -> dict[str, Any]:
        """Return the profile as plain data.

        Returns
        -------
        dict
            Every field, with phases and overlap costs as records.
        """
        return self._to_dict(copy_metadata=True)

    def _to_dict(self, *, copy_metadata: bool) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "workspace_bytes": self.workspace_bytes,
            "compatibility_digest": self.compatibility_digest,
            "resource_phases": None
            if self.resource_phases is None
            else [item.to_dict() for item in self.resource_phases],
            "overlap_costs": None
            if self.overlap_costs is None
            else [item.to_dict() for item in self.overlap_costs],
            "metadata": plain(self.metadata) if copy_metadata else self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskProfile:
        """Rebuild a profile from `to_dict` data.

        Parameters
        ----------
        value : dict
            The profile, in `to_dict` form.

        Returns
        -------
        TaskProfile
            The profile.

        Raises
        ------
        ValueError
            If a field is invalid.
        """
        data = dict(value)
        for key, record in (
            ("resource_phases", TaskResourcePhase),
            ("overlap_costs", TaskOverlapCost),
        ):
            if data.get(key) is not None:
                data[key] = tuple(
                    record.from_value(item, f"profile.{key}[{index}]")
                    for index, item in enumerate(data[key])
                )
        return cls(**data)

    def merged(self, later: TaskProfile) -> TaskProfile:
        """Return this profile with each field ``later`` supplies replaced.

        A supplied duration or workspace takes its classification from
        ``later``'s metadata, an estimate when unmarked; ``later``'s other
        metadata entries then replace this one's.

        Parameters
        ----------
        later : TaskProfile
            The profile whose supplied fields win.

        Returns
        -------
        TaskProfile
            The merged profile.

        Raises
        ------
        ValueError
            If the profiles name different implementations.
        """
        if (
            self.compatibility_digest is not None
            and later.compatibility_digest is not None
            and self.compatibility_digest != later.compatibility_digest
        ):
            raise ValueError("cannot merge profiles for different task definitions")
        metadata = dict(self.metadata)
        for field_name, kind in (
            ("duration_seconds", "duration_kind"),
            ("workspace_bytes", "memory_kind"),
        ):
            if getattr(later, field_name) is not None:
                metadata[kind] = later.metadata.get(kind, "estimated")
        metadata.update(later.metadata)
        values = {
            key: getattr(later, key)
            for key in (
                "duration_seconds",
                "workspace_bytes",
                "compatibility_digest",
                "resource_phases",
                "overlap_costs",
            )
            if getattr(later, key) is not None
        }
        return replace(self, **values, metadata=metadata)


@dataclass
class Profiles(Artifact):
    """Task profiles measured on one setup, by profile key.

    `measure_program` returns one, and `measure_overlap` an overlay to merge
    into it; one can also be written by hand, as estimates. It is editable:
    `apply_profiles` copies what it uses, so later edits change no
    `ProfiledProgram`.

    Parameters
    ----------
    setup : dict, default {}
        The device and software builds the profiles describe. `apply_profiles`
        requires it to equal the Program's ``setup``.
    tasks : dict of str to TaskProfile, default {}
        Profiles by the Program's profile keys.
    fixed_device_bytes : int, optional
        Device memory the worker needs outside the execution pool, in bytes:
        its context, libraries, retained provider state and requested
        headroom.
    metadata : dict, default {}
        Provenance and measurement details, such as ``profiling_options`` and
        ``task_cache``. ``memory_kind`` classifies ``fixed_device_bytes`` as a
        `TaskProfile`'s does its memory.
    """

    artifact_kind: ClassVar[str] = "profiles"
    setup: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, TaskProfile] = field(default_factory=dict)
    fixed_device_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def merge(self, *others: Profiles) -> Profiles:
        """Return a copy with the fields later collections supply replaced.

        Each collection is applied in turn: a task's profile merges as
        `TaskProfile.merged` does, a supplied ``fixed_device_bytes`` replaces
        the earlier one, and metadata entries replace earlier ones. Fields a
        collection leaves unspecified are kept. This collection is unchanged.

        Parameters
        ----------
        *others : Profiles
            Collections for the same setup, in order.

        Returns
        -------
        Profiles
            The merged collection.

        Raises
        ------
        ValueError
            If a collection describes another setup, or a task's profiles name
            different implementations.
        """
        result = Profiles.from_dict(self.to_dict())
        for other in others:
            if freeze(result.setup) != freeze(other.setup):
                raise ValueError("profiles describe different hardware/runtime setups")
            for key, record in other.tasks.items():
                result.tasks[key] = (
                    result.tasks[key].merged(record) if key in result.tasks else record
                )
            if other.fixed_device_bytes is not None:
                result.fixed_device_bytes = other.fixed_device_bytes
                result.metadata["memory_kind"] = other.metadata.get(
                    "memory_kind", "estimated"
                )
            result.metadata.update(other.metadata)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the collection as plain data.

        Returns
        -------
        dict
            ``setup``, ``tasks`` as `TaskProfile.to_dict` records,
            ``fixed_device_bytes`` and ``metadata``.

        Raises
        ------
        ValueError
            If ``fixed_device_bytes`` is not a nonnegative integer.
        """
        if self.fixed_device_bytes is not None:
            nonnegative_int(self.fixed_device_bytes, "fixed_device_bytes")
        return {
            "setup": plain(freeze(self.setup)),
            "tasks": {key: record.to_dict() for key, record in self.tasks.items()},
            "fixed_device_bytes": self.fixed_device_bytes,
            "metadata": plain(freeze(self.metadata)),
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Profiles:
        """Rebuild a collection from `to_dict` data.

        Parameters
        ----------
        value : dict
            The collection, in `to_dict` form.
        payloads : mapping of str to bytes, optional
            Binary payloads; profiles hold none, so it must be empty.

        Returns
        -------
        Profiles
            The collection.

        Raises
        ------
        ValueError
            If ``payloads`` is not empty, or a profile is invalid.
        """
        if payloads:
            raise ValueError("Profiles cannot contain executable payloads")
        return cls(
            setup=plain(freeze(value["setup"])),
            tasks={
                key: TaskProfile.from_dict(item) for key, item in value["tasks"].items()
            },
            fixed_device_bytes=value["fixed_device_bytes"],
            metadata=plain(freeze(value["metadata"])),
        )
