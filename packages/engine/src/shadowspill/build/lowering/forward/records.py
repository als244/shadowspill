"""Immutable bindings exchanged by forward-lowering phases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.build.lowering.registration import RegistrationBinding
from shadowspill.ir import TaskSpec
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.representations import RootInputKey
from shadowspill.task.entrypoints import TaskEntrypoint
from shadowspill.task.geometry import TaskStorageGeometry
from shadowspill.task.slots import ObjectSlot

from ..objects import ObjectCatalog


@dataclass(frozen=True, slots=True)
class ForwardObjects:
    catalog: ObjectCatalog
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[ObjectSlot, ...]
    root_objects: dict[RootInputKey, str]


@dataclass(frozen=True, slots=True)
class ForwardTaskGeometry:
    contracts: tuple[TaskStorageContract, ...]
    layouts: tuple[TaskStorageGeometry, ...]
    profile_ids: tuple[str, ...]
    control_profile_id: str


@dataclass(frozen=True, slots=True)
class ForwardTaskGraph:
    tasks: tuple[TaskSpec, ...]
    entrypoints: tuple[TaskEntrypoint, ...]
    #: What to call for each task. The entrypoint is neutral; this is not.
    executables: Mapping[str, GraphArtifact]
    produced_aliases: frozenset[str]
    # One entry per public pytree leaf; literals have no storage object.
    public_outputs: tuple[str | None, ...]
