"""Immutable bindings exchanged by forward-lowering phases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch.utils._pytree import TreeSpec

from shadowspill.ir import ResidencySpec, ShadowSpillProgram, TaskSpec
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.task.entrypoints import TaskEntrypoint
from shadowspill.task.layout import CompiledTaskLayout
from shadowspill.task.slots import ObjectSlot

from ..catalog import ObjectCatalog, RegistrationBinding
from ..profiles import TaskProfileCatalog


@dataclass(frozen=True, slots=True)
class LoweredForwardProgram:
    """Canonical forward program plus non-serialized PyTorch bindings."""

    program: ShadowSpillProgram
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    entrypoints: tuple[TaskEntrypoint, ...]
    #: What to call for each task. The entrypoint is neutral; this is not.
    executables: Mapping[str, GraphArtifact]
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[ObjectSlot, ...]
    public_outputs: tuple[str, ...]
    output_tree_spec: TreeSpec
    output_leaf_count: int


@dataclass(frozen=True, slots=True)
class ForwardObjects:
    catalog: ObjectCatalog
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[ObjectSlot, ...]
    root_objects: dict[int, str]


@dataclass(frozen=True, slots=True)
class ForwardPhysicalLayout:
    contracts: tuple[TaskStorageContract, ...]
    layouts: tuple[CompiledTaskLayout, ...]
    profiles: TaskProfileCatalog
    profile_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ForwardTaskGraph:
    tasks: tuple[TaskSpec, ...]
    entrypoints: tuple[TaskEntrypoint, ...]
    #: What to call for each task. The entrypoint is neutral; this is not.
    executables: Mapping[str, GraphArtifact]
    produced_aliases: frozenset[str]
    public_outputs: tuple[str, ...]
