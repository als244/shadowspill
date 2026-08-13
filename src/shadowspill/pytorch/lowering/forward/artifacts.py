"""Immutable bindings exchanged by forward-lowering phases."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils._pytree import TreeSpec

from shadowspill.ir import Program, ResidencySpec, TaskSpec
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.layout import CompiledTaskLayout

from ..catalog import ObjectCatalog, RegistrationBinding, TensorSlot
from ..profiles import TaskProfileCatalog
from ..task_binding import TaskStorageHandoff


@dataclass(frozen=True, slots=True)
class TaskEntrypoint:
    """Framework-only executable binding for one canonical task."""

    task_id: str
    module_target: str
    artifact: GraphArtifact
    input_slots: tuple[TensorSlot, ...]
    output_slots: tuple[TensorSlot, ...]
    replacement_output_leaves: tuple[int, ...] = ()
    storage_handoffs: tuple[TaskStorageHandoff, ...] = ()


@dataclass(frozen=True, slots=True)
class LoweredForwardProgram:
    """Canonical forward program plus non-serialized PyTorch bindings."""

    program: Program
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    entrypoints: tuple[TaskEntrypoint, ...]
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[TensorSlot, ...]
    output_tree_spec: TreeSpec
    output_leaf_count: int


@dataclass(frozen=True, slots=True)
class ForwardObjects:
    catalog: ObjectCatalog
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[TensorSlot, ...]
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
    produced_aliases: frozenset[str]
    public_outputs: tuple[str, ...]
