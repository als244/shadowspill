"""Admit installed initializers before opening sources or touching mutable state."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts.state_inventory import program_state
from shadowspill.artifacts.tensors import registration_name
from shadowspill.planning.shared_state import shared_state
from shadowspill.training.definitions.initialization import (
    InitializationContext,
    InitializationRuntime,
    ModelInitialization,
    TensorInitializer,
)
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill.training.preparation.setup import ResolvedSetup
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.frozen import FrozenBase

from .frozen import validate_base

if TYPE_CHECKING:
    import torch

    from shadowspill.artifacts.tensors import TensorChunk
    from shadowspill.execution.executor import Executor


@dataclass(frozen=True)
class PreparedInitialization:
    context: InitializationContext
    description: ModelInitialization
    source: DefinitionSource

    def signature(self) -> dict[str, Any]:
        """Portable startup evidence; no callback or filesystem path is serialized."""
        return {
            "logical_rank": self.context.logical_rank,
            "weights": self.context.weights.to_dict(),
            "model_source": self.source.to_dict(),
            "peak_bytes": self.description.peak_bytes,
            "chunk_bytes": self.description.chunk_bytes,
            "disk_bytes": self.description.disk_bytes,
            "temporary": self.context.temporary,
        }


def prepare_initialization(
    setup: ResolvedSetup,
) -> Mapping[int, PreparedInitialization]:
    """CPU creation preflight; inspection/preparation may omit initial weights."""
    model = setup.config.model
    if model.initialize is None:
        raise ValueError("worker creation requires explicit weights or a checkpoint")
    result = {}
    for rank in range(setup.context.parallelism.world_size):
        context = InitializationContext(
            model, model.initialize, setup.context.parallelism, rank
        )
        description = setup.model.describe_initialization(context)
        if description.peak_bytes > setup.config.execution.max_staging_bytes:
            raise ValueError(
                "model initialization exceeds the configured host staging limit"
            )
        result[rank] = PreparedInitialization(context, description, setup.model.source)
    return MappingProxyType(result)


def prepare_frozen_initialization(
    setup: ResolvedSetup, base: FrozenBase, rank: int
) -> PreparedInitialization:
    """A partial checkpoint, not historical fresh settings, selects its base."""
    validate_base(base, setup.config)
    assert base.model.initialize is not None
    context = InitializationContext(
        base.model, base.model.initialize, setup.context.parallelism, rank
    )
    model = replace(setup.model, settings=base.model)
    return PreparedInitialization(
        context, model.describe_initialization(context), base.source
    )


def prepare_temporary_initialization(
    prepared: PreparedPlans, rank: int
) -> PreparedInitialization | None:
    """Describe derived model buffers absent from an exported weights generation."""
    plans = prepared.plans[rank]
    primary = min(plans)
    programs = {
        "": plans[primary].program,
        **{name: plan.program for name, plan in plans.items() if name != primary},
    }
    if not any(
        not row["parameter"] and not row.get("persistent", True)
        for program in programs.values()
        for row in program.bindings.get("registrations", ())
    ):
        return None
    shared = shared_state(programs, logical_rank=rank)
    requested = set()
    for name, program in programs.items():
        inventory = program_state(program, rank)
        bindings = {item.identity: item for item in inventory.bindings}
        protected = {
            bindings[item.identity].alias_group_id
            for item in inventory.tensors
            if item.checkpoint
            and item.identity.role in {"parameter", "buffer", "master"}
        }
        aliases = {obj.object_id: obj.alias_group_id for obj in program.objects}
        sizes = {
            alias.alias_group_id: alias.size_bytes for alias in program.alias_groups
        }
        visited = set(shared.aliases[name]) | protected
        for row in program.bindings["registrations"]:
            alias = aliases[row["object_id"]]
            if alias in visited or not sizes[alias]:
                continue
            visited.add(alias)
            if not row["parameter"] and not row.get("persistent", True):
                requested.add(registration_name(row))
    if not requested:
        return None
    setup = prepared.inventory.layouts.setup
    model = setup.config.model
    assert model.initialize is not None
    context = InitializationContext(
        model,
        model.initialize,
        setup.context.parallelism,
        rank,
        tuple(sorted(requested)),
    )
    return PreparedInitialization(
        context, setup.model.describe_initialization(context), setup.model.source
    )


def restore_staging(
    initial: PreparedInitialization | None, capacity: int, *, limit: int = 64 << 20
) -> tuple[int, int]:
    """Bound a retained base loader plus destination and cast/read scratch."""
    peak = 0 if initial is None else initial.description.peak_bytes
    chunk = min(limit, (capacity - peak) // 2)
    if initial is not None:
        chunk = min(chunk, initial.description.chunk_bytes)
    if chunk < 1:
        raise ValueError("checkpoint base and restore chunks exceed admitted staging")
    return chunk, peak + 2 * chunk


def initialize(
    executor: "Executor",
    prepared: PreparedInitialization,
    runtime: InitializationRuntime,
    budget: BufferBudget,
) -> None:
    """Use the existing owned-region/default/master/optimizer initialization path."""
    if executor.logical_rank != prepared.context.logical_rank:
        raise ValueError("initializer belongs to another logical worker rank")
    description = prepared.description
    with (
        budget.reserve(description.peak_bytes, timeout=0),
        open_initialization(prepared, runtime) as fill,
    ):
        executor.initialize(fill, chunk_bytes=description.chunk_bytes)


def check_disk(
    prepared: PreparedInitialization, runtime: InitializationRuntime
) -> None:
    if prepared.description.disk_bytes > runtime.disk_bytes:
        raise ValueError("initializer exceeds the admitted worker disk allowance")


@contextmanager
def open_initialization(
    prepared: PreparedInitialization, runtime: InitializationRuntime
) -> Iterator[TensorInitializer]:
    """Borrow the existing validated loader under the caller's host reservation."""
    check_disk(prepared, runtime)
    description = prepared.description
    with description.open(runtime) as fill:
        if not callable(fill):
            raise TypeError("initialization context must yield a tensor filler")

        def fill_chunk(request: "TensorChunk", destination: "torch.Tensor") -> None:
            if (
                prepared.context.temporary is not None
                and request.name not in prepared.context.temporary
            ):
                raise ValueError(
                    "temporary initializer requested undeclared model state"
                )
            if (
                destination.device.type != "cpu"
                or destination.ndim != 1
                or not destination.is_contiguous()
                or destination.dtype != request.dtype
                or destination.numel() * destination.element_size()
                > description.chunk_bytes
            ):
                raise ValueError("initializer received an incompatible owned CPU chunk")
            fill(request, destination)

        yield fill_chunk
