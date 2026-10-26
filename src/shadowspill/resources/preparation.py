"""Provider setup for build/profiling, using ordinary device allocations."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import Program
from shadowspill.communication.binding.groups import GroupBindings

from .description import resource_groups
from .lifecycle import ResourceBindings


@dataclass
class PreparedResources:
    groups: GroupBindings | None
    retained_bytes: int


@contextmanager
def prepare_resources(
    program: Program,
    device: torch.device,
    rank_map: Mapping[int, int] | None = None,
) -> Iterator[PreparedResources]:
    members = resource_groups(program.resources)
    groups = None
    resources = ResourceBindings()
    try:
        if members:
            count = 1 if program.parallelism is None else program.parallelism.world_size
            if not dist.is_initialized() or dist.get_world_size() != count:
                raise ValueError(
                    "resource groups require rendezvous at the Program's world size"
                )
            groups = GroupBindings(
                identity=digest(
                    {name: value.to_dict() for name, value in program.resources.items()}
                ),
                groups=members,
                rank_map=rank_map,
                topology={},
                backend="nccl",
            )
        before = torch.cuda.memory_allocated(device)
        resources.open(
            program.resources,
            program.bindings.get("resource_sources", {}),
            device=device,
            groups=groups,
        )
        torch.cuda.synchronize(device)
        yield PreparedResources(
            groups, max(0, torch.cuda.memory_allocated(device) - before)
        )
    finally:
        try:
            resources.close()
        finally:
            if groups is not None:
                groups.close()
