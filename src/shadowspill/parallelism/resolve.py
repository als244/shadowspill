"""Validate model geometry against one central parallelism declaration."""

from dataclasses import dataclass
from math import prod

from .ownership import ParameterOwnership, StorageOwnership
from .placements import Sharded
from .selectors import select, validate_selectors
from .spec import Parallelism
from .storage import Component


@dataclass(frozen=True)
class ParameterGeometry:
    """Internal framework-neutral inventory entry; names share one tensor."""

    names: tuple[str, ...]
    shape: tuple[int, ...]
    trainable: bool
    storage_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("parameter geometry needs unique registered names")
        if any(type(size) is not int or size < 0 for size in self.shape):
            raise ValueError("parameter shapes must contain nonnegative integers")
        if self.storage_bytes is not None and (
            type(self.storage_bytes) is not int or self.storage_bytes < 0
        ):
            raise ValueError("parameter storage bytes must be a nonnegative integer")


def resolve_parameters(
    parallelism: Parallelism, parameters: tuple[ParameterGeometry, ...]
) -> tuple[ParameterOwnership, ...]:
    names = tuple(name for item in parameters for name in item.names)
    if len(set(names)) != len(names):
        raise ValueError("registered names must identify one tensor")
    validate_selectors(names, parallelism.parameter_placements)
    validate_selectors(names, parallelism.parameter_storage)
    return tuple(_resolve(parallelism, item) for item in parameters)


def _resolve(spec: Parallelism, item: ParameterGeometry) -> ParameterOwnership:
    placement = select(
        item.names, spec.parameter_placements, spec.default_parameter_placement
    )
    storage = select(
        item.names,
        {
            name: value.inherit(spec.default_parameter_storage)
            for name, value in spec.parameter_storage.items()
        },
        spec.default_parameter_storage,
    )
    group = spec.group(placement.group)
    shape = list(item.shape)
    if isinstance(placement, Sharded):
        if not -len(shape) <= placement.dim < len(shape):
            raise ValueError(
                f"{item.names[0]}: sharded dimension exceeds parameter rank"
            )
        shape[placement.dim] *= group.size
        copies = tuple(
            tuple(members[index] for members in group.groups)
            for index in range(group.size)
        )
    else:
        # Replicated denotes one logical parameter, including repeated group
        # instances. Distinct parameters must have distinct registered names.
        copies = (tuple(range(spec.world_size)),)
    by_rank = {rank: members for members in copies for rank in members}
    cutoff = storage.min_shard_bytes or 0
    if cutoff and item.storage_bytes is None:
        raise ValueError(f"{item.names[0]}: min_shard_bytes requires physical sizes")
    small = item.storage_bytes is not None and item.storage_bytes < cutoff

    def ownership(component: Component) -> StorageOwnership:
        groups = spec.group(storage.group_for(component)).groups
        for members in groups:
            if any(rank not in by_rank[members[0]] for rank in members):
                raise ValueError(
                    f"{item.names[0]} {component}: storage group combines different "
                    "logical parameter shards"
                )
        if small or getattr(storage, component) == "replicated":
            groups = tuple((rank,) for rank in range(spec.world_size))
        return StorageOwnership(prod(item.shape), groups)

    return ParameterOwnership(
        item.names[0],
        item.names,
        item.shape,
        tuple(shape),
        item.trainable,
        copies,
        ownership("weights"),
        ownership("gradients"),
        ownership("optimizer"),
        cutoff,
    )
