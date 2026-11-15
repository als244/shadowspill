"""Bind logical state identities to the existing owned checkpoint descriptors."""

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

from shadowspill.artifacts.state import StateId, StateRole, StateTensor
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.execution.controls import control_descriptors
from shadowspill.execution.tensor_layout import TensorLayout
from shadowspill.execution.tensors import owned_tensors
from shadowspill.parallelism import Sharded
from shadowspill.parallelism.selectors import select
from shadowspill.planning.shared_state import SharedState, StateLocation

from .mapping import LogicalView


@dataclass(frozen=True)
class LogicalTensor:
    """One logical view; tied views refer to the same owned archive descriptor.

    ``archive`` is absent only for an empty tensor with no allocated storage.
    This describes coordinates, not whether that representation was saved:
    weight selection and frozen dependencies remain separate archive policies.
    """

    tensor: StateTensor
    archive: TensorChunk | None
    view: LogicalView


def logical_tensors(
    layouts: Mapping[str, TensorLayout],
    shared: SharedState,
    logical_rank: int,
    *,
    roles: Collection[StateRole] | None = None,
) -> Iterator[LogicalTensor]:
    """Use execution's canonical owners, views and storage regions without I/O.

    Logical placement comes from saved Parallelism declarations; flat ownership
    comes from tensor_regions through owned_tensors. Neither is reconstructed
    from rank numbers, filenames or equal element counts.
    """
    physical: dict[StateLocation, TensorChunk] = {}
    host: dict[StateLocation, TensorChunk] = {}
    for name, layout in layouts.items():
        prefix = f"plans.{name}." if name else ""
        physical.update(
            ((name, tensor.alias), tensor.chunk(0, prefix))
            for tensor in owned_tensors(layout, skip_shared=bool(name))
        )
        host.update(
            ((name, chunk.name), chunk)
            for chunk in control_descriptors(
                layout.plan.program.bindings.get("optimizer_bindings", ()),
                shared_host=shared.host[name],
                prefix=prefix,
            )
        )
    visited: set[StateId] = set()
    for name, layout in layouts.items():
        inventory = program_state(layout.plan.program, logical_rank)
        bindings = {binding.identity: binding for binding in inventory.bindings}
        for tensor in inventory.tensors:
            if not tensor.checkpoint or tensor.identity in visited:
                continue
            if roles is not None and tensor.identity.role not in roles:
                continue
            visited.add(tensor.identity)
            binding = bindings[tensor.identity]
            geometry = tensor.geometry
            if binding.host_binding is not None:
                location = shared.host[name].get(
                    binding.host_binding, (name, binding.host_binding)
                )
                chunk = host.get(location)
            else:
                assert binding.alias_group_id is not None
                location = shared.aliases[name].get(
                    binding.alias_group_id, (name, binding.alias_group_id)
                )
                chunk = physical.get(location)
                if region := layout.regions.get(binding.object_id or ""):
                    geometry = region.geometry
            if chunk is None and prod(geometry["shape"]):
                raise ValueError(
                    f"checkpoint state has no owned export binding: {tensor.identity}"
                )
            if chunk is not None and str(chunk.dtype) != geometry["dtype"]:
                raise ValueError("tied checkpoint views require an explicit dtype map")
            shape, origin = _coordinates(tensor, geometry, layout, logical_rank)
            yield LogicalTensor(
                tensor,
                chunk,
                LogicalView(
                    shape=shape,
                    local_shape=tuple(geometry["shape"]),
                    origin=origin,
                    stride=tuple(geometry["stride"]),
                    storage_offset=geometry.get("storage_offset", 0),
                    storage_elements=0 if chunk is None else chunk.storage_elements,
                    region_offset=0 if chunk is None else chunk.region_offset,
                    region_elements=(
                        0
                        if chunk is None
                        else chunk.storage_elements
                        if chunk.region_elements is None
                        else chunk.region_elements
                    ),
                ),
            )


def _coordinates(
    tensor: StateTensor,
    geometry: Mapping[str, Any],
    layout: TensorLayout,
    logical_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    local_shape = tuple(geometry["shape"])
    origin = [0] * len(local_shape)
    owner = tensor.ownership
    spec = layout.plan.program.parallelism
    if not owner:
        if (
            spec is not None
            and spec.world_size > 1
            and tensor.identity.role in {"parameter", "buffer", "master"}
        ):
            raise ValueError("distributed model state has no saved logical ownership")
        return local_shape, tuple(origin)
    if tensor.identity.component_path:
        # Component geometry describes an encoding, not the parameter's logical
        # coordinates. A conversion declaration must describe that relationship.
        raise ValueError("encoded checkpoint state requires an explicit logical map")
    parameter_shape = tuple(owner["local_shape"])
    global_shape = tuple(owner["global_shape"])
    if tensor.identity.role == "optimizer" and local_shape != parameter_shape:
        # Scalar counters have their own logical scalar identity. Factored state
        # also keeps its own shape; a sharded parameter does not specify how an
        # unrelated non-scalar optimizer tensor maps into global coordinates.
        if local_shape and global_shape != parameter_shape:
            raise ValueError("optimizer state requires its own logical placement map")
        return local_shape, tuple(origin)
    if local_shape != parameter_shape:
        raise ValueError("state view differs from its declared local parameter shape")
    if spec is not None:
        placement = select(
            owner["names"],
            spec.parameter_placements,
            spec.default_parameter_placement,
        )
        expected = list(parameter_shape)
        if isinstance(placement, Sharded):
            if not -len(expected) <= placement.dim < len(expected):
                raise ValueError("saved logical placement exceeds the tensor rank")
            group = spec.group(placement.group)
            expected[placement.dim] *= group.size
            origin[placement.dim] = local_shape[placement.dim] * group.index(
                logical_rank
            )
        if tuple(expected) != global_shape:
            raise ValueError("saved logical placement disagrees with global ownership")
    elif global_shape != parameter_shape:
        raise ValueError("sharded logical state has no saved Parallelism declaration")
    return global_shape, tuple(origin)
