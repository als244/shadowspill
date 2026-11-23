"""Read the logical state inventory from existing capture/parallelism bindings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .component_layouts import saved_layouts
from .program import Program
from .state import StateBinding, StateId, StateInventory, StateTensor
from .tensors import tensor_geometry


def stored_gradients(program: Program) -> tuple[Mapping[str, Any], ...]:
    """The gradient bindings a Program keeps between calls.

    An accumulation's gradients: a backward Program's, and an optimizer-only
    Program's inputs. A complete step keeps none.
    """
    saved = program.bindings
    return tuple(
        saved.get(
            "persistent_gradients",
            saved.get("gradients", ())
            if saved.get("kind") == "forward_backward"
            else (),
        )
    )


def gradient_aliases(program: Program) -> frozenset[str]:
    """The alias groups holding the gradients ``program`` keeps between calls."""
    groups = {item.object_id: item.alias_group_id for item in program.objects}
    return frozenset(
        groups[item["gradient_object_id"]] for item in stored_gradients(program)
    )


def program_state(program: Program, logical_rank: int = 0) -> StateInventory:
    """Describe model state, masters, persistent gradients, moments and counters.

    Global/local shapes and ownership groups are retained from the generic saved
    parameter declarations. Physical tensor geometry remains separate, including
    wrapper components and shard padding. No dtype-width inference, model import,
    GPU initialization or alternate ownership resolver is involved.
    """
    world = 1 if program.parallelism is None else program.parallelism.world_size
    if type(logical_rank) is not int or not 0 <= logical_rank < world:
        raise ValueError("logical_rank is outside the saved Program")
    saved = program.bindings
    objects = {obj.object_id: obj for obj in program.objects}
    aliases = {alias.alias_group_id: alias for alias in program.alias_groups}
    geometry = tensor_geometry(program)
    component_layouts = saved_layouts(program)
    owners = {
        name: item
        for item in saved.get("parameter_ownership", ())
        for name in item["names"]
    }
    if any("component_layout" in owner for owner in owners.values()):
        raise ValueError("parameter ownership cannot declare component coordinates")
    tensors: list[StateTensor] = []
    bindings: list[StateBinding] = []

    def add(
        identity: StateId,
        object_id: str | None,
        *,
        parameter_name: str | None = None,
        host_binding: str | None = None,
        layout: Mapping[str, Any] | None = None,
        checkpoint: bool = True,
    ) -> None:
        source = layout if layout is not None else geometry.get(object_id or "")
        if source is None or not {"dtype", "shape", "stride"}.issubset(source):
            raise ValueError(f"state has no portable tensor geometry: {identity}")
        # Task/profile metadata may carry additional properties; only tensor
        # layout defines state compatibility across operation forms.
        view = {key: source[key] for key in ("dtype", "shape", "stride")}
        view["storage_offset"] = source.get("storage_offset", 0)
        owner = owners.get(parameter_name or "", {})
        component = (
            component_layouts.get((identity.name, identity.component_path))
            if identity.role == "parameter"
            else None
        )
        if component is not None:
            owner = {
                **owner,
                "component_layout": {
                    "axes": component.axes,
                    "encoding": component.encoding,
                    **(
                        {"blocks": component.blocks}
                        if any(size != 1 for size in component.blocks)
                        else {}
                    ),
                },
            }
        tensors.append(
            StateTensor(
                identity,
                view,
                parameter_name,
                owner,
                checkpoint,
                saved.get("optimizer_type")
                if identity.role in ("master", "optimizer")
                else None,
            )
        )
        if object_id is None:
            bindings.append(StateBinding(identity, host_binding=host_binding))
        else:
            obj = objects[object_id]
            alias = aliases[obj.alias_group_id]
            storage = {
                "device_id": alias.device_id,
                "object_offset_bytes": obj.offset_bytes,
                "object_size_bytes": obj.size_bytes,
                "size_bytes": alias.size_bytes,
                "spill_size_bytes": alias.spill_size_bytes,
                "transfer": None
                if alias.transfer is None
                else alias.transfer.to_dict(),
            }
            bindings.append(
                StateBinding(identity, object_id, obj.alias_group_id, storage=storage)
            )

    for item in saved.get("registrations", ()):
        identity = StateId(
            "parameter" if item["parameter"] else "buffer",
            item["name"],
            tuple(item.get("component_path", ())),
        )
        add(
            identity,
            item["object_id"],
            parameter_name=item["name"],
            checkpoint=item.get("persistent", True),
        )
    optimizer = saved.get("optimizer_bindings", ())
    mastered = {
        item["parameter_name"] for item in optimizer if item["role"] == "compute_copy"
    }
    for item in optimizer:
        role = item["role"]
        if role == "parameter" and item["parameter_name"] in mastered:
            identity = StateId(
                "master", item["parameter_name"], tuple(item.get("component_path", ()))
            )
        elif role == "state":
            identity = StateId(
                "optimizer", item["name"], tuple(item.get("component_path", ()))
            )
        else:
            continue  # Controls are call-local, compute copies already registered.
        add(
            identity,
            item["object_id"],
            parameter_name=item.get("parameter_name"),
            host_binding=item["name"] if item["object_id"] is None else None,
            layout=item if item["object_id"] is None else None,
        )
    for item in stored_gradients(program):
        identity = StateId(
            "gradient", item["parameter_name"], tuple(item.get("component_path", ()))
        )
        add(
            identity,
            item["gradient_object_id"],
            parameter_name=item["parameter_name"],
            checkpoint=False,
        )
    for item in saved.get("registered_state", ()):
        identity = StateId.from_dict(item["identity"])
        if identity.role not in ("registered", "rng"):
            raise ValueError("extension state must declare registered or rng role")
        add(
            identity,
            item.get("object_id"),
            host_binding=item.get("host_binding"),
            layout=item.get("geometry"),
            checkpoint=item.get("checkpoint", True),
        )
    return StateInventory(logical_rank, tuple(tensors), tuple(bindings))
