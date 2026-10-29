"""What the ranks of a distributed Program must agree on, whatever each built.

Ranks exchange weight shards, gradients and optimizer updates through the
collectives their Program lowers from its parallelism and shard layout. Two
Programs can take part in one group when they agree on that exchange: the
parallelism, which tensors exist, how they are placed and sharded, and their
geometry. Executables, profiles, budgets, the build's target and the number
of examples a rank accumulates are each rank's own, so a group may hold
Programs built for different hardware.
"""

from __future__ import annotations

from typing import Any

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tensors import registration_name


def shared_structure(program: Program) -> dict[str, Any]:
    """The exchange a Program takes part in, keyed by tensor names rather
    than the object identities a build assigns."""

    bindings = program.bindings
    aliases = {item.object_id: item.alias_group_id for item in program.objects}
    names: dict[str, list[str]] = {}
    for item in bindings.get("registrations", ()):
        names.setdefault(item["object_id"], []).append(registration_name(item))
    alias_names: dict[str, list[str]] = {}
    for object_id, found in names.items():
        alias_names.setdefault(aliases[object_id], []).extend(found)
    gradient_names = {
        aliases.get(item["gradient_object_id"], item["gradient_object_id"]): item[
            "parameter_name"
        ]
        for item in bindings.get("gradients", ())
        if "gradient_object_id" in item
    }
    geometry = bindings.get("model_tensor_geometry", {})
    gradient_geometry = bindings.get("gradient_geometry", {})
    return {
        "parallelism": None
        if program.parallelism is None
        else program.parallelism.to_dict(),
        "kind": bindings.get("kind"),
        "lowered": bindings.get("distributed_training_lowered"),
        "ownership": plain(bindings.get("parameter_ownership", [])),
        "tensors": sorted(
            (
                registration_name(item),
                bool(item.get("parameter")),
                item.get("trainable"),
            )
            for item in bindings.get("registrations", ())
        ),
        "geometry": {
            name: plain(geometry[object_id])
            for object_id, found in names.items()
            if object_id in geometry
            for name in found
        },
        "weight_shards": {
            " ".join(sorted(alias_names.get(alias, [alias]))): plain(record)
            for alias, record in bindings.get("weight_shards", {}).items()
        },
        "gradient_shards": {
            gradient_names.get(alias, alias): plain(record)
            for alias, record in bindings.get("gradient_shards", {}).items()
        },
        "gradient_geometry": {
            item["parameter_name"]: plain(gradient_geometry[item["gradient_object_id"]])
            for item in bindings.get("gradients", ())
            if item.get("gradient_object_id") in gradient_geometry
        },
    }


def agreement_digest(program: Program) -> str:
    """The digest every rank of a group must share."""

    return digest(shared_structure(program))
