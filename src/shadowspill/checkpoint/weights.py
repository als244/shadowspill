"""Choose an authoritative weight copy using captured ownership, never names alone."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from shadowspill.artifacts.tensors import registration_name

if TYPE_CHECKING:
    from shadowspill.artifacts.tensors import TensorChunk
    from shadowspill.execution import Executor

type WeightRepresentation = Literal["master", "compute"]


@dataclass(frozen=True)
class WeightSelection:
    derived: dict[str, str]
    retained: dict[str, str]


def _length(chunk: TensorChunk) -> int:
    return (
        chunk.storage_elements
        if chunk.region_elements is None
        else chunk.region_elements
    )


def select_weights(executor: Executor, weights: object) -> WeightSelection:
    """Omit a copy only when this rank can reconstruct its complete owned region.

    Encoded tensor components and incompatible layouts retain their physical
    state. Their conversion is not an ordinary dtype cast; the manifest reports
    these exceptions rather than silently treating a payload as a logical weight.
    """
    if not isinstance(weights, str) or weights not in {"master", "compute"}:
        raise ValueError("weights must be 'master' or 'compute'")
    chunks = {chunk.name: chunk for chunk in executor.tensor_chunks()}
    derived: dict[str, str] = {}
    retained: dict[str, str] = {}
    for prefix, plan in (
        ("", executor.plan),
        *((f"plans.{name}.", plan) for name, plan in executor.plans.items()),
    ):
        program = plan.program
        bindings = getattr(program, "bindings", {})
        copies: dict[str, list[dict[str, Any]]] = defaultdict(list)
        masters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for binding in bindings.get("optimizer_bindings", ()):
            if binding["role"] == "compute_copy":
                copies[binding["parameter_name"]].append(binding)
            elif binding["role"] == "parameter":
                masters[binding["parameter_name"]].append(binding)
        if not copies:
            continue
        aliases = {value.object_id: value.alias_group_id for value in program.objects}
        registrations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        names: dict[str, str] = {}
        for item in bindings.get("registrations", ()):
            alias = aliases[item["object_id"]]
            registrations[alias].append(item)
            name = prefix + registration_name(item)
            if name in chunks:
                names.setdefault(alias, name)
        for parameter, computed in copies.items():
            label = prefix + parameter
            owned = masters.get(parameter, ())
            if len(computed) != 1 or len(owned) != 1:
                retained[label] = "encoded or multiple physical weight components"
                continue
            compute, master = computed[0], owned[0]
            compute_alias = aliases.get(compute["object_id"])
            leaves = registrations.get(compute_alias, ()) if compute_alias else ()
            if (
                not leaves
                or compute.get("component_path")
                or master.get("component_path")
                or any(item.get("component_path") for item in leaves)
                or (master.get("initialization") or {}).get("kind") != "parameter"
            ):
                retained[label] = "no ordinary scalar-cast reconstruction contract"
                continue
            compute_name = names.get(compute_alias) if compute_alias else None
            master_name = prefix + "master." + master["name"]
            if compute_name is None or master_name not in chunks:
                retained[label] = "copies do not have independent owned storage here"
                continue
            source, target = (
                (chunks[master_name], chunks[compute_name])
                if weights == "master"
                else (chunks[compute_name], chunks[master_name])
            )
            if not _length(target):
                continue
            if (
                not source.dtype.is_floating_point
                or not target.dtype.is_floating_point
                or any(
                    getattr(source, key) != getattr(target, key)
                    for key in ("shape", "stride", "storage_offset", "storage_elements")
                )
            ):
                retained[label] = "physical weight layouts cannot use a storage cast"
                continue
            if (
                source.region_offset > target.region_offset
                or source.region_offset + _length(source)
                < target.region_offset + _length(target)
            ):
                retained[label] = "selected copy does not cover this rank's other copy"
                continue
            if target.name in derived and derived[target.name] != source.name:
                raise ValueError(f"ambiguous checkpoint weight source: {target.name}")
            derived[target.name] = source.name
    if set(derived) & set(derived.values()):
        raise ValueError("checkpoint weight reconstruction cannot contain chains")
    return WeightSelection(derived, retained)
