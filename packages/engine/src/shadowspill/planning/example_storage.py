"""Clone private example alias closures while retaining canonical shared state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import count
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.ir import AliasGroupSpec, ObjectSpec, ResidencySpec

_OBJECT_FIELDS = {
    "object_id",
    "source_object_id",
    "destination_object_id",
    "source",
    "stored",
    "padding",
    "destination",
    "mutations",
}


def references(value: Any, identities: set[str], *, key: str = "") -> set[str]:
    if isinstance(value, str):
        return (
            {value} & identities if key in _OBJECT_FIELDS | {"saved_aliases"} else set()
        )
    if isinstance(value, Mapping):
        return set().union(
            *(references(child, identities, key=name) for name, child in value.items())
        )
    if isinstance(value, (tuple, list)):
        return set().union(*(references(child, identities, key=key) for child in value))
    return set()


def remap(
    value: Any, objects: Mapping[str, str], aliases: Mapping[str, str], *, key: str = ""
) -> Any:
    """Rewrite only documented identity fields, never profile keys or literals."""
    if isinstance(value, Mapping):
        return {
            name: remap(child, objects, aliases, key=name)
            for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        if key == "saved_aliases":
            return [aliases.get(item, item) for item in value]
        if key == "mutations":
            return [objects.get(item, item) for item in value]
        return [remap(child, objects, aliases, key=key) for child in value]
    if key in _OBJECT_FIELDS and isinstance(value, str):
        return objects.get(value, value)
    return value


def identities(existing: set[str], prefix: str) -> Iterator[str]:
    # Native execution binds the numeric suffix of canonical plan-local IDs.
    start = 1 + max(
        (
            int(name.removeprefix(prefix))
            for name in existing
            if name.startswith(prefix) and name.removeprefix(prefix).isdigit()
        ),
        default=-1,
    )
    return (f"{prefix}{value:06d}" for value in count(start))


@dataclass
class StorageCopies:
    objects: list[ObjectSpec]
    aliases: list[AliasGroupSpec]
    initial: list[ResidencySpec]
    final: list[ResidencySpec]
    sources: dict[str, Any]
    object_maps: list[dict[str, str]]
    alias_maps: list[dict[str, str]]


def clone_storage(
    program: Program, shared: set[str], private_groups: Sequence[set[str]]
) -> StorageCopies:
    """Every selected occurrence owns all views of its private aliases."""
    objects = {obj.object_id: obj for obj in program.objects}
    aliases = {alias.alias_group_id: alias for alias in program.alias_groups}
    by_alias: dict[str, list[str]] = {}
    for obj in program.objects:
        by_alias.setdefault(obj.alias_group_id, []).append(obj.object_id)
    result = StorageCopies(
        [obj for obj in program.objects if obj.alias_group_id in shared],
        [alias for alias in program.alias_groups if alias.alias_group_id in shared],
        [item for item in program.initial_residency if item.alias_group_id in shared],
        [item for item in program.final_residency if item.alias_group_id in shared],
        {
            alias: plain(names)
            for alias, names in program.sources.items()
            if alias in shared
        },
        [],
        [],
    )
    new_aliases, new_objects = (
        identities(set(aliases), "alias_"),
        identities(set(objects), "object_"),
    )
    for private in private_groups:
        alias_map = {alias: next(new_aliases) for alias in sorted(private)}
        object_map = {
            value: next(new_objects)
            for value in sorted(value for alias in private for value in by_alias[alias])
        }
        result.alias_maps.append(alias_map)
        result.object_maps.append(object_map)
        result.aliases.extend(
            replace(aliases[alias], alias_group_id=alias_map[alias])
            for alias in sorted(private)
        )
        result.objects.extend(
            replace(
                objects[value],
                object_id=mapped,
                alias_group_id=alias_map[objects[value].alias_group_id],
            )
            for value, mapped in sorted(object_map.items())
        )
        result.sources.update(
            (updated, plain(program.sources[original]))
            for original, updated in alias_map.items()
            if original in program.sources
        )
        for entries, target in (
            (program.initial_residency, result.initial),
            (program.final_residency, result.final),
        ):
            target.extend(
                replace(item, alias_group_id=alias_map[item.alias_group_id])
                for item in entries
                if item.alias_group_id in private
            )
    return result
