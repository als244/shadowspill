"""Portable provider configuration; importing descriptions never loads a backend."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from shadowspill.artifacts._values import freeze, plain
from shadowspill.parallelism import GroupRef, Parallelism


@dataclass(frozen=True)
class Resource:
    """An importable factory, JSON options and named logical group arguments.

    The factory receives ``name``, ``device``, ``groups`` and ``options`` as keyword
    arguments and returns an object with ``close()``. Operators refer to the stable
    resource name; live objects never enter a Program or Plan archive.
    """

    factory: str
    options: Mapping[str, Any] = field(default_factory=dict)
    groups: Mapping[str, GroupRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module, separator, function = self.factory.partition(":")
        if (
            not separator
            or module == "__main__"
            or not all(part.isidentifier() for part in module.split("."))
            or not function.isidentifier()
        ):
            raise ValueError("resource factory must be an importable 'module:function'")
        if any(
            not isinstance(name, str) or not name or not isinstance(group, GroupRef)
            for name, group in self.groups.items()
        ):
            raise TypeError("resource groups must map argument names to GroupRef")
        object.__setattr__(self, "options", freeze(self.options))
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "factory": self.factory,
            "options": plain(self.options),
            "groups": {
                name: {"name": group.name, "groups": [list(g) for g in group.groups]}
                for name, group in self.groups.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Resource":
        return cls(
            value["factory"],
            value.get("options", {}),
            {
                name: GroupRef(**group)
                for name, group in value.get("groups", {}).items()
            },
        )


def validate_resources(
    resources: Mapping[str, Resource], parallelism: Parallelism | None
) -> None:
    for name, resource in resources.items():
        if not isinstance(name, str) or not name or not isinstance(resource, Resource):
            raise TypeError(
                "resources must map nonempty names to Resource descriptions"
            )
        for group in resource.groups.values():
            if parallelism is None or parallelism.group(group.name) != group:
                raise ValueError(f"resource {name!r} uses a group outside parallelism")


def resource_groups(resources: Mapping[str, Resource]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            {
                g
                for r in resources.values()
                for ref in r.groups.values()
                for g in ref.groups
            }
        )
    )
