"""Load explicitly configured installed components in controllers and workers."""

import re
from collections.abc import Sequence
from importlib.metadata import EntryPoint, PackageNotFoundError, distribution

from .registry import Registry

ENTRY_POINT_GROUP = "shadowspill.components"


def load_registry(packages: Sequence[str]) -> Registry:
    """Call installed ``register(registry)`` hooks from server configuration.

    Package names belong to trusted deployment configuration, never a model or
    compute request. Registration hooks must remain CPU descriptions and supply
    their source closures through the ordinary Registry API. Workers resolve the
    same saved setup afterward, rechecking those exact component sources.
    """
    if isinstance(packages, (str, bytes)):
        raise TypeError("component packages must be a sequence of distribution names")
    names = tuple(_name(name) for name in packages)
    if len(names) != len(set(names)):
        raise ValueError("component packages contain duplicate distribution names")
    hooks: list[tuple[str, EntryPoint]] = []
    # Resolve the complete allowlist before importing any registration hook.
    for name in names:
        try:
            installed = distribution(name)
        except PackageNotFoundError as error:
            raise ValueError(f"component package is not installed: {name}") from error
        entries = sorted(
            (
                entry
                for entry in installed.entry_points
                if entry.group == ENTRY_POINT_GROUP
            ),
            key=lambda entry: entry.name,
        )
        if not entries or len({entry.name for entry in entries}) != len(entries):
            raise ValueError(
                f"{name} must publish distinct {ENTRY_POINT_GROUP} entry points"
            )
        hooks.extend((name, entry) for entry in entries)
    registry = Registry()
    for name, entry in hooks:
        register = entry.load()
        if not callable(register):
            raise TypeError(f"{name}:{entry.name} must be a registration function")
        result = register(registry)
        if result is not None:
            raise TypeError(f"{name}:{entry.name} must register components in place")
    return registry


def _name(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", value) is None
    ):
        raise ValueError("component packages must name installed distributions")
    return re.sub(r"[-_.]+", "-", value).lower()
