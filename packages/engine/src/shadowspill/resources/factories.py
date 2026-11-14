"""Verify provider initialization code only when entering a GPU phase."""

from collections.abc import Callable, Mapping
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

from .description import Resource


def factory(resource: Resource, expected: str | None = None) -> Callable[..., Any]:
    module_name, name = resource.factory.split(":")
    module = import_module(module_name)
    if expected is not None and source_digest(resource) != expected:
        raise ValueError(f"resource factory source differs: {resource.factory}")
    value = getattr(module, name)
    if not callable(value):
        raise TypeError(f"resource factory is not callable: {resource.factory}")
    return value  # type: ignore[no-any-return]


def source_digest(resource: Resource) -> str:
    module = import_module(resource.factory.split(":")[0])
    if module.__file__ is None:
        raise ValueError(f"resource factory has no source: {resource.factory}")
    return sha256(Path(module.__file__).read_bytes()).hexdigest()


def sources(resources: Mapping[str, Resource]) -> dict[str, str]:
    return {name: source_digest(value) for name, value in resources.items()}
