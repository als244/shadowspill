"""Recreate provider resources once, closing them before their process groups."""

from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

import torch

from shadowspill.artifacts._values import plain
from shadowspill.communication.binding.groups import GroupBindings

from .description import Resource
from .factories import factory


class ResourceBindings:
    def __init__(self) -> None:
        self._stack = ExitStack()

    def open(
        self,
        resources: Mapping[str, Resource],
        sources: Mapping[str, str],
        *,
        device: torch.device,
        groups: GroupBindings | None = None,
    ) -> None:
        try:
            for name, description in sorted(resources.items()):
                if description.groups and groups is None:
                    raise ValueError(f"resource {name!r} requires bound logical groups")
                initialize = factory(description, sources[name])
                arguments: dict[str, Any] = {
                    key: groups.resolve(reference)
                    for key, reference in description.groups.items()
                    if groups is not None
                }
                value = initialize(
                    name=name,
                    device=device,
                    groups=arguments,
                    options=plain(description.options),
                )
                if not callable(getattr(value, "close", None)):
                    raise TypeError(
                        f"resource {name!r} must return an object with close()"
                    )
                self._stack.callback(value.close)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._stack.close()
