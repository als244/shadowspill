"""Portable configuration for a provider's build/export and load/bind lifecycle."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from shadowspill.artifacts._values import freeze, plain


@dataclass(frozen=True)
class OperatorPreparation:
    """An importable preparation factory and JSON options.

    The factory accepts ``phase`` (``build`` or ``load``), ``device`` and
    ``options``. A builder supplies ``capture()`` and ``export(captured)``;
    a loader supplies ``bind(manifest, read_blob)``. Both supply ``close()``.
    Capture surrounds the task's existing finalization invocation, never an
    additional execution. Export returns JSON metadata and digest-keyed immutable
    blobs. A bound task supplies ``activate()`` and cannot load/compile/tune.

    Preparation freezes selected implementations. It does not choose checkpoint
    cuts, execution regions, retention, chunk sizes or runtime residency.
    """

    factory: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.factory, str):
            raise TypeError("preparation factory must be a string")
        module, separator, function = self.factory.partition(":")
        if (
            not separator
            or module == "__main__"
            or not all(part.isidentifier() for part in module.split("."))
            or not function.isidentifier()
        ):
            raise ValueError(
                "preparation factory must be an importable 'module:function'"
            )
        if not isinstance(self.options, Mapping):
            raise TypeError("preparation options must be a mapping")
        object.__setattr__(self, "options", freeze(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {"factory": self.factory, "options": plain(self.options)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperatorPreparation":
        if set(value) != {"factory", "options"}:
            raise ValueError("invalid operator preparation description")
        return cls(value["factory"], value["options"])
