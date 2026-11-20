"""Portable configuration for a provider's build/export and load/bind lifecycle."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from shadowspill.artifacts._values import freeze, plain


@dataclass(frozen=True)
class OperatorPreparation:
    """An adapter that saves and restores the external kernels tasks select.

    The builders take a sequence of them as ``operator_preparation``. The
    factory is called with ``phase`` (``"build"`` or ``"load"``), ``device``
    and ``options``. At build it returns an object with ``capture()``, which
    surrounds each task's existing finalization run (never an extra run), and
    ``export(captured)``, which returns JSON metadata and immutable blobs keyed
    by digest. At load it returns one with ``bind(manifest, read_blob)``,
    whose bound task has ``activate()`` and must not load, compile or tune.
    Both have ``close()``. An adapter freezes the selected implementations
    only: it chooses no stages, regions, retention, chunk sizes or residency.

    Parameters
    ----------
    factory : str
        The factory, as an importable ``"module:function"`` outside
        ``__main__``.
    options : mapping, default {}
        JSON options passed to the factory.

    Raises
    ------
    TypeError
        If ``factory`` is not a string or ``options`` is not a mapping.
    ValueError
        If ``factory`` is not an importable ``"module:function"``.
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
        """Return the adapter as plain data.

        Returns
        -------
        dict
            ``factory`` and ``options``.
        """
        return {"factory": self.factory, "options": plain(self.options)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperatorPreparation":
        """Rebuild an adapter from a `to_dict` record.

        Parameters
        ----------
        value : mapping
            Exactly ``factory`` and ``options``.

        Returns
        -------
        OperatorPreparation
            The adapter.

        Raises
        ------
        ValueError
            If ``value`` has other keys, or the factory is invalid.
        """
        if set(value) != {"factory", "options"}:
            raise ValueError("invalid operator preparation description")
        return cls(value["factory"], value["options"])
