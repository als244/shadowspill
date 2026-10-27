"""GPU measurement and CPU profile application, loaded on demand."""

from importlib import import_module
from typing import Any

_PUBLIC = {
    "apply_profiles": ".apply",
    "measure_program": ".measure",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC[name], __name__), name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC)
