"""CPU-only planning from complete profile artifacts."""

from importlib import import_module
from typing import Any

_PUBLIC = {
    "MemoryBudgets": ".budgets",
    "plan_program": ".plan",
    "sweep_programs": ".sweep",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC[name], __name__), name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC)
