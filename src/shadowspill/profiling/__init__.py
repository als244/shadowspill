"""Phase exports, loaded only when requested during caller migration."""

from importlib import import_module
from typing import Any

_PUBLIC = {
    "derive_invariant_allocation_path": ".invariant",
    "CompiledManifestStore": ".manifest_store",
    "ProfilingMetadata": ".metadata",
    "ProfileStore": ".store",
    "apply_profiles": ".apply",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC[name], __name__), name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC)
