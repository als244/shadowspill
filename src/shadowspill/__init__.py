"""Framework-neutral memory planning and transparent execution support."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("shadowspill")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.1.0.dev0"

_PUBLIC = {
    "Program": "artifacts",
    "Profiles": "artifacts",
    "ProfiledProgram": "artifacts",
    "Plan": "artifacts",
    "TaskProfile": "artifacts",
    "SweepResult": "artifacts",
    "apply_profiles": "profiling.apply",
    "MemoryBudgets": "planning.budgets",
    "CommunicationInfo": "communication",
    "plan_program": "planning.plan",
    "sweep_programs": "planning.sweep",
    "write_report": "reporting",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_PUBLIC[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = ["__version__", *_PUBLIC]
