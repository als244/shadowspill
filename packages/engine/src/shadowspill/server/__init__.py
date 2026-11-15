"""Allocation service records and transport boundaries."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .allocation import Allocation as Allocation
    from .deployment import ServerLimits as ServerLimits
    from .service import Server as Server

__all__ = ["Allocation", "Server", "ServerLimits"]


def __getattr__(name: str) -> Any:
    modules = {
        "Allocation": ".allocation",
        "Server": ".service",
        "ServerLimits": ".deployment",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
