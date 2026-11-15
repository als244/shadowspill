"""Bounded checkpoint helpers over the public Executor tensor interface."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .load import load_checkpoint as load_checkpoint
    from .save import (
        save_checkpoint as save_checkpoint,
    )
    from .save import (
        write_checkpoint as write_checkpoint,
    )

__all__ = ["load_checkpoint", "save_checkpoint", "write_checkpoint"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = ".load" if name == "load_checkpoint" else ".save"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
