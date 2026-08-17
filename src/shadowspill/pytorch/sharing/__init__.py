"""Cross-callable runtime-object references for the PyTorch frontend."""

from .bindings import ResolvedSharedOutput, resolve_shared_outputs
from .declarations import (
    SharedConsistency,
    SharedInput,
    SharedOutput,
    shared_input,
    shared_output,
)
from .paths import PathComponent, PytreePath, format_path, resolve_path
from .references import StateRef, TensorRef

__all__ = [
    "PathComponent",
    "PytreePath",
    "ResolvedSharedOutput",
    "SharedConsistency",
    "SharedInput",
    "SharedOutput",
    "StateRef",
    "TensorRef",
    "format_path",
    "resolve_path",
    "resolve_shared_outputs",
    "shared_input",
    "shared_output",
]
