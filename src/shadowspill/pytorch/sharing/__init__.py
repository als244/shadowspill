"""Cross-callable runtime-object references for the PyTorch frontend."""

from shadowspill.runtime import ObjectConsistency

from .bindings import (
    ResolvedSharedInput,
    ResolvedSharedOutput,
    resolve_shared_inputs,
    resolve_shared_outputs,
)
from .declarations import (
    SharedInput,
    SharedOutput,
    shared_input,
    shared_output,
)
from .paths import PathComponent, PytreePath, format_path, resolve_path
from .references import StateRef, TensorRef

__all__ = [
    "ObjectConsistency",
    "PathComponent",
    "PytreePath",
    "ResolvedSharedInput",
    "ResolvedSharedOutput",
    "SharedInput",
    "SharedOutput",
    "StateRef",
    "TensorRef",
    "format_path",
    "resolve_path",
    "resolve_shared_inputs",
    "resolve_shared_outputs",
    "shared_input",
    "shared_output",
]
