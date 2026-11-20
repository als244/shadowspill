"""The records an extension package declares for the components it adds.

A component describes its inputs, controls, outputs and loss participation with
these records, and names its source with `ComponentSource`. The service checks
requests and data against them; the client needs nothing else from an
extension to train with it.
"""

from .protocol.components import (
    ComponentSource,
    OutputSet,
    Participation,
    TensorOutput,
)
from .protocol.controls import ControlSchema, ScalarControl
from .protocol.inputs import Capacity, InputSchema, TensorField

__all__ = [
    "Capacity",
    "ComponentSource",
    "ControlSchema",
    "InputSchema",
    "OutputSet",
    "Participation",
    "ScalarControl",
    "TensorField",
    "TensorOutput",
]
