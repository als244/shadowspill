"""Read per-example bindings from saved forward and training Programs."""

from collections.abc import Mapping, Sequence
from typing import Any


def single_forward(bindings: Mapping[str, Any]) -> bool:
    """Unexpanded forward calls retain their ordinary single-result ABI."""
    return bindings.get("kind") == "forward" and "forward_templates" not in bindings


def input_groups(bindings: Mapping[str, Any]) -> Sequence[Any]:
    roots = bindings["root_inputs"]
    return (roots,) if single_forward(bindings) else roots


def output_groups(bindings: Mapping[str, Any]) -> Sequence[Sequence[str | None]]:
    outputs = bindings["public_outputs"]
    return (outputs,) if single_forward(bindings) else outputs
