"""Validate a saved compiler device binding before GPU work begins."""

from .program import Program


def require_compiled_device(program: Program, device_index: int) -> None:
    index = program.bindings.get("compiled_device_index")
    if index is None:
        raise ValueError("Program has no recorded compiler device binding; rebuild it")
    if index != device_index:
        raise ValueError(
            f"Program was compiled for cuda:{index}, but this worker uses "
            f"cuda:{device_index}. Saved Inductor code cannot remap device indices. "
            "Use matching process-local CUDA numbering; CUDA_VISIBLE_DEVICES must "
            "be set before CUDA initialization"
        )
