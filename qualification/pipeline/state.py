"""Bounded initialization and reference-compatible snapshots; no live model."""

from pathlib import Path
from typing import Any, cast

import torch

from shadowspill import Executor
from shadowspill.execution.tensors import TensorChunk


def load_inputs(directory: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        torch.load(
            directory / "inputs.pt", map_location="cpu", weights_only=True, mmap=True
        ),
    )


def initialize(executor: Executor, inputs: dict[str, Any]) -> None:
    def fill(chunk: TensorChunk, destination: torch.Tensor) -> None:
        name = chunk.name.removeprefix("master.").removeprefix("compute.")
        tensor = inputs["tensors"][name]
        elements = tensor.untyped_storage().nbytes() // tensor.element_size()
        raw = tensor.as_strided((elements,), (1,), 0)
        destination.copy_(raw[chunk.offset : chunk.offset + destination.numel()])

    executor.initialize(fill)


def snapshot(executor: Executor, inputs: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct PyTorch's comparison format from public bounded tensor export."""
    tensors = {}

    def collect(chunk: TensorChunk, value: torch.Tensor) -> None:
        if chunk.name not in tensors:
            owner = torch.empty(chunk.storage_elements, dtype=value.dtype, device="cpu")
            tensors[chunk.name] = (owner, chunk)
        tensors[chunk.name][0][chunk.offset : chunk.offset + value.numel()].copy_(value)

    executor.export_tensors(collect)
    values = {
        name: owner.as_strided(chunk.shape, chunk.stride, chunk.storage_offset)
        for name, (owner, chunk) in tensors.items()
    }
    # Tied names are allowed by the model state dict; export emits each storage once.
    by_storage = {}
    model = {}
    for name in inputs["model_keys"]:
        source = inputs["tensors"][name]
        key = source.untyped_storage()._cdata
        value = values.get("master." + name, values.get(name))
        if value is not None:
            by_storage[key] = value
            model[name] = value
    for name in inputs["model_keys"]:
        if name not in model:
            source = inputs["tensors"][name]
            model[name] = by_storage[source.untyped_storage()._cdata].as_strided(
                source.shape, source.stride(), source.storage_offset()
            )
    state = {}
    for index, name in enumerate(inputs["parameter_names"]):
        prefix = f"optimizer.{name}."
        entries = {
            key[len(prefix) :]: value
            for key, value in values.items()
            if key.startswith(prefix)
        }
        if entries:
            state[index] = entries
    return {
        "model": model,
        "optimizer": {"state": state, "param_groups": inputs["param_groups"]},
    }
