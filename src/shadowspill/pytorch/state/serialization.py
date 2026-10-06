"""Store tensor representations as ordinary tensors in weights-only checkpoints.

The receiving model supplies the wrapper class and reconstruction metadata.
Checkpoint files contain component tensors and a description to validate, never
an executable wrapper or quantizer object.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from typing import Any

import torch

from shadowspill.pytorch.representations import (
    is_wrapper,
    map_tensor,
    tensor_components,
)


def _description(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach()
    result: dict[str, Any] = {"shape": tuple(value.shape), "dtype": str(value.dtype)}
    if is_wrapper(value):
        names, metadata = value.__tensor_flatten__()
        result.update(
            type=f"{type(value).__module__}.{type(value).__qualname__}",
            metadata=hashlib.sha256(pickle.dumps(metadata, protocol=5)).hexdigest(),
            components={name: _description(getattr(value, name)) for name in names},
        )
    return result


def encode_tensor_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve component bytes without adding a dense logical copy."""
    return {
        name: (
            {
                "__tensor_components__": _description(value),
                "tensors": dict(tensor_components(value)),
            }
            if isinstance(value, torch.Tensor) and is_wrapper(value)
            else value
        )
        for name, value in state.items()
    }


def decode_tensor_state(
    state: Mapping[str, Any], templates: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Reconstruct only types and metadata already present in the caller's model."""
    result = dict(state)
    for name, value in state.items():
        if not isinstance(value, Mapping) or "__tensor_components__" not in value:
            continue
        template = templates.get(name)
        if (
            template is None
            or not is_wrapper(template)
            or value["__tensor_components__"] != _description(template)
        ):
            raise ValueError(f"checkpoint tensor representation differs for {name!r}")
        components = dict(tensor_components(template))
        saved = value.get("tensors")
        if not isinstance(saved, Mapping) or set(saved) != set(components):
            raise ValueError(f"checkpoint tensor components differ for {name!r}")
        replacements = {}
        for path, component in components.items():
            source = saved[path]
            if (
                not isinstance(source, torch.Tensor)
                or is_wrapper(source)
                or tuple(source.shape) != tuple(component.shape)
                or source.dtype != component.dtype
            ):
                raise ValueError(
                    f"checkpoint tensor component differs for {name!r}: {path}"
                )
            replacements[id(component)] = source
        result[name] = map_tensor(
            template, lambda tensor, values=replacements: values[id(tensor)]
        )
    return result
