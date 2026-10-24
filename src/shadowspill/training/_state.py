"""JSON encoding for small loop state, without pickled code or model tensors."""

import base64
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def encode(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("loop state tensors must use ordinary CPU storage")
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        return [
            "tensor",
            str(value.dtype).removeprefix("torch."),
            list(value.shape),
            base64.b64encode(raw).decode("ascii"),
        ]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not supported in loop state")
        return [
            "array",
            value.dtype.str,
            list(value.shape),
            base64.b64encode(value.tobytes()).decode("ascii"),
        ]
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, Mapping):
        return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]
    if isinstance(value, (tuple, list)):
        return [
            "tuple" if isinstance(value, tuple) else "list",
            [encode(v) for v in value],
        ]
    raise TypeError(f"unsupported loop state value: {type(value).__name__}")


def decode(value: Any) -> Any:
    if not isinstance(value, list):
        if value is None or type(value) in (str, bool, int, float):
            return value
        raise ValueError("invalid loop state value")
    kind = value[0]
    if kind in {"dict", "tuple", "list"} and len(value) == 2:
        if kind == "dict":
            return {decode(k): decode(v) for k, v in value[1]}
        items = [decode(v) for v in value[1]]
        return tuple(items) if kind == "tuple" else items
    if kind in {"tensor", "array"} and len(value) == 4:
        _, name, shape, data = value
        raw = bytearray(base64.b64decode(data, validate=True))
        if kind == "tensor":
            dtype = getattr(torch, name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("invalid loop tensor dtype")
            if not raw:
                return torch.empty(0, dtype=dtype).reshape(shape)
            return torch.frombuffer(raw, dtype=dtype).reshape(shape)
        dtype = np.dtype(name)
        if dtype.hasobject:
            raise ValueError("object arrays are not supported in loop state")
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    raise ValueError("invalid loop state encoding")
