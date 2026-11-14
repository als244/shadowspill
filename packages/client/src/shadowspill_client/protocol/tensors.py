"""Portable tensor bytes and optional interoperability with already imported Torch."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .inputs import DTYPES


def storage_dtype(dtype: str) -> np.dtype[Any]:
    if dtype not in DTYPES:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    # BF16 stays bit-exact without requiring a NumPy extension or a GPU library.
    return np.dtype("uint16" if dtype == "bfloat16" else dtype).newbyteorder("<")


def tensor_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    if len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("tensor shape requires at most 32 nonnegative dimensions")
    return prod(shape) * storage_dtype(dtype).itemsize


@dataclass(frozen=True)
class TensorPayload:
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        size = tensor_bytes(self.dtype, self.shape)
        if type(self.data) is not bytes or len(self.data) != size:
            raise ValueError("tensor payload size differs from its shape and dtype")

    def array(self) -> NDArray[Any]:
        """Immutable NumPy view; BF16 exposes its uint16 bits explicitly."""
        return np.frombuffer(self.data, dtype=storage_dtype(self.dtype)).reshape(
            self.shape
        )


def _list_layout(value: Any, depth: int = 0) -> tuple[tuple[int, ...], str]:
    if depth > 32:
        raise ValueError("tensor nesting exceeds 32 dimensions")
    if isinstance(value, (list, tuple)):
        shape, kind = None, "bool"
        for child in value:
            child_shape, child_kind = _list_layout(child, depth + 1)
            if shape is not None and shape != child_shape:
                raise ValueError("tensor lists must be rectangular")
            shape = child_shape
            if child_kind == "float64" or (child_kind == "int64" and kind == "bool"):
                kind = child_kind
        return (len(value), *(shape or ())), kind if value else "float64"
    if type(value) is bool:
        return (), "bool"
    if type(value) is int:
        if not -(1 << 63) <= value < 1 << 63:
            raise ValueError("integer tensor values must fit int64")
        return (), "int64"
    if type(value) is float:
        return (), "float64"
    raise TypeError("tensor lists/scalars require booleans, integers or floats")


@dataclass(frozen=True)
class TensorSource:
    """Metadata-only view until staging space is reserved; never imports Torch."""

    value: Any
    dtype: str
    shape: tuple[int, ...]
    torch_tensor: bool = False

    def __post_init__(self) -> None:
        tensor_bytes(self.dtype, self.shape)

    @property
    def nbytes(self) -> int:
        return tensor_bytes(self.dtype, self.shape)

    @classmethod
    def inspect(cls, value: Any) -> TensorSource:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError("submission tensors require ordinary CPU storage")
            dtype = str(value.dtype).removeprefix("torch.")
            storage_dtype(dtype)
            return cls(value, dtype, tuple(value.shape), True)
        if isinstance(value, (np.ndarray, np.generic)):
            dtype = value.dtype.name
            storage_dtype(dtype)
            if value.dtype.hasobject:
                raise TypeError("object tensors are unsupported")
            return cls(value, dtype, tuple(value.shape))
        shape, dtype = _list_layout(value)
        return cls(value, dtype, shape)

    def copy(self) -> TensorPayload:
        value = self.value
        if self.torch_tensor:
            value = value.detach().resolve_conj().resolve_neg()
            if self.dtype == "bfloat16":
                torch = sys.modules["torch"]
                value = value.view(torch.uint16)
            value = value.numpy()
        array = np.asarray(value, dtype=storage_dtype(self.dtype))
        if tuple(array.shape) != self.shape:
            raise ValueError("input shape changed while taking its snapshot")
        return TensorPayload(self.dtype, self.shape, array.tobytes(order="C"))
