"""PyTorch view metadata layered over neutral runtime objects."""

from __future__ import annotations

import builtins
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from shadowspill.runtime import ObjectRef


def _addressable_elements(
    shape: tuple[int, ...], stride: tuple[int, ...], storage_offset: int
) -> int:
    if len(shape) != len(stride):
        raise ValueError("tensor reference shape and stride ranks differ")
    if storage_offset < 0:
        raise ValueError("tensor reference storage offset must be non-negative")
    if any(dimension < 0 for dimension in shape):
        raise ValueError("tensor reference dimensions must be non-negative")
    if any(value < 0 for value in stride):
        raise ValueError("negative tensor strides are not supported")
    if any(dimension == 0 for dimension in shape):
        return storage_offset
    maximum = storage_offset
    for dimension, value in zip(shape, stride, strict=True):
        maximum += (dimension - 1) * value
    return maximum + 1


@dataclass(frozen=True, slots=True)
class TensorRef:
    """One PyTorch tensor view of a runtime-owned storage root."""

    object: ObjectRef
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int = 0
    requires_grad: bool = False
    generation: int = 0
    retained_pools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.object._require_open()
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("tensor reference dtype must be torch.dtype")
        if self.generation < 0:
            raise ValueError("tensor reference generation must be non-negative")
        if any(not isinstance(pool, str) or not pool for pool in self.retained_pools):
            raise ValueError("tensor reference pool names must be non-empty strings")
        if len(self.retained_pools) != len(set(self.retained_pools)):
            raise ValueError("tensor reference pool names must be unique")
        extent = (
            _addressable_elements(self.shape, self.stride, self.storage_offset)
            * torch.empty((), dtype=self.dtype).element_size()
        )
        if extent > self.object.size_bytes:
            raise ValueError(
                "tensor view exceeds its runtime object: "
                f"required={extent}, object={self.object.size_bytes}"
            )

    @classmethod
    def from_tensor(
        cls,
        object: ObjectRef,
        tensor: torch.Tensor,
        *,
        generation: int = 0,
        retained_pools: tuple[str, ...] = (),
    ) -> TensorRef:
        """Capture public view metadata without retaining a tensor address."""

        return cls(
            object=object,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            requires_grad=bool(tensor.requires_grad),
            generation=generation,
            retained_pools=retained_pools,
        )

    @property
    def closed(self) -> bool:
        """Whether the underlying runtime-object reference was released."""

        return self.object.closed

    def close(self) -> None:
        """Release this view's runtime-object ownership."""

        self.object.close()

    def __enter__(self) -> TensorRef:
        self.object._require_open()
        return self

    def __exit__(self, *exception: builtins.object) -> None:
        del exception
        self.close()


class StateRef(Mapping[str, TensorRef]):
    """An immutable named collection of runtime-backed tensor views."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, TensorRef]) -> None:
        copied = dict(values)
        if any(not isinstance(name, str) or not name for name in copied):
            raise ValueError("state reference names must be non-empty strings")
        if any(not isinstance(value, TensorRef) for value in copied.values()):
            raise TypeError("state reference values must be TensorRef instances")
        self._values = MappingProxyType(copied)

    def __getitem__(self, key: str) -> TensorRef:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"StateRef({dict(self._values)!r})"

    @property
    def closed(self) -> bool:
        """Whether every distinct runtime-object reference was released."""

        return all(value.closed for value in self._values.values())

    def close(self) -> None:
        """Release every distinct storage root represented by this state."""

        seen: set[int] = set()
        for value in self._values.values():
            identity = id(value.object)
            if identity in seen:
                continue
            seen.add(identity)
            value.close()

    def __enter__(self) -> StateRef:
        for value in self._values.values():
            value.object._require_open()
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()


__all__ = ["StateRef", "TensorRef"]
