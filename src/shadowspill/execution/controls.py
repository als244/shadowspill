"""Bounded CPU controls and per-invocation tensor hyperparameters."""

from collections.abc import Callable, Iterator, Mapping
from math import prod
from typing import Any

import torch

from .buffer_controls import BufferControls
from .storage import Storage, dtype_of
from .tensor_spans import spans
from .tensors import TensorChunk


class Controls:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.bindings = {
            b["name"]: b
            for b in storage.plan.program.bindings.get("optimizer_bindings", ())
            if b["object_id"] is None or b["role"] == "hyperparameter"
        }
        self.buffers = BufferControls(storage)
        if self.buffers.bindings.keys() & self.bindings.keys():
            raise ValueError(
                "model buffer names must not collide with optimizer controls"
            )
        self.host = {}
        for name, binding in self.bindings.items():
            if binding["object_id"] is not None:
                continue
            extent = 1 + sum(
                (n - 1) * stride
                for n, stride in zip(binding["shape"], binding["stride"], strict=True)
            )
            if binding["device"] != "cpu" or max(prod(binding["shape"]), extent) > 4096:
                raise ValueError(
                    f"unplanned control {name} must be a bounded CPU tensor"
                )
            self.host[name] = torch.empty_strided(
                binding["shape"],
                binding["stride"],
                dtype=dtype_of(binding),
                device="cpu",
            )

    def initialize(
        self,
        fill_tensor: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
        restore: bool = False,
    ) -> None:
        for name, value in self.host.items():
            initializer = self.bindings[name]["initialization"]
            if (
                not restore
                and initializer
                and initializer["kind"] in {"constant", "value"}
            ):
                value.copy_(torch.as_tensor(initializer["value"], dtype=value.dtype))
            else:
                for request, chunk in self._chunks(name, value, chunk_bytes):
                    fill_tensor(request, chunk)

    def export(
        self,
        callback: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
    ) -> None:
        for name, value in self.host.items():
            for request, chunk in self._chunks(name, value, chunk_bytes):
                callback(request, chunk)

    @staticmethod
    def _chunks(
        name: str,
        value: torch.Tensor,
        chunk_bytes: int,
    ) -> Iterator[tuple[TensorChunk, torch.Tensor]]:
        count = value.untyped_storage().nbytes() // value.element_size()
        raw = value.as_strided((count,), (1,), 0)
        width = max(1, chunk_bytes // value.element_size())
        for start in range(0, count, width):
            yield (
                TensorChunk(
                    name,
                    tuple(value.shape),
                    value.dtype,
                    start,
                    count,
                    tuple(value.stride()),
                    int(value.storage_offset()),
                ),
                raw[start : start + width],
            )

    def apply(self, values: Mapping[str, Any]) -> None:
        unknown = (
            set(values)
            - {
                name
                for name, b in self.bindings.items()
                if b["role"] == "hyperparameter"
            }
            - self.buffers.bindings.keys()
        )
        if unknown:
            raise ValueError(f"unknown runtime hyperparameters: {sorted(unknown)}")
        buffers = self.buffers.prepare(values)
        prepared = []
        for name, binding in self.bindings.items():
            if binding["role"] != "hyperparameter":
                continue
            initial = binding["initialization"]
            supplied = values.get(name, None)
            if name not in values:
                if not initial or initial["kind"] not in {"value", "constant"}:
                    raise ValueError(f"runtime hyperparameter {name} needs a value")
                supplied = initial["value"]
            if isinstance(supplied, torch.Tensor) and supplied.device.type != "cpu":
                raise ValueError("hyperparameters must be CPU values")
            value = torch.as_tensor(supplied, dtype=dtype_of(binding))
            if tuple(value.shape) != tuple(binding["shape"]):
                raise ValueError(f"runtime hyperparameter {name} has the wrong shape")
            prepared.append((name, binding, value))
        # Validate all names/shapes before changing any input.
        for name, binding, value in prepared:
            object_id = binding["object_id"]
            if object_id is None:
                self.host[name].copy_(value)
            else:
                alias = self.storage.aliases[object_id]
                flat = value.reshape(-1)
                for logical, start, count in spans(binding["shape"], binding["stride"]):
                    for offset, destination in self.storage.spill.chunks(
                        alias,
                        dtype=value.dtype,
                        start=start,
                        length=count,
                        read=False,
                        write=True,
                    ):
                        at = logical + offset - start
                        destination.copy_(flat[at : at + destination.numel()])
        self.buffers.apply(buffers)
