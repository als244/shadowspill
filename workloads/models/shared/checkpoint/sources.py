"""Hugging Face tensor layouts: direct, concatenated, permuted and expert banks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from mlops.sequential_moe.precision.safetensors import SafeTensors


@dataclass(frozen=True)
class Source:
    """The checkpoint tensors a parameter is made of, and how."""

    keys: tuple[str, ...]
    assemble: Callable[[list[torch.Tensor]], torch.Tensor] | None = None
    copy_range: Callable[[SafeTensors, int, torch.Tensor], None] | None = None

    def read(self, reader: SafeTensors, *, meta: bool = False) -> torch.Tensor:
        values = [
            reader.tensor(key, meta=True) if meta else reader.tensor(key)
            for key in self.keys
        ]
        if self.assemble is None:
            if len(values) != 1:
                raise ValueError("several tensors need an assembly")
            return values[0]
        return self.assemble(values)

    def copy_into(
        self, reader: SafeTensors, offset: int, destination: torch.Tensor
    ) -> None:
        """Copy a flat output range without assembling the complete parameter."""
        if (
            destination.device.type != "cpu"
            or destination.ndim != 1
            or not destination.is_contiguous()
        ):
            raise ValueError(
                "checkpoint destination must be a flat contiguous CPU buffer"
            )
        if (
            offset < 0
            or offset + destination.numel() > self.read(reader, meta=True).numel()
        ):
            raise ValueError("checkpoint slice leaves tensor bounds")
        try:
            if self.copy_range is not None:
                self.copy_range(reader, offset, destination)
            elif self.assemble is None and len(self.keys) == 1:
                value = reader.tensor(self.keys[0]).view(-1)
                destination.copy_(value[offset : offset + destination.numel()])
            else:
                raise ValueError(
                    "checkpoint transform has no bounded copy implementation"
                )
        finally:
            for key in self.keys:
                reader.release_pages(key)


def tensor(key: str) -> Source:
    return Source((key,))


def concatenated(*keys: str, dim: int = 0) -> Source:
    """Several tensors joined along ``dim``, in the order given."""
    from .ranges import copy_parts

    parts = tuple(tensor(key) for key in keys)
    return Source(
        tuple(keys),
        lambda values: torch.cat(values, dim=dim),
        (lambda reader, offset, target: copy_parts(parts, reader, offset, target))
        if dim == 0
        else None,
    )


def permuted(key: str, permute: Callable[[torch.Tensor], torch.Tensor]) -> Source:
    return Source((key,), lambda values: permute(values[0]))


def bank(experts: Sequence[Source]) -> Source:
    """A ``[E, ...]`` bank stacked from one source per expert."""
    keys = tuple(key for expert in experts for key in expert.keys)
    counts = [len(expert.keys) for expert in experts]

    def assemble(values: list[torch.Tensor]) -> torch.Tensor:
        parts, start = [], 0
        for expert, count in zip(experts, counts, strict=True):
            chunk = values[start : start + count]
            start += count
            parts.append(expert.assemble(chunk) if expert.assemble else chunk[0])
        return torch.stack(parts, dim=0)

    from .ranges import copy_parts

    return Source(
        keys,
        assemble,
        lambda reader, offset, target: copy_parts(experts, reader, offset, target),
    )


def deinterleaved(key: str, heads: int) -> Source:
    from .ranges import copy_deinterleaved

    return Source(
        (key,),
        lambda values: deinterleave_heads(values[0], heads),
        lambda reader, offset, target: copy_deinterleaved(
            reader.tensor(key), heads, offset, target
        ),
    )


def deinterleave_heads(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """Rows ordered ``(head, half)`` as Hugging Face's gated attention keeps
    query and gate per head, to ``(half, head)``: all queries, then all
    gates."""

    rows, width = weight.shape
    return (
        weight.view(heads, 2, rows // (2 * heads), width)
        .transpose(0, 1)
        .reshape(rows, width)
    )
