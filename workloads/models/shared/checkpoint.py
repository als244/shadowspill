"""Read a local Hugging Face checkpoint into a family's model, a parameter
at a time, without a converted copy on disk.

A family's ``checkpoint.py`` subclasses ``Checkpoint`` with the reader that
turns ``config.json`` into its ``Config`` and ``sources(name, config)``, the
checkpoint tensors one of the model's parameters is made of: one tensor as
it is, a permutation of one, a concatenation of several along their first
dimension, or a bank stacked from per-expert tensors. ``load_into`` reads
each source once from the memory-mapped safetensors shards and writes each
parameter once, a tied parameter once for both names, through ``copy``,
which by default handles a plain tensor and an FSDP2 shard. Keys the family
does not use, other than the prefixes it declares it ignores (a vision
tower, a multi-token-prediction head), are reported, so a layout the reader
does not know is noticed rather than silently left at its initialization.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mlops.sequential_moe.precision.safetensors import SafeTensors


@dataclass(frozen=True)
class Source:
    """The checkpoint tensors a parameter is made of, and how."""

    keys: tuple[str, ...]
    assemble: Callable[[list[torch.Tensor]], torch.Tensor] | None = None

    def read(self, reader: SafeTensors) -> torch.Tensor:
        values = [reader.tensor(key) for key in self.keys]
        if self.assemble is None:
            if len(values) != 1:
                raise ValueError("several tensors need an assembly")
            return values[0]
        return self.assemble(values)


def tensor(key: str) -> Source:
    return Source((key,))


def concatenated(*keys: str, dim: int = 0) -> Source:
    """Several tensors joined along ``dim``, in the order given."""
    return Source(tuple(keys), lambda values: torch.cat(values, dim=dim))


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

    return Source(keys, assemble)


def copy_into(parameter: torch.Tensor, value: torch.Tensor) -> None:
    """Write a full tensor into a parameter: as it is, or this rank's rows
    of an FSDP2 shard, which shards dimension zero in ``torch.chunk``'s
    pieces with the last ranks possibly empty."""

    if value.shape != tuple(parameter.shape):
        raise ValueError(
            f"shape {tuple(value.shape)} does not fit {tuple(parameter.shape)}"
        )
    with torch.no_grad():
        if hasattr(parameter, "to_local") and hasattr(parameter, "device_mesh"):
            placements = getattr(parameter, "placements", ())
            if (
                len(placements) != 1
                or not getattr(placements[0], "is_shard", lambda: False)()
                or placements[0].dim != 0
            ):
                raise ValueError("only dimension-zero sharded parameters are loaded")
            mesh = parameter.device_mesh
            rank, world = mesh.get_local_rank(), mesh.size()
            pieces = value.chunk(world, dim=0)
            local = parameter.to_local()
            piece = pieces[rank] if rank < len(pieces) else value[:0]
            if tuple(piece.shape) != tuple(local.shape):
                raise ValueError(
                    f"rank {rank}'s shard holds {tuple(local.shape)},"
                    f" not {tuple(piece.shape)}"
                )
            local.copy_(piece.to(dtype=local.dtype, device=local.device))
        else:
            parameter.copy_(value.to(dtype=parameter.dtype, device=parameter.device))


class Checkpoint:
    """A local Hugging Face checkpoint directory of one family."""

    #: The checkpoint's metadata (``config.json``) as the family's ``Config``.
    config_from_metadata: Callable[..., Any]
    #: Key prefixes the family's text decoder does not use.
    ignored_prefixes: tuple[str, ...] = ()

    def __init__(self, directory: str | Path, **overrides: Any) -> None:
        self.directory = Path(directory)
        metadata_path = self.directory / "config.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{self.directory} has no config.json")
        self.metadata = json.loads(metadata_path.read_text())
        self.config = type(self).config_from_metadata(self.metadata, **overrides)
        self.reader = SafeTensors(self.directory)

    def sources(self, name: str, config: Any) -> Source:
        """The checkpoint tensors parameter ``name`` is made of."""
        raise NotImplementedError

    def keys(self) -> list[str]:
        return sorted(self.reader.info)

    def load_into(
        self,
        model: torch.nn.Module,
        *,
        copy: Callable[[torch.Tensor, torch.Tensor], None] = copy_into,
        rename: Callable[[str], str] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Fill every parameter of ``model`` from the checkpoint; returns what
        was written and the checkpoint keys nothing used. ``rename`` turns a
        parameter's name in the model given into the model's own (a harness
        wraps blocks in modules of their own before it loads)."""

        used: set[str] = set()
        written: list[str] = []
        seen: dict[int, str] = {}
        parameters = list(model.named_parameters())
        for index, (name, parameter) in enumerate(parameters, start=1):
            if id(parameter) in seen:
                continue
            seen[id(parameter)] = name
            if rename is not None:
                name = rename(name)
            source = self.sources(name, self.config)
            value = source.read(self.reader)
            copy(parameter, value)
            used.update(source.keys)
            for key in source.keys:
                self.reader.release_pages(key)
            del value
            written.append(name)
            if progress is not None:
                progress(index, len(parameters), name)
        unused = [
            key
            for key in self.keys()
            if key not in used and not key.startswith(self.ignored_prefixes)
        ]
        return {"written": written, "unused": unused}

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()


def layer_name(name: str) -> tuple[int, str] | None:
    """``(layer, rest)`` for a block parameter ``blocks.N.rest``."""
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "blocks" and parts[1].isdigit():
        return int(parts[1]), parts[2]
    return None


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


__all__ = [
    "Checkpoint",
    "Source",
    "bank",
    "concatenated",
    "copy_into",
    "deinterleave_heads",
    "layer_name",
    "permuted",
    "tensor",
]
