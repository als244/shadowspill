"""Bounded initialization for expert parallelism: the values a seed gives the
default implementation, whatever the ranks.

Dense tensors come from the catalog initializer under their own names. An
expert-parallel layer's tensors hold what that initializer draws for the same
model without expert parallelism:

``router_weight``
    the router's weight.
``gate_up_weight``
    this rank's experts of the gate/up bank. The layer keeps each gate row
    beside its up row; the default bank holds an expert's gate rows and then
    its up rows, so row ``2j`` here is gate row ``j`` there and row ``2j + 1``
    is up row ``j``.
``down_weight``
    this rank's experts of the down bank.

A rank owns a contiguous run of experts, so its bank is a slice of the full
one. The seed, a tensor's name and an element's place in the full tensor decide
every value; the number of ranks and the choice of implementation do not.
"""

import math
from dataclasses import replace

import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.checkpoint.initialization import fill_generated, validate_chunk

#: The default implementation's names for the gate/up and down banks.
BANKS = {
    "qwen3moe": ("w13", "w2"),
    "qwen35moe": ("w13", "w2"),
    "olmoe": ("w13_experts", "w2_experts"),
}


class Initialize:
    def __init__(
        self, catalog_initializer, *, expert_rank, ranks, banks, std, router_dtype
    ):
        self.catalog = catalog_initializer
        self.expert_rank, self.ranks = expert_rank, ranks
        self.gate_up, self.down = banks
        self.std, self.router_dtype = std, router_dtype

    def __call__(self, request, destination):
        # Quack exposes physical BF16 payloads under logical FP32 parameters.
        # A packed compute weight and its master are requested separately; both
        # take the same values at payload precision, even when their chunks
        # differ, so the first update starts from the loaded weight.
        request = replace(
            request,
            name=request.name.removeprefix("master.").removesuffix("._data"),
        )
        if ".moe.experts." not in request.name:
            self.catalog(request, destination)
            return
        validate_chunk(request, destination)
        prefix, leaf = request.name.rsplit("experts.", 1)
        if leaf == "router_weight":
            self._draw(
                prefix + "router.weight",
                request.shape,
                request.offset,
                destination,
                self.router_dtype,
            )
        elif leaf == "down_weight":
            self._draw(
                prefix + self.down,
                self._full(request.shape),
                self.expert_rank * math.prod(request.shape) + request.offset,
                destination,
                torch.bfloat16,
            )
        elif leaf == "gate_up_weight":
            self._gate_up(prefix + self.gate_up, request, destination)
        else:
            # Tensors the default implementation lays out differently keep
            # their own names. They are the same on every rank.
            fill_generated(
                request, destination, sample=self._normal, seed=self.catalog.seed
            )

    def _full(self, shape):
        return (shape[0] * self.ranks, *shape[1:])

    def _draw(self, name, shape, offset, destination, payload):
        """Fill ``destination`` from the default implementation's ``name``,
        rounded to the layer's payload precision."""
        fill_generated(
            TensorChunk(
                name, tuple(shape), destination.dtype, offset, math.prod(shape)
            ),
            destination,
            sample=lambda *where: self.catalog.values(*where).to(payload),
            seed=self.catalog.seed,
            block_elements=self.catalog.block_elements,
        )

    def _gate_up(self, name, request, destination):
        _, rows, width = request.shape
        if rows % 2:
            raise ValueError("a gate/up bank needs an even number of rows")
        full, half = self._full(request.shape), rows // 2
        start, stop = request.offset, request.offset + destination.numel()
        # The chunk's whole rows are filled in place; a row it only partly
        # holds, at either end, is cut from a generated copy.
        body = slice(-(-start // width), stop // width)
        whole = destination[body.start * width - start : body.stop * width - start]
        whole = whole.view(-1, width) if body.stop > body.start else None
        first, last = start // width, -(-stop // width)
        row = first
        while row < last:
            expert, begin = divmod(row, rows)
            end = min(rows, begin + last - row)
            base = (self.expert_rank * request.shape[0] + expert) * rows
            for parity, origin in ((0, 0), (1, half)):
                local = range(begin + (parity - begin) % 2, end, 2)
                if not local:
                    continue
                # One expert's run of gate rows, or of up rows: at most half a
                # bank's rows, so the scratch stays far below a chunk.
                values = torch.empty(len(local) * width, dtype=torch.bfloat16)
                self._draw(
                    name,
                    full,
                    (base + origin + local[0] // 2) * width,
                    values,
                    torch.bfloat16,
                )
                values = values.view(len(local), width)
                held = row - begin + local[0]  # The chunk's first of these rows.
                for index, flat in (
                    (0, held),
                    (len(local) - 1, held + 2 * (len(local) - 1)),
                ):
                    if body.start <= flat < body.stop:
                        continue
                    low, high = max(start, flat * width), min(stop, (flat + 1) * width)
                    destination[low - start : high - start].copy_(
                        values[index, low - flat * width : high - flat * width]
                    )
                inside = range(
                    max(0, -(-(body.start - held) // 2)),
                    min(len(local), -(-(body.stop - held) // 2)),
                )
                if whole is not None and inside:
                    target = held + 2 * inside.start - body.start
                    whole[target : target + 2 * len(inside) : 2].copy_(
                        values[inside.start : inside.stop]
                    )
            row += end - begin

    def _normal(self, name, shape, start, stop, generator):
        values = torch.empty(stop - start, dtype=torch.float32).normal_(
            std=self.std, generator=generator
        )
        return values.to(torch.bfloat16)
