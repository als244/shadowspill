"""Write one destination at a time, preserving LoRA initialization order."""

import math

import torch
from torch import nn

from .experts import convert_expert
from .tensors import check_rows


def initialize_lora(reader, targets, adopted, on_tensor):
    # Construct only one factor at a time; imported meta parameters have no
    # initialized bytes. copy_ also supports non-addressable spill pools.
    for name, is_a in reader.lora_initializers.items():
        if name in adopted:
            continue
        target = targets[name]
        value = torch.empty(tuple(target.shape), dtype=target.dtype, device="cpu")
        if is_a:
            # Per expert for a stacked [E, rank, in] factor: fan-in ``in``.
            nn.init.kaiming_uniform_(value.view(-1, value.shape[-1]), a=math.sqrt(5))
        else:
            nn.init.zeros_(value)
        target.copy_(value)
        if on_tensor is not None:
            on_tensor(name)


def load_copies(reader, targets, adopted, progress, write_tensor, on_tensor):
    grouped = {}
    for copy in reader.copies:
        if copy.destination in adopted:
            continue
        grouped.setdefault(copy.destination, []).append(copy)
    total = len(reader.copies) + sum(len(c.experts) for c in reader.conversions)
    done = 0
    for destination, copies in grouped.items():
        target = targets[destination]
        joined = not (
            len(copies) == 1 and copies[0].start is None and copies[0].index is None
        )
        if joined:
            # Assemble joined gate/up and Q/K/V rows, and a layer's expert
            # bank, before publishing: each destination is written once,
            # even to an SSD pool. Scratch is one joined projection or one
            # component of one layer's bank.
            check_rows(destination, copies, tuple(target.shape))

        filled = False

        def fill(staging, selected=tuple(copies), name=destination, before=done):
            nonlocal filled
            filled = True
            for index, copy in enumerate(selected, start=1):
                source = reader.weights.source.tensor(copy.source)
                if copy.reciprocal:
                    if not bool(torch.isfinite(source).all() & (source > 0).all()):
                        raise ValueError(f"Invalid NVFP4 global scale: {copy.source}")
                    source = source.reciprocal()
                view = copy.view({name: staging})
                view.copy_(source)
                del source, view
                reader.weights.source.release_pages(copy.source)
                if progress is not None:
                    progress(before + index, total, copy.source)

        if write_tensor is None:
            staging = (
                torch.empty(tuple(target.shape), dtype=target.dtype, device="cpu")
                if joined
                else target
            )
            fill(staging)
            if staging is not target:
                target.copy_(staging)
            del staging
        else:
            write_tensor(target, fill)
        done += len(copies)
        if on_tensor is not None:
            on_tensor(destination)
        if progress is not None and not filled:
            progress(done, total, copies[-1].source)
    return done, total


def load_conversions(reader, targets, adopted, progress, on_tensor, done, total):
    # Banks that change form: decoded and quantized an expert at a time,
    # on the GPU when there is one, into host staging written once.
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    for conversion in reader.conversions:
        stagings = {
            destination: torch.empty(
                tuple(targets[destination].shape),
                dtype=targets[destination].dtype,
                device="cpu",
            )
            for destination in conversion.destinations
            if destination not in adopted
        }
        for index, prefix in conversion.experts:
            if stagings:
                values = convert_expert(reader, prefix, conversion, device)
                for destination, value in zip(
                    conversion.destinations, values, strict=True
                ):
                    if destination in stagings:
                        stagings[destination][index].copy_(value)
                del values
            done += 1
            if progress is not None:
                progress(done, total, prefix)
        for destination, staging in stagings.items():
            targets[destination].copy_(staging)
            if on_tensor is not None:
                on_tensor(destination)
        del stagings
