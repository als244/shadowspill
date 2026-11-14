"""Copy checkpoint tensors into an ordinary model, one source at a time."""

import math

import torch
from torch import nn

from .experts import convert_expert
from .tensors import check_rows


def initialize_lora(reader, targets, on_tensor):
    for name, is_a in reader.lora_initializers.items():
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


def load_copies(reader, targets, progress, on_tensor):
    grouped = {}
    for copy in reader.copies:
        grouped.setdefault(copy.destination, []).append(copy)
    total = len(reader.copies) + sum(len(c.experts) for c in reader.conversions)
    done = 0
    for destination, copies in grouped.items():
        if (
            len(copies) > 1
            or copies[0].index is not None
            or copies[0].start is not None
        ):
            check_rows(destination, copies, tuple(targets[destination].shape))
        for copy in copies:
            source = reader.weights.source.tensor(copy.source)
            if copy.reciprocal:
                if not bool(torch.isfinite(source).all() & (source > 0).all()):
                    raise ValueError(f"Invalid NVFP4 global scale: {copy.source}")
                source = source.reciprocal()
            copy.view(targets).copy_(source)
            del source
            reader.weights.source.release_pages(copy.source)
            done += 1
            if progress is not None:
                progress(done, total, copy.source)
        if on_tensor is not None:
            on_tensor(destination)
    return done, total


def load_conversions(reader, targets, progress, on_tensor, done, total):
    # Convert one expert at a time; no second full expert bank is staged.
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    for conversion in reader.conversions:
        for index, prefix in conversion.experts:
            values = convert_expert(reader, prefix, conversion, device)
            for destination, value in zip(conversion.destinations, values, strict=True):
                targets[destination][index].copy_(value)
            del values
            done += 1
            if progress is not None:
                progress(done, total, prefix)
        if on_tensor is not None:
            for destination in conversion.destinations:
                on_tensor(destination)
