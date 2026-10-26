"""Bounded reads of declared checkpoint copies into caller-provided storage."""

from bisect import bisect_right

import torch


def index_copies(copies, targets):
    """Resolve joined rows and expert banks once, using meta tensors only."""
    grouped = {}
    for copy in copies:
        view = copy.view(targets)
        if not view.is_contiguous():
            raise ValueError(f"checkpoint copy is not contiguous: {copy.destination}")
        start = view.storage_offset()
        grouped.setdefault(copy.destination, []).append(
            (start, start + view.numel(), copy)
        )
    result = {}
    for name, spans in grouped.items():
        spans.sort(key=lambda span: span[0])
        position = 0
        for start, stop, _ in spans:
            if start != position:
                raise ValueError(f"incomplete or overlapping checkpoint copies: {name}")
            position = stop
        target = targets[name]
        if position != target.numel():
            raise ValueError(f"incomplete checkpoint copies: {name}")
        result[name] = (
            tuple(target.shape),
            target.dtype,
            tuple(stop for _, stop, _ in spans),
            tuple(spans),
        )
    return result


def copy_chunk(reader, name, shape, offset, destination):
    """Copy only intersecting file-backed ranges; never assemble a whole bank."""
    if reader._chunk_copies is None:
        raise RuntimeError("Call build_model() before loading")
    if name not in reader._chunk_copies:
        if name in reader.lora_initializers:
            kind = "adapter initialization"
        elif any(name in item.destinations for item in reader.conversions):
            kind = "expert storage conversion"
        else:
            raise KeyError(f"no checkpoint tensor named {name}")
        raise ValueError(f"{name} needs {kind}, not a direct checkpoint copy")
    declared, dtype, ends, spans = reader._chunk_copies[name]
    if tuple(shape) != declared or destination.dtype != dtype:
        raise ValueError(f"checkpoint shape or dtype differs for {name}")
    if (
        destination.device.type != "cpu"
        or destination.ndim != 1
        or not destination.is_contiguous()
    ):
        raise ValueError("checkpoint destination must be a flat contiguous CPU buffer")
    end = offset + destination.numel()
    if offset < 0 or end > ends[-1]:
        raise ValueError("checkpoint slice leaves tensor bounds")
    for index in range(bisect_right(ends, offset), len(spans)):
        start, stop, copy = spans[index]
        if start >= end:
            break
        left, right = max(start, offset), min(stop, end)
        target = destination[left - offset : right - offset]
        source = reader.weights.source
        try:
            value = source.tensor(copy.source).view(-1)[left - start : right - start]
            if copy.reciprocal:
                if not bool(torch.isfinite(value).all() & (value > 0).all()):
                    raise ValueError(f"Invalid NVFP4 global scale: {copy.source}")
                target.copy_(value).reciprocal_()
            else:
                target.copy_(value)
        finally:
            source.release_pages(copy.source)
