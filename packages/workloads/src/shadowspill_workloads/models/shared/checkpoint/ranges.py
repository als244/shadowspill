"""Copy contiguous source ranges for concatenation, expert banks and gated Q."""


def copy_parts(parts, reader, offset, destination):
    written = 0
    for part in parts:
        elements = part.read(reader, meta=True).numel()
        if offset >= elements:
            offset -= elements
            continue
        count = min(elements - offset, destination.numel() - written)
        part.copy_into(reader, offset, destination[written : written + count])
        written += count
        if written == destination.numel():
            return
        offset = 0
    if written != destination.numel():
        raise ValueError("checkpoint parts do not cover the requested range")


def copy_deinterleaved(value, heads, offset, destination):
    rows, width = value.shape
    if heads < 1 or rows % (2 * heads):
        raise ValueError("gated query rows must divide into two halves per head")
    block = rows // (2 * heads) * width
    flat = value.view(-1)
    written = 0
    while written < destination.numel():
        half, rest = divmod(offset + written, heads * block)
        head, inner = divmod(rest, block)
        start = (2 * head + half) * block + inner
        count = min(block - inner, destination.numel() - written)
        destination[written : written + count].copy_(flat[start : start + count])
        written += count
