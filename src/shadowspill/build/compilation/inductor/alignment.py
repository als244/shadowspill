"""Give Inductor the alignment of rank-dependent views without changing layouts."""

from collections.abc import Sequence

import torch
from torch._guards import detect_fake_mode


def alignment_examples(
    inputs: Sequence[object], offsets: tuple[tuple[int, int], ...]
) -> Sequence[object]:
    """Only code generation sees these fake offsets; task contracts stay local.

    A rank's optimizer view can begin inside a gathered parameter allocation.
    A zero-offset example would promise alignment and make Inductor clone an
    unaligned runtime view. Compile from an actual shard stride instead. Apply
    this after output-alias normalization so it cannot rebase semantic views.
    """
    if not offsets:
        return inputs
    mode = detect_fake_mode(inputs)
    if mode is None:
        raise ValueError("alignment specialization requires fake compiler inputs")
    result = list(inputs)
    with mode:
        for index, offset in offsets:
            value = inputs[index]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cuda"
                or value.ndim != 1
                or value.stride() != (1,)
                or offset <= 0
            ):
                raise ValueError("owned optimizer inputs must be flat CUDA views")
            result[index] = torch.empty(
                value.numel() + offset, dtype=value.dtype, device=value.device
            ).narrow(0, offset, value.numel())
    return result
