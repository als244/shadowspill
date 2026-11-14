"""Finite checks with bounded scratch, including noncontiguous gradients."""

import torch


@torch.no_grad()
def all_finite(value, *, chunk_elements=1 << 20):
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    if not value.is_floating_point() and not value.is_complex():
        return True
    valid = torch.ones((), device=value.device, dtype=torch.bool)
    pending = [value.detach()]
    while pending:
        part = pending.pop()
        if part.numel() <= chunk_elements:
            valid.logical_and_(torch.isfinite(part).all())
            continue
        # Narrow views instead of flattening: a transpose must not create a
        # full contiguous gradient copy just to validate its values.
        axis = max(range(part.ndim), key=lambda d: part.shape[d])
        length = part.shape[axis]
        midpoint = length // 2
        pending.append(part.narrow(axis, midpoint, length - midpoint))
        pending.append(part.narrow(axis, 0, midpoint))
    return bool(valid)


def validate_gradients(implementation, x, output, training):
    for name, value in [
        ("output", output),
        ("input gradient", x.grad),
        *implementation.named_gradients(),
    ]:
        if value is None or not all_finite(value):
            raise AssertionError(f"missing or nonfinite {name}")
        if name not in {"output", "input gradient"} and value.dtype != torch.float32:
            raise AssertionError(f"{name} gradient is {value.dtype}, expected FP32")
    if training == "lora" and any(
        p.grad is not None
        for _, p in implementation.named_parameters()
        if not p.requires_grad
    ):
        raise AssertionError("frozen base received parameter gradients")
