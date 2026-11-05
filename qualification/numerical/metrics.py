"""Per-tensor BF16 qualification metrics and deterministic state hashing."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TensorMetrics:
    cosine: float
    relative_l2: float
    sign_agreement: float
    maximum_absolute_error: float
    reference_norm: float = 0.0
    actual_norm: float = 0.0
    difference_norm: float = 0.0
    reference_maximum_absolute: float = 0.0
    actual_maximum_absolute: float = 0.0
    numel: int = 0
    #: Precision-scaled reference magnitude, retained for the absolute-error
    #: bound. This is eps * magnitude, not the exact binary storage spacing.
    resolution: float = 0.0
    #: Exact binary spacing at the reference maximum (minimum subnormal at
    #: zero). Older records without this metadata get no eligibility slack.
    reference_maximum_ulp: float = 0.0


def state_digest(value: object) -> str:
    """Hash one nested CPU/device state without depending on pickle layout."""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def compare_states(
    reference: object,
    actual: object,
) -> tuple[dict[str, TensorMetrics], tuple[str, ...], tuple[str, ...]]:
    """Compare matching nested states and return tensor and scalar evidence.

    Structural disagreements are returned apart from value disagreements, and
    the separation is the point rather than tidiness. Two states that do not
    have the same shape are not two answers to the same question: whatever
    values happen to line up are being compared across tensors that do not
    correspond, so every metric taken over them describes nothing. Reporting
    such a run as "outside tolerance" names the symptom of a comparison that
    was never valid, and sends the reader looking for a numerical fault that
    is not there.
    """

    tensors: dict[str, TensorMetrics] = {}
    exact_failures: list[str] = []
    structure_failures: list[str] = []

    def compare(left: object, right: object, path: str) -> None:
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor):
                structure_failures.append(f"{path}: expected tensor")
                return
            if left.shape != right.shape or left.dtype != right.dtype:
                structure_failures.append(
                    f"{path}: geometry {left.shape}/{left.dtype} != "
                    f"{right.shape}/{right.dtype}"
                )
                return
            if not (left.is_floating_point() or left.is_complex()):
                if not torch.equal(left, right):
                    left_values = left.reshape(-1)[:8].tolist()
                    right_values = right.reshape(-1)[:8].tolist()
                    exact_failures.append(
                        f"{path}: integral tensor differs "
                        f"{left_values} != {right_values}"
                    )
                return
            tensors[path] = tensor_metrics(left, right)
            return
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                structure_failures.append(f"{path}: mapping keys differ")
                return
            for key in sorted(left, key=str):
                compare(left[key], right[key], f"{path}/{key}")
            return
        if isinstance(left, (list, tuple)):
            if not isinstance(right, type(left)) or len(left) != len(right):
                structure_failures.append(f"{path}: sequence differs")
                return
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=True)
            ):
                compare(left_item, right_item, f"{path}/{index}")
            return
        if left != right:
            exact_failures.append(f"{path}: {left!r} != {right!r}")

    compare(reference, actual, "state")
    return tensors, tuple(exact_failures), tuple(structure_failures)


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> TensorMetrics:
    # Widen before products and subtraction, not just their reductions. A
    # small nonzero FP32 product can underflow before an FP64 sum sees it.
    left = reference.detach().double().reshape(-1)
    right = actual.detach().double().reshape(-1)
    if left.numel() == 0:
        return TensorMetrics(1.0, 0.0, 1.0, 0.0)
    reference_maximum = float(left.abs().max())
    resolution = reference_maximum_ulp = 0.0
    if reference.is_floating_point():
        precision = torch.finfo(reference.dtype)
        resolution = precision.eps * reference_maximum
        if math.isfinite(reference_maximum):
            minimum_spacing = precision.eps * precision.tiny
            reference_maximum_ulp = (
                max(
                    math.ldexp(precision.eps, math.frexp(reference_maximum)[1] - 1),
                    minimum_spacing,
                )
                if reference_maximum
                else minimum_spacing
            )
    difference = right - left
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    difference_norm = torch.linalg.vector_norm(difference)
    # An absolute denominator floor changes both metrics when a tensor is
    # rescaled. Handle actual zeros explicitly and keep every nonzero norm.
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        cosine = 1.0 if torch.equal(left, right) else 0.0
    else:
        cosine = float((torch.dot(left, right) / (left_norm * right_norm)).clamp(-1, 1))
    relative_l2 = (
        (0.0 if float(difference_norm) == 0.0 else float("inf"))
        if float(left_norm) == 0.0
        else float(difference_norm / left_norm)
    )
    sign_agreement = float((torch.sign(left) == torch.sign(right)).float().mean())
    return TensorMetrics(
        cosine=cosine,
        relative_l2=relative_l2,
        sign_agreement=sign_agreement,
        maximum_absolute_error=float(difference.abs().max()),
        reference_norm=float(left_norm),
        actual_norm=float(right_norm),
        difference_norm=float(difference_norm),
        reference_maximum_absolute=reference_maximum,
        actual_maximum_absolute=float(right.abs().max()),
        numel=left.numel(),
        resolution=resolution,
        reference_maximum_ulp=reference_maximum_ulp,
    )


def cpu_state(value: Any) -> Any:
    """Clone a nested checkpoint into ordinary CPU-owned values."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


__all__ = [
    "TensorMetrics",
    "compare_states",
    "cpu_state",
    "state_digest",
    "tensor_metrics",
]
