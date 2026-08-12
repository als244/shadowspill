"""Per-tensor BF16 qualification metrics and deterministic state hashing."""

from __future__ import annotations

import hashlib
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
) -> tuple[dict[str, TensorMetrics], tuple[str, ...]]:
    """Compare matching nested states and return tensor and scalar evidence."""

    tensors: dict[str, TensorMetrics] = {}
    exact_failures: list[str] = []

    def compare(left: object, right: object, path: str) -> None:
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor):
                exact_failures.append(f"{path}: expected tensor")
                return
            if left.shape != right.shape or left.dtype != right.dtype:
                exact_failures.append(
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
                exact_failures.append(f"{path}: mapping keys differ")
                return
            for key in sorted(left, key=str):
                compare(left[key], right[key], f"{path}/{key}")
            return
        if isinstance(left, (list, tuple)):
            if not isinstance(right, type(left)) or len(left) != len(right):
                exact_failures.append(f"{path}: sequence differs")
                return
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=True)
            ):
                compare(left_item, right_item, f"{path}/{index}")
            return
        if left != right:
            exact_failures.append(f"{path}: {left!r} != {right!r}")

    compare(reference, actual, "state")
    return tensors, tuple(exact_failures)


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> TensorMetrics:
    left = reference.detach().float().reshape(-1)
    right = actual.detach().float().reshape(-1)
    if left.numel() == 0:
        return TensorMetrics(1.0, 0.0, 1.0, 0.0)
    difference = right - left
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = left_norm * right_norm
    cosine = (
        1.0
        if float(denominator) == 0.0 and torch.equal(left, right)
        else float(torch.dot(left, right) / denominator.clamp_min(1e-30))
    )
    relative_l2 = float(
        torch.linalg.vector_norm(difference) / left_norm.clamp_min(1e-30)
    )
    sign_agreement = float((torch.sign(left) == torch.sign(right)).float().mean())
    return TensorMetrics(
        cosine=cosine,
        relative_l2=relative_l2,
        sign_agreement=sign_agreement,
        maximum_absolute_error=float(difference.abs().max()),
        reference_norm=float(left_norm),
        actual_norm=float(right_norm),
        difference_norm=float(torch.linalg.vector_norm(difference)),
        reference_maximum_absolute=float(left.abs().max()),
        actual_maximum_absolute=float(right.abs().max()),
        numel=left.numel(),
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
