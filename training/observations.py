"""Small training results collected at the existing step synchronization boundary."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def parameter_norms(
    parameter: torch.Tensor, gradient: torch.Tensor
) -> dict[str, torch.Tensor]:
    """FP32 L2 norms of compute weights and final gradients before the update."""

    return {
        "param_norm": torch.linalg.vector_norm(parameter, dtype=torch.float32),
        "grad_norm": torch.linalg.vector_norm(gradient, dtype=torch.float32),
    }


def to_host(value: Any) -> Any:
    """Copy only returned summaries; synchronize once per contributing GPU.

    No GPU packing allocation is made outside an admitted task. Copies share
    pinned host storage by dtype, and all are submitted before waiting. Returned
    CPU views own that storage; no device tensor is retained by the result.
    """

    leaves, spec = tree_flatten(value)
    groups = defaultdict(list)
    result = list(leaves)
    for index, leaf in enumerate(leaves):
        if isinstance(leaf, torch.Tensor):
            if leaf.device.type == "cpu":
                result[index] = leaf.detach().clone()
            else:
                if leaf.device.type != "cuda":
                    raise ValueError(f"metric copies do not support {leaf.device.type}")
                groups[(leaf.device, leaf.dtype)].append((index, leaf))
    devices = set()
    for (device, dtype), tensors in groups.items():
        host = torch.empty(
            sum(t.numel() for _, t in tensors), dtype=dtype, pin_memory=True
        )
        offset = 0
        for index, tensor in tensors:
            view = host[offset : offset + tensor.numel()].view(tensor.shape)
            view.copy_(tensor.detach(), non_blocking=True)
            result[index] = view
            offset += tensor.numel()
        devices.add(device)
    for device in devices:
        torch.cuda.current_stream(device).synchronize()
    return tree_unflatten(result, spec)


@dataclass(frozen=True)
class StepObservations:
    losses: tuple[float, ...]
    metrics: tuple[Any, ...] = ()
    parameter_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[float]:
        return iter(self.losses)

    @classmethod
    def collect(cls, losses, metrics=(), parameter_metrics=None):
        host_losses, host_metrics, host_parameters = to_host(
            (losses, metrics, parameter_metrics or {})
        )
        return cls(
            tuple(value.item() for value in host_losses),
            tuple(host_metrics),
            host_parameters,
        )


@dataclass(frozen=True)
class MetricTable:
    """CPU-only, indexed observations for JSON and W&B tables."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class MetricSummary:
    """A reducer's step-level scalars and optional detail tables."""

    scalars: Mapping[str, float] = field(default_factory=dict)
    tables: Mapping[str, MetricTable] = field(default_factory=dict)


def parameter_scalars(
    observations: Mapping[str, Any], parameter_sizes: Mapping[str, int]
) -> dict[str, float]:
    """CPU norm summaries, derived scales, and module shares of squared gradients.

    Sizes are metadata only. Module shares include descendants; nonoverlapping
    modules covering the observed parameters sum to one when gradients are
    nonzero. Empty tensors omit RMS, and all-zero gradients have zero shares.
    """

    result = {}
    squared = defaultdict(float)
    module_squared = defaultdict(float)
    for name, value in observations.items():
        name = name.removeprefix("model.")
        path = name.split(".")
        path = [part.zfill(2) if part.isdecimal() else part for part in path]
        module = ".".join(path[:-1]) or "parameters"
        stats = value if isinstance(value, Mapping) else {"parameter_metrics": value}
        scalars = {}
        for metric, scalar_value in stats.items():
            if not isinstance(scalar_value, torch.Tensor) or scalar_value.numel() != 1:
                raise ValueError(
                    "parameter metric logging requires named scalar tensors"
                )
            if scalar_value.device.type != "cpu":
                raise ValueError(
                    "parameter logging expects CPU summaries after the step"
                )
            scalar = scalar_value.item()
            scalars[metric] = scalar
            if metric in {"grad_norm", "param_norm"}:
                squared[metric] += scalar * scalar
                size = parameter_sizes[name]
                if size:
                    scalars[metric.replace("_norm", "_rms")] = scalar / size**0.5
            if metric == "grad_norm":
                modules = [".".join(path[:i]) for i in range(1, len(path))]
                for ancestor in modules or ["parameters"]:
                    module_squared[ancestor] += scalar * scalar
        if "grad_norm" in scalars and "param_norm" in scalars:
            scalars["grad_weight_ratio"] = scalars["grad_norm"] / (
                scalars["param_norm"] + 1e-12
            )
        result.update(
            (f"{metric}/{module}/{path[-1]}", scalar)
            for metric, scalar in scalars.items()
        )
    for metric, total in squared.items():
        result[f"{metric}/global/l2"] = total**0.5
    total = squared["grad_norm"]
    for module, value in module_squared.items():
        result[f"grad_squared_share/{module}"] = value / total if total != 0 else 0.0
    return result
