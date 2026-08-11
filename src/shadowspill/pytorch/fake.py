"""Storage-free CUDA replicas for capture before model materialization."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map

from .contracts import CaptureError, TensorSpec


def fake_cuda_model(
    model: nn.Module, mode: FakeTensorMode, *, device_index: int = 0
) -> nn.Module:
    """Copy module structure while preserving registered storage aliasing."""

    registrations = tuple(model.named_parameters(remove_duplicate=False)) + tuple(
        model.named_buffers(remove_duplicate=False)
    )
    groups: dict[tuple[int, int], list[torch.Tensor]] = {}
    for name, tensor in registrations:
        if tensor.device.type != "cpu" or tensor.layout is not torch.strided:
            raise CaptureError(
                f"registered tensor {name!r} must be a strided CPU tensor"
            )
        storage = tensor.untyped_storage()
        groups.setdefault((int(storage.data_ptr()), int(storage.nbytes())), []).append(
            tensor
        )

    memo: dict[int, Any] = {}
    device = torch.device("cuda", device_index)
    with mode:
        for (_address, storage_bytes), tensors in groups.items():
            storage_owner = torch.empty(storage_bytes, dtype=torch.uint8, device=device)
            for tensor in tensors:
                replica = torch.empty(0, dtype=tensor.dtype, device=device).set_(
                    storage_owner.untyped_storage(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                )
                replica.requires_grad_(bool(tensor.requires_grad))
                if isinstance(tensor, nn.Parameter):
                    replica = nn.Parameter(
                        replica, requires_grad=bool(tensor.requires_grad)
                    )
                memo[id(tensor)] = replica
        try:
            return copy.deepcopy(model, memo)
        except BaseException as exc:
            raise CaptureError("model structure cannot be copied for capture") from exc


def fake_cuda_inputs(
    values: Any, mode: FakeTensorMode, *, device_index: int = 0
) -> Any:
    """Replace only tensor leaves with storage-free fixed CUDA geometry."""

    device = torch.device("cuda", device_index)

    def convert(value: object) -> object:
        if isinstance(value, TensorSpec):
            shape = value.shape
            stride = value.resolved_stride
            dtype = value.dtype
            requires_grad = value.requires_grad
        elif isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            stride = tuple(value.stride())
            dtype = value.dtype
            requires_grad = bool(value.requires_grad)
        else:
            return value
        result = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        result.requires_grad_(requires_grad)
        return result

    with mode:
        return tree_map(convert, values)
