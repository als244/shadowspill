"""Allocate independent stage inputs while preserving shared storage views."""

import copy
from collections.abc import Sequence

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.accelerator import accelerator_device, is_accelerator


def materialize_example_arguments(
    arguments: Sequence[object], *, device_ordinal: int
) -> tuple[object, ...]:
    """Create concrete values while preserving every storage alias and view."""

    device = accelerator_device(device_ordinal)
    storages: dict[tuple[int, str], torch.Tensor] = {}
    results: list[object] = []
    with torch.no_grad():
        for argument in arguments:
            if not isinstance(argument, torch.Tensor):
                results.append(copy.deepcopy(argument))
                continue
            if argument.layout is not torch.strided:
                raise CaptureError(
                    "compiled task examples currently require strided tensors"
                )
            source_storage = argument.untyped_storage()
            target_device = (
                device if is_accelerator(argument.device) else argument.device
            )
            key = (source_storage._cdata, str(target_device))
            storage = storages.get(key)
            if storage is None:
                storage = torch.empty(
                    source_storage.nbytes(), dtype=torch.uint8, device=target_device
                )
                storage.zero_()
                storages[key] = storage
            tensor = torch.empty(0, dtype=argument.dtype, device=target_device)
            tensor.set_(
                storage.untyped_storage(),
                argument.storage_offset(),
                tuple(argument.shape),
                tuple(argument.stride()),
            )
            tensor.requires_grad_(argument.requires_grad)
            results.append(tensor)
    return tuple(results)
