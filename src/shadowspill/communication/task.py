"""The packed collective callable shared by calibration and execution."""

import torch
import torch.distributed as dist

from .operations import launch_collective
from .requests import CollectiveRequest


class PackedCollective:
    """One output allocation or an in-place update, joined to the caller stream.

    This is an internal task implementation, not an execution admission bypass.
    The Plan must separately account for communicator resources and allocation
    requirements. CUDA execution uses NCCL; CPU/Gloo supports functional tests.
    """

    def __init__(self, request: CollectiveRequest, group: dist.ProcessGroup):
        self.group, self.request = group, request
        self.dtype = getattr(torch, request.dtype.removeprefix("torch."))

    def __call__(
        self, source: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        request = self.request
        elements, count = request.elements, len(request.ranks)
        gathered = request.operation == "all_gather"
        returned: tuple[torch.Tensor, ...]
        self._validate(source, elements // count if gathered else elements)
        if source.is_cuda and dist.get_backend(self.group) != "nccl":
            raise ValueError("CUDA framework collective tasks require NCCL")
        if gathered and request.inplace:
            if target is None:
                raise ValueError("in-place gather requires its full output buffer")
            self._validate(target, elements)
            offset = dist.get_rank(self.group) * (elements // count)
            if (
                source.device != target.device
                or source.untyped_storage()._cdata != target.untyped_storage()._cdata
                or source.storage_offset() != target.storage_offset() + offset
            ):
                raise ValueError("gather input must be this rank's ordered output view")
            output, returned = target, ()
        else:
            if target is not None:
                raise ValueError("only an in-place gather accepts a second tensor")
            if request.operation == "all_reduce":
                output, returned = source, ()
            else:
                output = torch.empty(
                    elements if gathered else elements // count,
                    device=source.device,
                    dtype=self.dtype,
                )
                returned = (output,)
        work = launch_collective(request.operation, output, source, self.group)
        if source.is_cuda:
            # NCCL records a real stream dependency, including when users set
            # TORCH_NCCL_BLOCKING_WAIT. No CPU wait belongs in task dispatch.
            work.block_current_stream()
        else:
            work.wait()
        return returned

    def _validate(self, value: torch.Tensor, elements: int) -> None:
        if (
            value.dtype != self.dtype
            or tuple(value.shape) != (elements,)
            or value.stride() != (1,)
        ):
            raise ValueError("collective tensor differs from the saved packed layout")
