"""Initialize saved collective forms; no calibration or implementation selection."""

import torch

from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.communication.task import PackedCollective


def warm_collectives(
    requests: tuple[CollectiveRequest, ...], groups: GroupBindings, device: torch.device
) -> None:
    for request in requests:
        arguments: tuple[torch.Tensor, ...] = ()
        failure = None
        active = groups.logical_rank in request.ranks
        try:
            if active:
                arguments = _arguments(request, groups.logical_rank, device)
        except Exception as error:
            failure = str(error)
        groups.check(failure)
        try:
            if active:
                function = PackedCollective(request, groups.for_members(request.ranks))
                for _ in range(2):
                    outputs = function(
                        arguments[0], arguments[1] if len(arguments) == 2 else None
                    )
                    torch.cuda.synchronize(device)
                    del outputs
                del arguments
        except Exception as error:
            failure = str(error)
        groups.check(failure)


def _arguments(
    request: CollectiveRequest, logical_rank: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    dtype = getattr(torch, request.dtype.removeprefix("torch."))
    gathered = request.operation == "all_gather"
    count = request.elements // len(request.ranks) if gathered else request.elements
    if gathered and request.inplace:
        target = torch.zeros(request.elements, dtype=dtype, device=device)
        offset = request.ranks.index(logical_rank) * count
        return target.narrow(0, offset, count), target
    return (torch.zeros(count, dtype=dtype, device=device),)
