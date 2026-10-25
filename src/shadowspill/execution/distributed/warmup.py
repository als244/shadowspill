"""Initialize saved collective forms; no calibration or implementation selection."""

import torch

from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.communication.task import PackedCollective


def warm_collectives(
    requests: tuple[CollectiveRequest, ...], groups: GroupBindings, device: torch.device
) -> None:
    for request in requests:
        arguments, failure = (), None
        active = groups.logical_rank in request.ranks
        try:
            if active:
                dtype = getattr(torch, request.dtype.removeprefix("torch."))
                tensor = torch.zeros(request.elements, dtype=dtype, device=device)
                if request.operation == "all_gather":
                    count = request.elements // len(request.ranks)
                    source = tensor.narrow(
                        0, request.ranks.index(groups.logical_rank) * count, count
                    )
                    arguments = (source, tensor) if request.inplace else (source,)
                else:
                    arguments = (tensor,)
        except Exception as error:
            failure = str(error)
        groups.check(failure)
        try:
            if active:
                function = PackedCollective(request, groups.for_members(request.ranks))
                for _ in range(2):
                    outputs = function(*arguments)
                    torch.cuda.synchronize(device)
                    del outputs
                del arguments, tensor
        except Exception as error:
            failure = str(error)
        groups.check(failure)
