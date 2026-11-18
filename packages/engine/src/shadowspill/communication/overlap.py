"""CPU-only identities and representative routes for paired timing measurements."""

import math
from collections import defaultdict
from dataclasses import dataclass

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program

from .info import CommunicationInfo
from .requests import CollectiveRequest
from .task_costs import task_collective_cost
from .transfers import transfer_collectives


def timing_class(
    requests: tuple[CollectiveRequest, ...], info: CommunicationInfo
) -> str:
    """Message size is the interpolation coordinate, never the route identity.

    Different dtypes, group membership/order, buffer aliasing or backend settings
    must not accidentally reuse another collective's interference measurement.
    """
    return "collective_" + digest(
        {
            "schema": 1,
            "route": [
                {
                    key: value
                    for key, value in r.to_dict().items()
                    if key != "tensor_bytes"
                }
                for r in requests
            ],
            "topology": plain(info.topology),
            "backend": plain(info.metadata.get("backend_setup", {})),
        }
    )


@dataclass(frozen=True)
class OverlapRoute:
    timing_class: str
    bytes: int
    duration_seconds: float
    requests: tuple[CollectiveRequest, ...]

    @property
    def buffer_bytes(self) -> int:
        return max(r.tensor_bytes for r in self.requests)


#: The most sizes one route is probed at.
MAX_ROUTE_SIZES = 10


def spread_sizes(sizes: list[int], limit: int = MAX_ROUTE_SIZES) -> list[int]:
    """Up to ``limit`` of the sorted ``sizes``, spread evenly in log size.

    The extremes are always kept. Each point between them is the actual size
    nearest an even step from smallest to largest in ``log1p`` bytes, the
    coordinate the simulator interpolates overlap rates in. Sizes that cluster
    therefore share a probe, and a wide range gets one in every part of it.
    """
    if len(sizes) <= limit:
        return list(sizes)
    scale = [math.log1p(size) for size in sizes]
    chosen = {0, len(sizes) - 1}
    for step in range(1, limit - 1):
        target = scale[0] + (scale[-1] - scale[0]) * step / (limit - 1)
        chosen.add(min(range(len(sizes)), key=lambda i: abs(scale[i] - target)))
    return [sizes[index] for index in sorted(chosen)]


def overlap_routes(
    program: Program, info: CommunicationInfo
) -> tuple[OverlapRoute, ...]:
    """Actual objects per route, spread over its sizes; no synthetic traffic sizes.

    An overlap rate changes steeply with message size, and the simulator
    interpolates between the sizes probed, so a route is probed across its
    whole range (see ``spread_sizes``) rather than at its ends and one middle.
    A reduction and its replica sum form one ordered transfer, as at execution.
    All members of a group family participate concurrently in paired probes.
    """
    by_class: dict[str, dict[int, OverlapRoute]] = defaultdict(dict)
    for requests in transfer_collectives(program.bindings).values():
        costs = [
            task_collective_cost(
                tuple(r for r in requests if r.operation == operation),
                info,
                program.setup,
            )
            for operation in dict.fromkeys(r.operation for r in requests)
        ]
        if not any(cost.transfer_bytes for cost in costs):
            continue
        if any(cost.source not in {"measurement", "identity"} for cost in costs):
            raise ValueError(
                "overlap calibration requires measured collective durations"
            )
        name = timing_class(requests, info)
        size = sum(cost.transfer_bytes for cost in costs)
        by_class[name][size] = OverlapRoute(
            name, size, sum(cost.duration_seconds for cost in costs), requests
        )
    result: list[OverlapRoute] = []
    for _, routes in sorted(by_class.items()):
        result.extend(routes[size] for size in spread_sizes(sorted(routes)))
    return tuple(result)
