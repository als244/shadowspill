"""Worker-owned groups and transfers reconstructed from saved Plan descriptions."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.plan import Plan
from shadowspill.communication.binding.backend import shared_backend_setup
from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.requests import CollectiveRequest, collective_requests
from shadowspill.parallelism.agreement import agreement_digest
from shadowspill.resources.description import resource_groups

from .storage import ShardedTransfers
from .warmup import warm_collectives


def backend_differs(backend: Mapping[str, Any], plan: Plan) -> bool:
    """Whether the workers' backend setup differs from the one ``plan``'s
    calibration recorded. A forward Plan may use a subset of a training Plan's
    groups, and a Plan that issues no collectives was calibrated without a
    backend, so no setup applies to it."""
    expected = plain(plan.communication_info.metadata.get("backend_setup", {}))
    if not expected and not any(
        len(request.ranks) > 1 for request in collective_requests(plan.program, None)
    ):
        return False
    observed = (
        {
            **backend,
            "groups": {
                name: backend.get("groups", {}).get(name)
                for name in expected.get("groups", {})
            },
        }
        if backend
        else {}
    )
    return bool(observed != expected)


class DistributedWorker:
    def __init__(self, plans: Mapping[str, Plan], rank_map: Mapping[int, int] | None):
        self.plans = dict(plans)
        self.transfers: dict[str, ShardedTransfers] = {}
        first = next(iter(plans.values()))
        self.requests: tuple[CollectiveRequest, ...] = ()
        failure = None
        try:
            self.requests = tuple(
                sorted(
                    {
                        request
                        for plan in plans.values()
                        for request in collective_requests(plan.program, None)
                    },
                    key=lambda request: (
                        request.ranks,
                        request.operation,
                        request.tensor_bytes,
                        request.dtype,
                        request.inplace,
                    ),
                )
            )
            for plan in plans.values():
                if plan.simulation_only_reasons:
                    raise ValueError(
                        "Plan is simulation-only: "
                        + "; ".join(plan.simulation_only_reasons)
                    )
                if (
                    plan.program.parallelism is None
                    or plan.program.parallelism.world_size != dist.get_world_size()
                ):
                    raise ValueError("rendezvous world size differs from the Plan")
                if (
                    plan.communication_info.topology
                    != first.communication_info.topology
                ):
                    raise ValueError("admitted Plans require the same logical topology")
        except Exception as error:
            failure = str(error)
        self.groups = GroupBindings(
            # Ranks agree on what their Plans exchange; what each built,
            # profiled or budgeted is its own.
            identity=digest(
                {
                    name: agreement_digest(plan.program)
                    for name, plan in sorted(plans.items())
                }
            ),
            groups=[request.ranks for request in self.requests]
            + [
                ranks
                for plan in plans.values()
                for ranks in resource_groups(plan.program.resources)
            ],
            rank_map=rank_map,
            topology=first.communication_info.topology,
            backend="nccl",
            startup_error=failure,
        )

    def prepare(self, device: int) -> None:
        """Initialize fixed backend resources before admitting the worker arena."""
        devices = self.groups.gather(str(torch.cuda.get_device_properties(device).uuid))
        if len(set(devices)) != len(devices):
            raise ValueError("distributed workers require distinct CUDA devices")
        backend = shared_backend_setup(
            self.groups, self.requests, device_ordinal=device
        )
        failure = None
        if any(backend_differs(backend, plan) for plan in self.plans.values()):
            failure = "communication backend differs from the saved calibration"
        self.groups.check(failure)
        warm_collectives(self.requests, self.groups, torch.device("cuda", device))
        failure = None
        try:
            for plan in self.plans.values():
                key = plan.program.digest
                if key not in self.transfers:
                    self.transfers[key] = ShardedTransfers(
                        plan.program, self.groups, device
                    )
        except Exception as error:
            failure = str(error)
        self.groups.check(failure)

    def descriptors(self, plan: Plan) -> Mapping[str, int]:
        return self.transfers[plan.program.digest].descriptors

    def close(self) -> None:
        for transfers in self.transfers.values():
            transfers.close()
        self.transfers.clear()
        self.groups.close()
