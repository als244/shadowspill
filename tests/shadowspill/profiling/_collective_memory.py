"""One-GPU NCCL checks allocation observation, not multi-rank performance."""

import sys

import torch
import torch.distributed as dist

from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.info import CommunicationInfo
from shadowspill.communication.memory_measurement import measure_memory
from shadowspill.communication.memory_session import CommunicationMemory
from shadowspill.communication.probe import CollectiveProbe
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationOperation,
)

device = torch.device("cuda", 0)
torch.cuda.set_device(device)
dist.init_process_group("gloo", init_method=sys.argv[1], world_size=1, rank=0)
requests = tuple(
    CollectiveRequest(operation, (0,), 4096, "torch.float32", inplace=inplace)
    for operation, inplace in (
        ("all_gather", False),
        ("all_gather", True),
        ("reduce_scatter", False),
        ("all_reduce", True),
    )
)
memory = CommunicationMemory(device)
bindings = GroupBindings(
    identity="collective-allocation-observation",
    groups=[(0,)],
    rank_map=None,
    topology={},
    backend="nccl",
)
try:
    rows = []
    for request in requests:
        probe = CollectiveProbe(request, bindings.for_members((0,)), device)
        probe.sample()
        record = measure_memory(probe, device, bindings)
        assert record is not None and record["memory_kind"] == "measured", record
        assert record["workspace_bytes"] == 0, record
        contract = TaskAllocationContract.from_dict(record["allocation_contract"])
        allocations = sum(
            step.operation is TaskAllocationOperation.ALLOCATE
            for step in contract.steps
        )
        assert allocations == int(not request.inplace), record
        rows.append(
            {
                **request.to_dict(),
                "setup": build_setup(device),
                "topology": bindings.topology,
                "memory": {
                    "workspace_bytes": record["workspace_bytes"],
                    "memory_kind": "measured",
                    "per_rank": [{"logical_rank": 0, **record}],
                },
            }
        )
        del probe
    fixed, metadata = memory.finish(bindings, requests, build_setup(device))
    assert fixed is None or fixed >= 0
    assert metadata["memory_scope"]["groups"] == [(0,)]
    assert len(metadata["memory_per_rank"]) == 1
    info = CommunicationInfo(
        1e9,
        1e9,
        topology=bindings.topology,
        fixed_device_bytes=fixed,
        metadata=metadata,
        measurements=tuple(rows),
    )
    assert CommunicationInfo.from_dict(info.to_dict()) == info
    assert torch.cuda.memory.get_allocator_backend() == "native"
    print("collective memory observed without execution pools", flush=True)
finally:
    bindings.close()
    dist.destroy_process_group()
