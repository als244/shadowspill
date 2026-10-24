"""Fresh workers bind native descriptors and fill only their pinned host shards."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace


def program(shard_ranks=2, world_size=4):
    from shadowspill import Parallelism, Program
    from shadowspill.ir import AliasGroupSpec, DeviceSpec, ObjectSpec

    spec = Parallelism(
        ranks_per_replica=shard_ranks,
        num_replicas=world_size // shard_ranks,
        groups={"all_ranks": (tuple(reversed(range(shard_ranks))),)},
    )
    return Program(
        devices=(DeviceSpec("cuda_0", "worker", "cuda", 0),),
        alias_groups=(AliasGroupSpec("alias_000001", "cuda_0", 32),),
        objects=(ObjectSpec("object_000001", "alias_000001", 0, 30),),
        tasks=(),
        task_definitions=(),
        parallelism=spec,
        bindings={
            "kind": "forward",
            "entrypoints": [],
            "task_inputs": {},
            "root_inputs": [],
            "root_sources": [],
            "registrations": [{"name": "weight", "object_id": "object_000001"}],
            "model_tensor_geometry": {
                "object_000001": {
                    "dtype": "torch.bfloat16",
                    "shape": (15,),
                    "stride": (1,),
                }
            },
            "weight_shards": {
                "alias_000001": {
                    "groups": spec.group("all_ranks").groups,
                    "shard_bytes": 32 // shard_ranks,
                    "logical_bytes": 30,
                    "padded_bytes": 32,
                    "dtype": "torch.uint8",
                }
            },
        },
    )


def worker(index, rendezvous, archive, mode):
    import torch
    import torch.distributed as dist

    from shadowspill import Program
    from shadowspill.communication.binding.groups import GroupBindings
    from shadowspill.execution.distributed.storage import WeightTransfers
    from shadowspill.execution.storage import Storage
    from shadowspill.execution.tensors import visit_tensors
    from shadowspill.ir import (
        MemoryLocation,
        ResidencySpec,
        ShadowSpillProgram,
        TransferPhase,
        TransferSpec,
    )
    from shadowspill.memory import DevicePool, PinnedHostPool, TransferRoute
    from shadowspill.pytorch.frontend import PyTorchFrontend
    from shadowspill.runtime import Runtime
    from shadowspill.runtime.plan import RuntimeBridge
    from shadowspill.runtime.plan.lifecycle import abort_plan, begin_plan

    source = Program.load(archive)
    world_size = source.parallelism.world_size
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=index,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    device = index if mode == "nccl" else 0
    bound = None
    try:
        assert not torch.cuda.is_initialized()
        with Runtime(
            frontend=PyTorchFrontend(),
            calibrate=False,
            pools={
                "execution": DevicePool(
                    2 << 30, device=device, external_headroom=512 << 20
                ),
                "spill": PinnedHostPool(1 << 20),
            },
            routes={
                "fetch": TransferRoute("spill", "execution"),
                "evict": TransferRoute("execution", "spill"),
            },
        ) as runtime:
            groups = source.bindings["weight_shards"]["alias_000001"]["groups"]
            bound = GroupBindings(
                identity=source.digest,
                groups=groups,
                rank_map=(
                    {0: 1, 1: 0} if world_size == 2 else {0: 2, 1: 0, 2: 3, 3: 1}
                ),
                topology={},
                backend="nccl" if mode == "nccl" else "gloo",
            )
            transfers = WeightTransfers(source, bound, device)
            memory = begin_plan(
                runtime,
                execution="execution",
                spill="spill",
                execution_budget=None,
                spill_budget=None,
                dynamic_scratch_reserve_bytes=0,
                execution_device=device,
                require_calibration=False,
            )
            shard_bytes = source.bindings["weight_shards"]["alias_000001"][
                "shard_bytes"
            ]
            ir = ShadowSpillProgram(
                devices=source.devices,
                objects=source.objects,
                tasks=(),
                profiles=(),
                alias_groups=(
                    replace(
                        source.alias_groups[0],
                        transfer=TransferSpec(
                            shard_bytes,
                            (TransferPhase("copy", bytes=shard_bytes),),
                            (TransferPhase("copy", bytes=shard_bytes),),
                        ),
                    ),
                ),
            )
            bridge = RuntimeBridge(
                runtime,
                ir,
                memory.plan_handle,
                execution_pool_id=memory.execution.pool_id,
                spill_pool_id=memory.spill.pool_id,
            )
            try:
                plan = SimpleNamespace(
                    program=source,
                    schedule=SimpleNamespace(
                        initial_residency=(
                            ResidencySpec("alias_000001", MemoryLocation.SPILL),
                        )
                    ),
                )
                state = Storage(
                    plan,
                    bridge,
                    torch.device("cuda", device),
                    logical_rank=bound.logical_rank,
                    transfer_descriptors=transfers.descriptors,
                )
                # Registration retained the native descriptor independently.
                transfers.close()
                assert not transfers.descriptors
                host = state.host("alias_000001")
                assert host.numel() == shard_bytes and host.is_pinned()
                region = state.regions["object_000001"]
                calls = []

                def fill(request, destination):
                    calls.append(request.offset)
                    assert destination.is_pinned() and destination.numel() <= 3
                    destination.copy_(
                        torch.arange(
                            request.offset, request.offset + destination.numel()
                        )
                        + 1
                    )

                visit_tensors(state, fill, 6, initialize=True)
                expected = torch.zeros(region.capacity, dtype=torch.bfloat16)
                expected[: region.length] = (
                    torch.arange(region.offset, region.offset + region.length) + 1
                )
                torch.testing.assert_close(host.view(torch.bfloat16), expected)
                assert calls[0] == region.offset
                rows = bound.gather(
                    (bound.logical_rank, host.view(torch.bfloat16).tolist())
                )
                by_rank = dict(rows)
                width = source.parallelism.ranks_per_replica
                for rank in range(width, world_size):
                    assert by_rank[rank] == by_rank[rank % width]
                del host
                if mode != "host":
                    from tests.shadowspill.execution._shard_roundtrips import roundtrips

                    roundtrips(state, bound if mode == "nccl" else None)
                state.clear_views()
            finally:
                transfers.close()
                bridge.objects.unregister(bridge.objects.registered_aliases())
                abort_plan(runtime, memory.plan_handle)
            bound.close()
            bound = None
    finally:
        if bound is not None:
            bound.close()
        dist.destroy_process_group()
