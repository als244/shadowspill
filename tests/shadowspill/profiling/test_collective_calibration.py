"""Real CPU backend checks orchestration without pretending to measure GPUs."""

from datetime import timedelta


def _worker(worker, rendezvous):
    import torch
    import torch.distributed as dist

    from shadowspill import CommunicationInfo
    from shadowspill.communication.aggregation import combine_host
    from shadowspill.communication.binding.backend import shared_backend_setup
    from shadowspill.communication.binding.groups import GroupBindings
    from shadowspill.communication.collectives import collective_cost
    from shadowspill.communication.measure_collectives import measure_collectives
    from shadowspill.communication.probe import CollectiveProbe
    from shadowspill.communication.requests import CollectiveRequest
    from shadowspill.task.profiling import ProfilingOptions

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        world_size=4,
        rank=worker,
        timeout=timedelta(seconds=30),
    )
    bound = None
    try:
        requests = tuple(
            CollectiveRequest(op, group, 64, "torch.float32", inplace=inplace)
            for group in ((3, 0), (2, 1), (3, 1, 0, 2))
            for op in ("all_gather", "reduce_scatter", "all_reduce")
            for inplace in ((False, True) if op == "all_gather" else (None,))
        )
        bound = GroupBindings(
            identity="calibration",
            groups=[row.ranks for row in requests],
            rank_map={0: 2, 1: 0, 2: 3, 3: 1},
            topology={},
            backend="gloo",
            timeout=timedelta(seconds=30),
        )
        assert bound.topology == {"node0": (0, 1, 2, 3)}
        backend = shared_backend_setup(bound, requests, device_ordinal=None)
        assert set(backend["groups"]) == {"3,0", "2,1", "3,1,0,2"}
        assert backend["nccl_version"] is None
        combined = combine_host(
            CommunicationInfo(
                (worker + 1) * 1e9,
                (worker + 2) * 1e9,
                fetch_concurrent_bytes_per_second=(worker + 1) * 1e8,
                evict_concurrent_bytes_per_second=(worker + 2) * 1e8,
                fetch_latency_seconds=worker * 1e-6,
                evict_latency_seconds=worker * 2e-6,
            ),
            bound,
        )
        assert combined.fetch_bytes_per_second == 1e9
        assert combined.evict_concurrent_bytes_per_second == 2e8
        assert combined.evict_latency_seconds == 6e-6
        assert all(
            row == combined.to_dict() for row in bound.gather(combined.to_dict())
        )
        # Repetition keeps finite inputs, exact logical order and bounded buffers.
        for request in requests:
            if bound.logical_rank not in request.ranks:
                continue
            probe = CollectiveProbe(
                request, bound.for_members(request.ranks), torch.device("cpu")
            )
            same_storage = (
                probe.input.untyped_storage()._cdata
                == probe.output.untyped_storage()._cdata
            )
            assert same_storage == request.inplace
            for _ in range(4):
                assert probe.sample() > 0
            if request.operation == "all_gather":
                expected = (
                    torch.tensor(request.ranks)
                    .add(1)
                    .float()
                    .repeat_interleave(16 // len(request.ranks))
                )
            else:
                expected = torch.full_like(
                    probe.output, sum(rank + 1 for rank in request.ranks)
                )
            torch.testing.assert_close(probe.output, expected)
        dist.barrier()
        rows = measure_collectives(
            requests,
            device=torch.device("cpu"),
            setup={"test_backend": "gloo"},
            options=ProfilingOptions(
                conditioning_seconds=0.001,
                conditioning_wall_seconds=0.1,
                measurement_seconds=0.002,
                measurement_wall_seconds=0.2,
            ),
            bindings=bound,
            backend_setup=backend,
        )
        assert all(peer == rows for peer in bound.gather(rows))
        for row in rows:
            peers = row["per_rank"]
            assert len({peer["samples"] for peer in peers}) == 1
            assert len(peers) == len(row["ranks"])
            assert row["duration_seconds"] > 0
            info = CommunicationInfo(
                1e9,
                1e9,
                topology=bound.topology,
                measurements=(row,),
                metadata={"backend_setup": backend},
            )
            cost = collective_cost(
                info,
                **{
                    key: row[key]
                    for key in (
                        "operation",
                        "ranks",
                        "tensor_bytes",
                        "dtype",
                        "setup",
                        "inplace",
                    )
                },
            )
            assert cost.source == "measurement"
            assert cost.duration_seconds == row["duration_seconds"]
        # Stop decisions are coordinated even when only one worker wants more.
        assert bound.continue_sampling(worker == 0, True)
        assert not bound.continue_sampling(True, worker != 2)
        bound.close()
        bound = None
    finally:
        if bound is not None:
            bound.close()
        dist.destroy_process_group()


def test_collective_sampling_and_logical_groups_across_real_workers(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)


def _gpu_worker(worker, rendezvous, output):
    from dataclasses import replace

    import torch
    import torch.distributed as dist

    from shadowspill import CommunicationInfo, Parallelism, measure_communication
    from shadowspill.communication.requests import collective_requests
    from shadowspill.pytorch.toolchain import build_setup
    from tests.shadowspill.offline.test_sharded_forward import forward

    device = torch.device("cuda", worker)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        world_size=2,
        rank=worker,
        timeout=timedelta(seconds=60),
    )
    try:
        source = forward()
        shard = source.bindings["weight_shards"]["weight_storage"]
        source = replace(
            source,
            setup=build_setup(device),
            parallelism=Parallelism.dp(ranks_per_replica=2),
            bindings={
                **source.bindings,
                "weight_shards": {
                    "weight_storage": {**shard, "groups": ((1, 0),)},
                },
            },
        )
        requests = [request.to_dict() for request in collective_requests(source, None)]
        requests.extend(
            dict(
                operation=operation,
                ranks=(1, 0),
                tensor_bytes=size,
                dtype="torch.float32",
                inplace=inplace,
            )
            for size in (4096, 1 << 20, 16 << 20)
            for operation, inplace in (
                ("all_gather", False),
                ("all_gather", True),
                ("reduce_scatter", False),
                ("reduce_scatter", True),
                ("all_reduce", True),
            )
        )
        info = measure_communication(
            program=source,
            collectives=requests,
            rank_map={0: 1, 1: 0},
            device=device,
            sizes_bytes=(1024, 1 << 20, 16 << 20),
            warmup_seconds=0.005,
            measure_seconds=0.01,
        )
        assert len(info.measurements) == len(requests)
        for observation in info.measurements:
            assert observation["ranks"] == (1, 0)
            assert observation["duration_seconds"] > 0
            if observation["operation"] == "reduce_scatter" and observation["inplace"]:
                assert observation["zero_duration_seconds"] > 0
            memory = observation["memory"]
            assert memory["memory_kind"] == "measured", memory
            assert memory["workspace_bytes"] >= 0
            assert {row["logical_rank"] for row in memory["per_rank"]} == {0, 1}
            assert all("allocation_contract" in row for row in memory["per_rank"])
        assert info.topology == {"node0": (0, 1)}
        assert info.metadata["backend_setup"]["nccl_version"]
        assert info.measurements[0]["backend_setup"] == info.metadata["backend_setup"]
        rows = [None, None]
        dist.all_gather_object(rows, info.to_dict())
        assert rows[0] == rows[1]
        if worker == 0:
            info.save(output)
            assert CommunicationInfo.load(output).to_dict() == info.to_dict()
            print(
                "NCCL calibration:",
                len(info.measurements),
                "requests; retained bytes:",
                info.fixed_device_bytes,
                "; per rank:",
                info.metadata["memory_per_rank"],
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def test_nccl_calibration_on_two_gpus(tmp_path):
    import pytest
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs to qualify NCCL calibration")
    from time import monotonic

    workers = mp.spawn(
        _gpu_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "communication.ssp")),
        nprocs=2,
        join=False,
    )
    try:
        deadline = monotonic() + 180
        while not workers.join(timeout=1):
            if monotonic() >= deadline:
                pytest.fail("NCCL calibration workers exceeded their deadline")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
        for process in workers.processes:
            process.join(timeout=5)
