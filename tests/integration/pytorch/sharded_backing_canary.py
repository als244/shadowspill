"""torchrun canary: native reduce-scatter/offload/reload across repeated passes."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import Runtime
from shadowspill.pytorch.distributed._optimizer import UpdateLayout, _gather, _gradient
from shadowspill.runtime.abi import ObjectBinding, ObjectDescription, runtime_library
from shadowspill.runtime.timing import Marker
from shadowspill.runtime.trace import (
    begin_runtime_trace,
    end_runtime_trace,
    prepare_runtime_trace,
    read_runtime_trace,
)
from tests.integration.pytorch.overlap_canary import (
    _action,
    _admit_task,
    _bind_plan_object,
    _create_plan,
    _require_ok,
    _submit_actions,
    _wait_idle,
)
from tests.integration.pytorch.runtime_helpers import begin_task
from workloads.recipes.text.distributed import data_parallel_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    ordinal = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    with Runtime(
        pools={
            "execution": device(device=ordinal, physical_capacity=2 << 30),
            "spill": pinned_host(capacity=64 << 20),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
        calibrate=False,
        control_group=dist.group.WORLD,
    ) as runtime:
        specification, resources = data_parallel_groups(
            device=torch.device(f"cuda:{ordinal}"),
            shard_size=args.shard_size,
            shard_gradients=True,
            timeout=180,
        )
        with resources():
            sharding = specification.transfer_sharding
            assert sharding is not None
            group, replicas = sharding.group, sharding.replica_group
            local_rank = dist.get_rank(group)
            library, core = runtime._installed.library, runtime_library()
            plan = _create_plan(library)
            stream = torch.cuda.current_stream().cuda_stream
            object_id = 4001
            logical = 65539
            capacity = (logical + args.shard_size - 1) // args.shard_size
            padded = capacity * args.shard_size
            gradient = torch.zeros(
                padded, dtype=torch.float32, device=f"cuda:{ordinal}"
            )
            descriptor = torch.ops.shadowspill._create_sharded_transfer(
                group.group_name,
                ordinal,
                padded * 4,
                torch.float32,
                None if replicas is None else replicas.group_name,
                logical * 4,
            )
            try:
                _require_ok(
                    core.shadowspill_register_object(
                        runtime._runtime_handle,
                        ctypes.byref(
                            ObjectDescription(
                                object_id=object_id,
                                size_bytes=padded * 4,
                                transfer=descriptor,
                            )
                        ),
                    ),
                    "register sharded gradient",
                )
            finally:
                torch.ops.shadowspill._close_sharded_transfer(descriptor)
            _bind_plan_object(library, plan, object_id)
            try:
                _admit_task(
                    library,
                    plan,
                    99,
                    (),
                    (_action(object_id, 3, "invalid.write_back"),),
                )
            except AssertionError as error:
                assert "task admission" in str(error)
            else:
                raise AssertionError(
                    "in-place gradient reduction accepted retained write-back"
                )
            binding = ObjectBinding()
            _require_ok(
                core.shadowspill_plan_publish_initial_allocation(
                    plan, object_id, gradient.data_ptr(), ctypes.byref(binding)
                ),
                "publish gradient",
            )
            # Optimizer ownership spans the full DP group independently of the
            # (possibly smaller) sharding groups. Include real FP32 masters and
            # Adam moments, with BF16 computation weights published after step.
            layout = UpdateLayout(
                (logical,),
                torch.bfloat16,
                specification.group.group_name,
                world,
                rank,
                master=True,
                gradient_group_name=specification.group.group_name,
                gradient_group_size=world,
            )
            initial = ((torch.arange(logical) % 29).float() - 14) / 32
            oracle = torch.nn.Parameter(initial.clone())
            optimizer_capacity = layout.capacity
            padded_initial = torch.nn.functional.pad(
                initial, (0, optimizer_capacity * world - logical)
            )
            master = torch.nn.Parameter(
                padded_initial.narrow(0, layout.start, optimizer_capacity).cuda()
            )
            options = dict(
                lr=0.005, betas=(0.8, 0.95), eps=1e-4, weight_decay=0.01, foreach=False
            )
            optimizer = torch.optim.AdamW([master], **options)
            oracle_optimizer = torch.optim.AdamW([oracle], **options)
            pattern = ((torch.arange(logical) % 17).float() - 8) / 64
            gpu_pattern = pattern.cuda()
            valid = max(0, min(capacity, logical - local_rank * capacity))
            origin = None
            if args.trace_output is not None:
                torch.cuda.synchronize()
                origin = Marker(runtime._runtime_handle)
                prepare_runtime_trace(
                    runtime._runtime_handle,
                    event_capacity=65536,
                    allocation_event_capacity=65536,
                )
                origin.record(stream)
                begin_runtime_trace(runtime._runtime_handle, step_id=1, origin=origin)
                torch.ops.shadowspill._collective_trace_begin(4096)
            for step in range(args.steps):
                gradient.zero_()
                expected = torch.zeros(logical)
                for pass_index in range(args.passes):
                    for microbatch in range(2):
                        coefficient = (step + 1) * (pass_index + 1) * (microbatch + 1)
                        # The loss has one global normalization. Collective
                        # reductions are SUM; there is no extra group average.
                        gradient[:logical].add_(
                            gpu_pattern, alpha=coefficient * (rank + 1) / world
                        )
                        expected.add_(pattern, alpha=coefficient * (world + 1) / 2)
                    if rank == pass_index % world:
                        time.sleep(0.025)
                    boundary = 100 + (step * args.passes + pass_index) * 3
                    torch.ops.shadowspill._dematerialize_storages([gradient])
                    _submit_actions(
                        library,
                        plan,
                        boundary,
                        stream,
                        (_action(object_id, 1, "gradient.scatter_offload"),),
                    )
                    _require_ok(_wait_idle(library), "eviction completion")
                    shard = torch.empty(capacity, dtype=torch.float32)
                    _require_ok(
                        core.shadowspill_read_object(
                            runtime._runtime_handle,
                            object_id,
                            1,
                            shard.data_ptr(),
                            capacity * 4,
                        ),
                        "read physical shard",
                    )
                    reference = torch.nn.functional.pad(expected, (0, padded - logical))
                    torch.testing.assert_close(
                        shard,
                        reference.narrow(0, local_rank * capacity, capacity),
                        rtol=0,
                        atol=0,
                    )
                    _submit_actions(
                        library,
                        plan,
                        boundary + 1,
                        stream,
                        (_action(object_id, 2, "gradient.owned_carry"),),
                    )
                    task = _admit_task(library, plan, boundary + 2, (object_id,), ())
                    rebound = begin_task(
                        library, task, boundary + 2, stream, expected_bindings=1
                    )
                    torch.ops.shadowspill._acquire_storages(
                        [gradient], [rebound[0].pointer]
                    )
                    carry = torch.zeros(padded)
                    if replicas is None or dist.get_rank(replicas) == 0:
                        start = local_rank * capacity
                        carry[start : start + valid] = expected[start : start + valid]
                    torch.testing.assert_close(gradient.cpu(), carry, rtol=0, atol=0)
                    _require_ok(
                        library.shadowspill_pytorch_after_task_handle(task, stream),
                        "end pass",
                    )
                # Exercise production optimizer reduction/publication helpers
                # against a complete, unsharded optimizer and state history.
                master.grad = _gradient(gradient[:logical], layout)
                full_expected = torch.nn.functional.pad(
                    expected, (0, optimizer_capacity * world - logical)
                )
                torch.testing.assert_close(
                    master.grad.cpu(),
                    full_expected.narrow(0, layout.start, optimizer_capacity),
                    rtol=0,
                    atol=0,
                )
                oracle.grad = expected
                optimizer.step()
                oracle_optimizer.step()
                actual = _gather(master, layout).cpu()
                torch.testing.assert_close(
                    actual, oracle.detach().to(torch.bfloat16), rtol=0, atol=0
                )
                for name in ("exp_avg", "exp_avg_sq"):
                    state = oracle_optimizer.state[oracle][name]
                    state = torch.nn.functional.pad(
                        state, (0, optimizer_capacity * world - logical)
                    )
                    torch.testing.assert_close(
                        optimizer.state[master][name].cpu(),
                        state.narrow(0, layout.start, optimizer_capacity),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                assert optimizer.state[master]["step"].item() == step + 1
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "step": step + 1,
                            "event": "step_matches_unsharded",
                        }
                    ),
                    flush=True,
                )
            if origin is not None:
                torch.cuda.synchronize()
                _require_ok(_wait_idle(library), "trace completion")
                collective_trace = torch.ops.shadowspill._collective_trace_end()
                end_runtime_trace(runtime._runtime_handle)
                runtime_trace = read_runtime_trace(runtime._runtime_handle)
                origin.release()
                expected_evictions = args.steps * args.passes
                expected_replicas = expected_evictions if replicas is not None else 0
                assert len(collective_trace["records"]) == (
                    expected_evictions + expected_replicas + 2 * args.steps
                ), collective_trace
                assert not any(
                    collective_trace[name]
                    for name in ("dropped", "errors", "missing_timings")
                ), collective_trace
                for record in collective_trace["records"]:
                    assert record["issued_at_ns"] is not None
                    assert 0 <= record["started_at_ns"] <= record["finished_at_ns"]
                    if record["source"] == "evict":
                        assert record["object_id"] == object_id
                        assert record["task_id"] >= 100
                        completion = next(
                            event
                            for event in runtime_trace.events
                            if event.kind.name == "TRANSFER_COMPLETED"
                            and event.task_id == record["task_id"]
                            and event.object_id == object_id
                            and event.detail_0 == 1
                        )
                        assert completion.lane_started_at_ns <= record["started_at_ns"]
                        assert completion.bytes == capacity * 4
                        assert (
                            record["finished_at_ns"] <= completion.lane_finished_at_ns
                        )
                args.trace_output.mkdir(parents=True, exist_ok=True)
                (args.trace_output / f"collectives-rank-{rank}.json").write_text(
                    json.dumps(
                        {
                            "collectives": collective_trace,
                            "runtime_events": [
                                event.as_dict() for event in runtime_trace.events
                            ],
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    json.dumps({"rank": rank, "collective_trace_passed": True}),
                    flush=True,
                )
                # A delayed compute stream must delay GPU start, not inflate
                # collective execution by treating host issue as start. Also
                # exercise the bounded buffer and a second independent session.
                torch.cuda.synchronize()
                second_origin = Marker(runtime._runtime_handle)
                second_origin.record(stream)
                begin_runtime_trace(
                    runtime._runtime_handle, step_id=2, origin=second_origin
                )
                torch.ops.shadowspill._collective_trace_begin(1)
                torch.cuda._sleep(100_000_000)
                _gather(master, layout)
                _gather(master, layout)
                torch.cuda.synchronize()
                overflow = torch.ops.shadowspill._collective_trace_end()
                end_runtime_trace(runtime._runtime_handle)
                second_origin.release()
                assert len(overflow["records"]) == 1 and overflow["dropped"] == 1
                assert overflow["errors"] == overflow["missing_timings"] == 0
                queued = overflow["records"][0]
                assert queued["started_at_ns"] - queued["issued_at_ns"] > 1_000_000
                (args.trace_output / f"overflow-rank-{rank}.json").write_text(
                    json.dumps(overflow, indent=2) + "\n"
                )
                _gather(master, layout)
                torch.cuda.synchronize()
                assert torch.ops.shadowspill._collective_trace_end() == {}
                print(
                    json.dumps(
                        {"rank": rank, "collective_queue_and_overflow_passed": True}
                    ),
                    flush=True,
                )
            torch.ops.shadowspill._dematerialize_storages([gradient])
            _submit_actions(
                library, plan, 999, stream, (_action(object_id, 0, "release"),)
            )
            _require_ok(_wait_idle(library), "release completion")
            _require_ok(core.shadowspill_plan_close(plan), "plan close")
            core.shadowspill_plan_destroy(plan)
            _require_ok(
                core.shadowspill_unregister_object(runtime._runtime_handle, object_id),
                "unregister",
            )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "passes": args.passes,
                        "steps": args.steps,
                        "transfer_shard_size": args.shard_size,
                        "optimizer_shard_size": world,
                        "microbatches_per_pass": 2,
                        "logical_elements": logical,
                        "cpu_bytes": capacity * 4,
                        "gpu_bytes": padded * 4,
                        "passed": True,
                    }
                ),
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
