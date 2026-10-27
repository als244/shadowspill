"""Measure one router/expert forward-backward case on one GPU or an EP group."""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import timedelta

from baselines.common import write_json

from .arguments import parse
from .config import requested_shape
from .implementations.registry import build
from .measurement import measure_iteration, prepare_tuning_inputs, restore_inputs
from .request import describe
from .results import write_result


def main(argv=None, *, session=None):
    args = parse(argv)
    # CUDA libraries must see the selected device before their first import.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank if len(visible) > 1 else 0]
    import_started = time.perf_counter()
    import torch

    phase_seconds = {"imports": time.perf_counter() - import_started}
    cuda_started = time.perf_counter()
    torch.cuda.set_device(0)
    phase_seconds["cuda_setup"] = time.perf_counter() - cuda_started
    torch.set_num_threads(args.cpu_threads)
    run(args, phase_seconds, session=session)


def run(args, phase_seconds, *, session):
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    root = args.output / f"rank-{rank:05d}"
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    implementation = None
    stage = "initialization"
    try:
        setup_started = time.perf_counter()
        group_reused = dist.is_initialized()
        if world > 1 and not group_reused:
            dist.init_process_group(
                "nccl", device_id=torch.device("cuda", 0), timeout=timedelta(minutes=10)
            )
        phase_seconds["distributed_setup"] = time.perf_counter() - setup_started
        if args.backend == "sonicmoe":
            from .candidates import install

            install(args.sonicmoe_candidate_policy, root)
        from .implementations.tensors import tensor

        shape = requested_shape(args, world)
        torch.manual_seed(args.seed + rank)
        setup_started = time.perf_counter()
        constructor = build if session is None else session.acquire
        implementation = constructor(
            args.backend,
            shape=shape,
            precision=args.precision,
            training=args.training,
            tokens=args.tokens,
            world_size=world,
            seed=args.seed,
            router_weight_dtype=args.router_weight_dtype,
        )
        phase_seconds["model_setup"] = time.perf_counter() - setup_started
        setup_started = time.perf_counter()
        x = tensor(
            (args.tokens, shape.width), seed=args.seed + 100 + rank, std=1.0
        ).requires_grad_()
        dy = tensor(tuple(x.shape), seed=args.seed + 200 + rank, std=0.05)
        trainable = [
            (name, p)
            for name, p in implementation.named_parameters()
            if p.requires_grad
        ]
        phase_seconds["input_initialization"] = time.perf_counter() - setup_started
        request = describe(
            args,
            shape,
            implementation,
            x,
            dy,
            rank=rank,
            world=world,
            group_reused=group_reused,
            session=session,
        )
        tuning_originals = prepare_tuning_inputs(x, dy, request, world)
        write_json(root / "request.json", request)
        samples = []
        output = None
        stage = "warmup"
        phase_seconds["warmup"] = phase_seconds["measurement"] = 0.0
        for iteration in range(args.warmup + args.iterations):
            iteration_started = time.perf_counter()
            if iteration == args.warmup and tuning_originals is not None:
                restore_inputs(x, dy, tuning_originals, request)
                tuning_originals = None
            implementation.zero_grad()
            x.grad = None
            output = None
            if iteration == args.warmup:
                stage = "measurement"
            record, output = measure_iteration(
                implementation,
                x,
                dy,
                iteration=iteration,
                warmup=args.warmup,
                world=world,
            )
            phase = "warmup" if record["warmup"] else "measurement"
            phase_seconds[phase] += time.perf_counter() - iteration_started
            if iteration >= args.warmup:
                samples.append(record)
            with (root / "iterations.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if rank == 0:
                print(
                    f"{args.backend} {args.precision} {args.training} {args.tokens} "
                    f"{'warmup' if record['warmup'] else 'sample'} "
                    f"{iteration + 1}: {record['wall_seconds'] * 1000:.3f} ms",
                    flush=True,
                )
        stage = "memory_measurement"
        phase_started = time.perf_counter()
        from .memory import measure

        timing_peak_allocated = torch.cuda.max_memory_allocated()
        timing_peak_reserved = torch.cuda.max_memory_reserved()
        output = None
        if world > 1:
            dist.barrier()
        output, memory = measure(torch, implementation, x, dy)
        phase_seconds["memory_measurement"] = time.perf_counter() - phase_started
        # Persist completed observations before validation allocates any scratch.
        write_json(
            root / "measurements.json",
            {
                "request": request,
                "samples": samples,
                "memory": memory,
                "timing_peak_allocated_bytes": timing_peak_allocated,
                "timing_peak_reserved_bytes": timing_peak_reserved,
                "validation": "pending",
            },
        )
        stage = "validation"
        phase_started = time.perf_counter()
        from .validation import validate_gradients

        validate_gradients(implementation, x, output, args.training)
        phase_seconds["validation"] = time.perf_counter() - phase_started
        local = {
            "status": "completed",
            "phase_seconds": phase_seconds,
            "request": request,
            "samples": samples,
            "memory": memory,
            "peak_allocated_bytes": max(
                timing_peak_allocated,
                *(p["peak_allocated_bytes"] for p in memory.values()),
            ),
            "peak_reserved_bytes": max(
                timing_peak_reserved,
                *(p["peak_reserved_bytes"] for p in memory.values()),
            ),
            "total_seconds": time.perf_counter() - started,
            "finite_outputs_and_gradients": True,
            "trainable_parameters": {name: p.numel() for name, p in trainable},
        }
        write_json(root / "result.json", local)
        gathered = [None] * world
        if world > 1:
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if rank == 0:
            write_result(args, shape, request, gathered)
        if any(
            name == "shadowspill" or name.startswith("shadowspill.")
            for name in __import__("sys").modules
        ):
            raise AssertionError("standalone layer benchmark imported ShadowSpill")
    except BaseException as error:
        from .memory import snapshot

        write_json(
            root / "failure.json",
            {
                "status": "out_of_memory"
                if isinstance(error, torch.OutOfMemoryError)
                else "failed",
                "stage": stage,
                "memory_after_failure": snapshot(torch),
                "phase_seconds": phase_seconds,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if session is None:
            if implementation is not None:
                implementation.close()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
