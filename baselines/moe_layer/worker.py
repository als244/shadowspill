"""Measure one router/expert forward-backward case on one GPU or an EP group."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from baselines.common import write_json

from .config import add_shape_arguments, requested_shape, summarize
from .implementations.registry import BACKENDS, build


def main(argv=None, *, session=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=list(BACKENDS), required=True)
    parser.add_argument("--precision", choices=["bf16", "fp8_current"], required=True)
    parser.add_argument("--training", choices=["full", "lora"], required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--tiny", action="store_true", help="small qualification geometry"
    )
    parser.add_argument(
        "--sonicmoe-candidate-policy",
        choices=["baseline", "expanded", "expanded-k"],
        default="baseline",
    )
    add_shape_arguments(parser)
    args = parser.parse_args(argv)
    if min(args.tokens, args.warmup, args.iterations, args.cpu_threads) < 1:
        parser.error("token, step and thread counts must be positive")
    # CUDA libraries must see the selected device before their first import.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank if len(visible) > 1 else 0]
    import_started = time.perf_counter()
    import torch
    import torch.distributed as dist

    phase_seconds = {"imports": time.perf_counter() - import_started}
    cuda_started = time.perf_counter()
    torch.cuda.set_device(0)
    phase_seconds["cuda_setup"] = time.perf_counter() - cuda_started
    torch.set_num_threads(args.cpu_threads)
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
        versions = {}
        for name in (
            "torch",
            "mlops",
            "sonic-moe",
            "quack-kernels",
            "moonep",
            "nvidia-cutlass-dsl",
            "transformer-engine",
            "megatron-core",
            "megatron-bridge",
            "deep-ep",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None

        def fingerprint(value):
            sample = value[:16, :16].detach().contiguous().cpu()
            return hashlib.sha256(
                sample.view(torch.uint16).numpy().tobytes()
            ).hexdigest()

        request = {
            **vars(args),
            "output": str(args.output),
            "rank": rank,
            "world_size": world,
            "process_id": os.getpid(),
            "process_group_reused": group_reused,
            "model_reused": False if session is None else session.reused,
            "model_instance": id(implementation.module),
            "shape": asdict(shape),
            "device": str(torch.cuda.get_device_properties(0)),
            "hostname": platform.node(),
            "versions": versions,
            "flops_per_rank": shape.flops(args.tokens, args.training),
            "gradient_dtype": "float32",
            "quack_tuning": {
                "distributed": os.environ.get("MLOPS_QUACK_TUNE_DISTRIBUTED") == "1",
                "warmup_ms": float(os.environ.get("MLOPS_QUACK_TUNE_WARMUP_MS", "50")),
                "force_cache_update": bool(os.environ.get("QUACK_FORCE_CACHE_UPDATE")),
                "cache_dir": os.environ.get("QUACK_CACHE_DIR", str(Path.home() / ".quack/cache")),
                "timed_ms": float(os.environ.get("MLOPS_QUACK_TUNE_TIME_MS", "180")),
                "timing_windows": int(os.environ.get("MLOPS_QUACK_TUNE_SAMPLES", "3")),
            },
            "input_dtype": "bfloat16",
            "lora_compute_dtype": "bfloat16",
            "lora_alpha": shape.lora_alpha,
            "implementation": implementation.metadata,
            "routing": "real top-k then renormalized softmax",
            "input_seed": args.seed + 100 + rank,
            "gradient_seed": args.seed + 200 + rank,
            "input_fingerprint": fingerprint(x),
            "gradient_fingerprint": fingerprint(dy),
            "memory_measurement": "extra warmed untimed forward/backward pass",
            "rank_synchronization": (
                "barriers before and after each sample, outside timing"
            ),
            "auxiliary_loss": False,
            "activation_transport": "bf16",
            "optimizer": False,
            "dp_gradient_sync": False,
            "ep_router_sync": False,
            "timing": (
                "forward and backward; host wall plus GPU events; "
                "slowest rank per iteration"
            ),
            "performance_result": not args.tiny,
        }
        tuning_originals = None
        if os.environ.get("MLOPS_QUACK_TUNE_DISTRIBUTED") == "1" and world > 1:
            # Identical activations produce identical routing on replicated DP
            # weights. Preserve each rank's actual benchmark inputs on CPU.
            tuning_originals = (
                x.detach().to("cpu", copy=True),
                dy.to("cpu", copy=True),
            )
            dist.broadcast(x.detach(), src=0)
            dist.broadcast(dy, src=0)
            tuning_fingerprints = (fingerprint(x), fingerprint(dy))
            peers = [None] * world
            dist.all_gather_object(peers, tuning_fingerprints)
            if len(set(peers)) != 1:
                raise AssertionError("tuning inputs differ across ranks")
            request["quack_tuning"]["activation_fingerprint"] = tuning_fingerprints[0]
            request["quack_tuning"]["gradient_fingerprint"] = tuning_fingerprints[1]
            request["quack_tuning"]["inputs"] = (
                "rank-zero activations and output gradient"
            )
        write_json(root / "request.json", request)
        samples = []
        output = None
        stage = "warmup"
        phase_seconds["warmup"] = phase_seconds["measurement"] = 0.0
        for iteration in range(args.warmup + args.iterations):
            iteration_started = time.perf_counter()
            if iteration == args.warmup and tuning_originals is not None:
                with torch.no_grad():
                    x.copy_(tuning_originals[0])
                    dy.copy_(tuning_originals[1])
                tuning_originals = None
                if (
                    fingerprint(x) != request["input_fingerprint"]
                    or fingerprint(dy) != request["gradient_fingerprint"]
                ):
                    raise AssertionError("measurement inputs were not restored")
            implementation.zero_grad()
            x.grad = None
            output = None
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier()
                torch.cuda.synchronize()
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            if iteration == args.warmup:
                torch.cuda.reset_peak_memory_stats()
                stage = "measurement"
            stamp = time.perf_counter()
            events[0].record()
            output = implementation.forward(x)
            events[1].record()
            implementation.backward(output, dy)
            events[2].record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - stamp
            record = {
                "iteration": iteration,
                "warmup": iteration < args.warmup,
                "wall_seconds": wall,
                "gpu_seconds": events[0].elapsed_time(events[2]) / 1000,
                "forward_gpu_seconds": events[0].elapsed_time(events[1]) / 1000,
                "backward_gpu_seconds": events[1].elapsed_time(events[2]) / 1000,
            }
            # Match measurement boundaries across DP and EP ranks, without
            # adding replicated-gradient synchronization to this layer benchmark.
            if world > 1:
                dist.barrier()
                torch.cuda.synchronize()
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
                    f"{iteration + 1}: {wall * 1000:.3f} ms",
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
        from .validation import all_finite

        for name, value in [
            ("output", output),
            ("input gradient", x.grad),
            *implementation.named_gradients(),
        ]:
            if value is None or not all_finite(value):
                raise AssertionError(f"missing or nonfinite {name}")
            if (
                name not in {"output", "input gradient"}
                and value.dtype != torch.float32
            ):
                raise AssertionError(f"{name} gradient is {value.dtype}, expected FP32")
        if args.training == "lora" and any(
            p.grad is not None
            for _, p in implementation.named_parameters()
            if not p.requires_grad
        ):
            raise AssertionError("frozen base received parameter gradients")
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
            timing = summarize(
                [[s["wall_seconds"] for s in r["samples"]] for r in gathered]
            )
            gpu = summarize(
                [[s["gpu_seconds"] for s in r["samples"]] for r in gathered]
            )
            flops = shape.flops(args.tokens, args.training)
            phases = {}
            for phase, key in (
                ("forward", "forward_gpu_seconds"),
                ("backward", "backward_gpu_seconds"),
                ("total", "gpu_seconds"),
            ):
                summary = summarize(
                    [[sample[key] for sample in rank["samples"]] for rank in gathered]
                )
                phases[phase] = {
                    **summary,
                    "effective_tflops_per_gpu": flops[phase]
                    / summary["median_seconds"]
                    / 1e12,
                }
            write_json(
                args.output / "result.json",
                {
                    "status": "completed",
                    "request": request,
                    "timing": timing,
                    "gpu_timing": gpu,
                    "phase_timings": phases,
                    "memory_by_rank": {
                        str(item["request"]["rank"]): item["memory"]
                        for item in gathered
                    },
                    "effective_tflops_per_gpu": flops["total"]
                    / timing["median_seconds"]
                    / 1e12,
                    "gpu_event_tflops_per_gpu": flops["total"]
                    / gpu["median_seconds"]
                    / 1e12,
                    "tokens_per_second_per_gpu": args.tokens / timing["median_seconds"],
                    "peak_allocated_bytes": max(
                        r["peak_allocated_bytes"] for r in gathered
                    ),
                    "all_ranks_validated": len(gathered),
                    "performance_result": not args.tiny,
                },
            )
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
