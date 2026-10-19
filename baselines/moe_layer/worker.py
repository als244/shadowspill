"""Measure one router/expert forward-backward case on one GPU or an EP group."""

from __future__ import annotations

import argparse
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

from .config import Shape, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sonicmoe", "quackmoe"], required=True)
    parser.add_argument("--precision", choices=["bf16", "fp8_current"], required=True)
    parser.add_argument("--training", choices=["full", "lora"], required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--tiny", action="store_true", help="small qualification geometry"
    )
    args = parser.parse_args(argv)
    if min(args.tokens, args.warmup, args.iterations, args.cpu_threads) < 1:
        parser.error("token, step and thread counts must be positive")
    # CUDA libraries must see the selected device before their first import.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank if len(visible) > 1 else 0]
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    root = args.output / f"rank-{rank:05d}"
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    layer = buffer = None
    stage = "initialization"
    try:
        if args.backend == "quackmoe":
            dist.init_process_group(
                "nccl", device_id=torch.device("cuda", 0), timeout=timedelta(minutes=10)
            )
        elif world != 1:
            raise ValueError("SonicMoE DP compute benchmark requires world size one")
        from .layers import build, tensor

        shape = (
            Shape(
                width=128,
                hidden=128,
                experts=max(world, 8),
                top_k=2,
                lora_rank=args.lora_rank,
            )
            if args.tiny
            else Shape(lora_rank=args.lora_rank)
        )
        torch.manual_seed(args.seed + rank)
        layer, buffer = build(
            shape,
            backend=args.backend,
            precision=args.precision,
            training=args.training,
            tokens=args.tokens,
            world_size=world,
            seed=args.seed,
        )
        x = tensor(
            (args.tokens, shape.width), seed=args.seed + 100 + rank, std=1.0
        ).requires_grad_()
        dy = tensor(tuple(x.shape), seed=args.seed + 200 + rank, std=0.05)
        trainable = [
            (name, p) for name, p in layer.named_parameters() if p.requires_grad
        ]
        versions = {}
        for name in (
            "torch",
            "mlops",
            "sonic-moe",
            "quack-kernels",
            "moonep",
            "nvidia-cutlass-dsl",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        request = {
            **vars(args),
            "output": str(args.output),
            "rank": rank,
            "world_size": world,
            "shape": asdict(shape),
            "device": str(torch.cuda.get_device_properties(0)),
            "hostname": platform.node(),
            "versions": versions,
            "flops_per_rank": shape.flops(args.tokens, args.training),
            "gradient_dtype": "float32",
            "input_dtype": "bfloat16",
            "lora_compute_dtype": "bfloat16",
            "lora_alpha": shape.lora_rank,
            "routing": "real top-k then renormalized softmax",
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
        write_json(root / "request.json", request)
        samples = []
        stage = "warmup"
        for iteration in range(args.warmup + args.iterations):
            layer.zero_grad(set_to_none=True)
            x.grad = None
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
            output = layer(x)
            events[1].record()
            output.backward(dy)
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
        stage = "validation"
        for name, value in [
            ("output", output),
            ("input gradient", x.grad),
            *[(name, p.grad) for name, p in trainable],
        ]:
            if value is None or not bool(torch.isfinite(value).all()):
                raise AssertionError(f"missing or nonfinite {name}")
            if (
                name not in {"output", "input gradient"}
                and value.dtype != torch.float32
            ):
                raise AssertionError(f"{name} gradient is {value.dtype}, expected FP32")
        if args.training == "lora" and any(
            p.grad is not None for p in layer.parameters() if not p.requires_grad
        ):
            raise AssertionError("frozen base received parameter gradients")
        local = {
            "status": "completed",
            "request": request,
            "samples": samples,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
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
            write_json(
                args.output / "result.json",
                {
                    "status": "completed",
                    "request": request,
                    "timing": timing,
                    "gpu_timing": gpu,
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
        write_json(
            root / "failure.json",
            {
                "status": "out_of_memory"
                if isinstance(error, torch.OutOfMemoryError)
                else "failed",
                "stage": stage,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if layer is not None:
            layer.close()
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
