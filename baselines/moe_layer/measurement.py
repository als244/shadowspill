"""Aligned warmup inputs and synchronized forward/backward timing."""

import os
import time

from .request import fingerprint


def prepare_tuning_inputs(x, dy, request, world):
    import torch.distributed as dist

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
        request["quack_tuning"]["inputs"] = "rank-zero activations and output gradient"
    return tuning_originals


def restore_inputs(x, dy, originals, request):
    import torch

    with torch.no_grad():
        x.copy_(originals[0])
        dy.copy_(originals[1])
    if (
        fingerprint(x) != request["input_fingerprint"]
        or fingerprint(dy) != request["gradient_fingerprint"]
    ):
        raise AssertionError("measurement inputs were not restored")


def measure_iteration(implementation, x, dy, *, iteration, warmup, world):
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
        torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    if iteration == warmup:
        torch.cuda.reset_peak_memory_stats()
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
        "warmup": iteration < warmup,
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
    return record, output
