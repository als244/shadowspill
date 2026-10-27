"""Trace one compiled graph pair on the single GPU visible to this process."""

from __future__ import annotations

import argparse
import importlib
import os
import time
import traceback
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

from benchmarking._serialization import atomic_json
from shadowspill.build.compilation import compile_task
from shadowspill.pytorch.executable import restore_executable

from .inputs import materialize


def restored(cached):
    with FakeTensorMode():
        pair = cached.restore()

    # FakeTensorProp owns the graph node values. Keep compiler examples in
    # that same mode instead of the separate metadata reconstruction mode.
    def consistent(artifact):
        return replace(
            artifact,
            example_arguments=tuple(
                n.meta["val"]
                for n in artifact.graph_module.graph.nodes
                if n.op == "placeholder"
            ),
        )

    return replace(
        pair, forward=consistent(pair.forward), backward=consistent(pair.backward)
    )


def describe(values):
    return [
        {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "stride": list(v.stride()),
            "device": str(v.device),
        }
        if isinstance(v, torch.Tensor)
        else {"value": repr(v)}
        for v in values
    ]


def run(args):
    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    import mlops

    mlops.providers.ensure_implementations_registered()
    # Keep isolated trace captures reproducible across worker processes.
    import torch._inductor.config as inductor

    with inductor.patch({"deterministic": True, "reorder_for_peak_memory": False}):
        return trace(args)


@torch.no_grad()
def trace(args):
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    for module in payload.get("registration_modules", ()):
        importlib.import_module(module)
    pair = restored(payload["pair"])
    print("Materializing isolated forward inputs", flush=True)
    inputs = materialize(pair.forward, payload["policies"])
    print("Compiling forward", flush=True)
    forward_task = compile_task(pair.forward)
    forward = restore_executable(
        forward_task.payload, dependencies=forward_task.dependencies
    )
    del forward_task
    print("Running forward to obtain authentic saved tensors", flush=True)
    output = forward(*inputs)
    leaves, _ = tree_flatten(output)
    original_count = pair.forward.output_count - pair.saved_value_count
    saved = tuple(leaves[original_count:])
    if len(saved) != pair.saved_value_count:
        raise ValueError("forward saved-value arity changed")
    # VJPs take the actual forward's saved tensors and deterministic cotangents.
    tangents = []
    for _index, example in enumerate(
        pair.backward.example_arguments[pair.saved_value_count :]
    ):
        if not example.is_floating_point():
            raise ValueError("unexpected non-floating cotangent")
        tangent = torch.empty_strided(
            tuple(example.shape),
            tuple(example.stride()),
            dtype=example.dtype,
            device=example.device,
        )
        tangent.normal_(std=0.05)
        tangents.append(tangent)
    backward_inputs = (*saved, *tangents)
    print("Compiling backward", flush=True)
    backward_task = compile_task(
        replace(pair.backward, example_arguments=backward_inputs)
    )
    backward = restore_executable(
        backward_task.payload, dependencies=backward_task.dependencies
    )
    del backward_task
    del output, leaves, saved, backward_inputs
    result = None
    output = None
    samples = []

    def iteration(index, traced):
        nonlocal output, result
        output = result = None
        torch.cuda.synchronize()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        nvtx = torch.cuda.nvtx.range if traced else lambda _: nullcontext()
        with nvtx(f"iteration/{index}"):
            events[0].record()
            with nvtx("forward"):
                output = forward(*inputs)
            events[1].record()
            values, _ = tree_flatten(output)
            authentic = values[original_count:]
            with nvtx("backward"):
                result = backward(*authentic, *tangents)
            events[2].record()
        torch.cuda.synchronize()
        return {
            "iteration": index,
            "forward_ms": events[0].elapsed_time(events[1]),
            "backward_ms": events[1].elapsed_time(events[2]),
            "total_ms": events[0].elapsed_time(events[2]),
        }

    for index in range(args.warmup):
        item = iteration(index, False)
        print(
            f"Warmup {index + 1}/{args.warmup}: {item['total_ms']:.3f} ms", flush=True
        )
    output = result = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if args.capture:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        label = (
            f"stage/{args.stage}/variant/{payload['variant']}/"
            f"seqlen/{payload['workload']['sequence_length']}/"
            f"batch/{payload['workload']['batch_size']}"
        )
        with torch.cuda.nvtx.range(label):
            for index in range(args.iterations):
                samples.append(iteration(index, True))
    finally:
        torch.cuda.synchronize()
        if args.capture:
            torch.cuda.cudart().cudaProfilerStop()
    # Check user-visible forward values and the produced gradients, not masks
    # or other saved intermediates that may legitimately contain infinity.
    forward_leaves, _ = tree_flatten(output)
    gradient_leaves, _ = tree_flatten(result)
    for phase, values in (
        ("forward", forward_leaves[:original_count]),
        ("backward", gradient_leaves),
    ):
        for i, value in enumerate(values):
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"nonfinite {phase} output {i}")
    record = {
        "status": "completed",
        "stage": args.stage,
        "variant": payload["variant"],
        "artifact": str(args.artifact),
        "workload": payload["workload"],
        "samples": samples,
        "profiled": args.capture,
        "timing_note": "CUDA events under Nsight; diagnostic, not sweep throughput",
        "device": str(torch.cuda.get_device_properties(0)),
        "device_identity": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_ordinal": 0,
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "name": torch.cuda.get_device_properties(0).name,
        },
        "forward_inputs": describe(inputs),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "forward_digest": pair.forward.compatibility_digest,
        "backward_digest": pair.backward.compatibility_digest,
        "compiler_settings": {"deterministic": True, "reorder_for_peak_memory": False},
        "finite_outputs_and_gradients": True,
    }
    atomic_json(args.output / "result.json", record)
    print(
        f"Completed {args.stage}/{payload['variant']}: {args.iterations} iterations",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    begun = time.time()
    try:
        run(args)
    except BaseException as error:
        atomic_json(
            args.output / "failure.json",
            {
                "status": "out_of_memory"
                if isinstance(error, torch.OutOfMemoryError)
                else "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - begun,
            },
        )
        raise


if __name__ == "__main__":
    main()
