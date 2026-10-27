"""Aggregate matching samples across ranks and write one case result."""

from baselines.common import write_json

from .config import summarize


def write_result(args, shape, request, gathered):
    timing = summarize([[s["wall_seconds"] for s in r["samples"]] for r in gathered])
    gpu = summarize([[s["gpu_seconds"] for s in r["samples"]] for r in gathered])
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
            "effective_tflops_per_gpu": flops[phase] / summary["median_seconds"] / 1e12,
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
                str(item["request"]["rank"]): item["memory"] for item in gathered
            },
            "effective_tflops_per_gpu": flops["total"]
            / timing["median_seconds"]
            / 1e12,
            "gpu_event_tflops_per_gpu": flops["total"] / gpu["median_seconds"] / 1e12,
            "tokens_per_second_per_gpu": args.tokens / timing["median_seconds"],
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in gathered),
            "all_ranks_validated": len(gathered),
            "performance_result": not args.tiny,
        },
    )
