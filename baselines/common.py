"""Workload geometry and measurements shared by standalone baselines."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def accumulation_steps(global_tokens, sequence_length, microbatch_tokens, world_size):
    if min(global_tokens, sequence_length, microbatch_tokens, world_size) < 1:
        raise ValueError("token counts and world size must be positive")
    if microbatch_tokens % sequence_length:
        raise ValueError("microbatch tokens must contain whole sequences")
    local_step, remainder = divmod(global_tokens, world_size)
    if remainder or local_step % microbatch_tokens:
        raise ValueError("global tokens must divide into complete rank microbatches")
    return local_step // microbatch_tokens


def recompute_layers(layer_count, fraction):
    """Spread the selected checkpointed layers evenly through the model."""
    if layer_count < 1 or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("recompute fraction must be in [0, 1]")
    count = round(layer_count * fraction)
    return tuple(
        i
        for i in range(layer_count)
        if (i + 1) * count // layer_count > i * count // layer_count
    )


def summarize_cycles(rank_cycles, *, global_tokens, useful_flops_per_step=None):
    if not rank_cycles or not rank_cycles[0]:
        raise ValueError("no completed measurement cycles")
    width = len(rank_cycles[0])
    if any(len(row) != width for row in rank_cycles):
        raise ValueError("ranks completed different numbers of steps")
    if any(not math.isfinite(t) or t <= 0 for row in rank_cycles for t in row):
        raise ValueError("invalid cycle time")
    slowest = [max(values) for values in zip(*rank_cycles, strict=True)]
    seconds = statistics.median(slowest)
    return {
        "step_seconds": seconds,
        "global_tokens_per_second": global_tokens / seconds,
        "per_rank_cycle_seconds": rank_cycles,
        "slowest_rank_cycle_seconds": slowest,
        "aggregation": "median of per-step maximum rank time, excluding warmup",
        "useful_tflops_per_gpu": None
        if useful_flops_per_step is None
        else (useful_flops_per_step / seconds / len(rank_cycles) / 1e12),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
