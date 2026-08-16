"""Registered model construction for Program-corpus workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from models.full_model import (
    FullModelCase,
    build_case,
    manifest_for,
)

from .matrix import ProgramRequest


def build_program_case(request: ProgramRequest) -> FullModelCase:
    """Construct one registered model with the request's packed geometry."""

    if request.model.preset != "throughput":
        raise ValueError(f"unsupported model preset {request.model.preset!r}")
    manifest = manifest_for(
        request.model.family,
        request.model.implementation,
    )
    manifest = replace(
        manifest,
        sequence_length=request.sequence_length,
        sequences_per_microbatch=request.sequences_per_microbatch,
        accumulation_count=request.accumulation_rounds,
        device_physical_capacity_bytes=(request.runtime.execution_pool_capacity_bytes),
        spill_budget_bytes=request.runtime.spill_budget_bytes,
        head_scratch_bytes=(
            manifest.head_scratch_bytes
            if request.model.head_scratch_bytes is None
            else request.model.head_scratch_bytes
        ),
    )
    return build_case(manifest, seed=request.seed)


def profiling_metadata(case: FullModelCase) -> tuple[object, ...]:
    """Preserve authentic packed sequence geometry for every microbatch."""

    result: list[object] = []
    for index, microbatch in enumerate(case.microbatches):
        lengths = microbatch[2]
        if not isinstance(lengths, Sequence):
            raise TypeError(f"microbatch {index} sequence lengths are not a sequence")
        result.append({"sequence_lengths": [int(value) for value in lengths]})
    return tuple(result)


__all__ = ["build_program_case", "profiling_metadata"]
