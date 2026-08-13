"""PlanReport construction and persistent lineage publication."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace

import torch.nn as nn

from shadowspill.ir import ExecutionPlan, MemoryActionKind
from shadowspill.planner import PressureFitResult

from .._profiling_metadata import ProfilingMetadata
from ..cache import PlanningCache
from ..profiling import ProfilingResult
from ..public import (
    PlanCacheArtifact,
    PlanCompilerProfile,
    PlanDiagnostics,
    PlanPhaseTiming,
    PlanProfilingMetadata,
    PlanReport,
    PlanTaskStage,
    PlanUniqueStage,
)
from ..runtime import PlanMemory


def cache_artifacts(cache: PlanningCache) -> tuple[PlanCacheArtifact, ...]:
    """Return immutable public evidence for every touched cache artifact."""

    return tuple(
        PlanCacheArtifact(
            item.category,
            item.kind,
            item.digest,
            str(item.path),
            item.access,
            item.schema,
            item.dependencies,
        )
        for item in cache.artifacts()
    )


def build_forward_report(
    signature_digest: str,
    execution_plan: ExecutionPlan,
    profiles: ProfilingResult,
    timings: tuple[tuple[str, int], ...],
    started: int,
    *,
    recomputation_cache_hit: bool = False,
    pressurefit_results: tuple[PressureFitResult, ...] = (),
    captured_stage_count: int = 0,
    aot_unique_stage_abis: int = 0,
    aot_graph_pair_cache_hits: int = 0,
    aot_graph_pair_cache_misses: int = 0,
    task_stage_map: tuple[PlanTaskStage, ...] = (),
    unique_stages: tuple[PlanUniqueStage, ...] = (),
    compiler_phase_timings_ns: tuple[tuple[str, int], ...] = (),
    compiler_phase_timings_by_abi: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ] = (),
    cache_directories: tuple[tuple[str, str], ...] = (),
    touched_cache_artifacts: tuple[PlanCacheArtifact, ...] = (),
    profiling_metadata: tuple[ProfilingMetadata, ...] = (),
    memory: PlanMemory,
) -> PlanReport:
    """Build complete forward planning evidence without writing it."""

    identity = {
        "mode": "forward",
        "signature": signature_digest,
        "artifacts": [
            entrypoint.abi_digest for entrypoint in execution_plan.entrypoints
        ],
        "profiling_metadata": [item.digest for item in profiling_metadata],
    }
    capture_identity = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sizes = {
        item.alias_group_id: item.size_bytes
        for item in execution_plan.program.alias_groups
    }
    actions = execution_plan.schedule.actions
    elapsed = time.perf_counter_ns() - started
    nested_capture = any(
        name
        in {
            "objective_export",
            "export_archival",
            "stage_partition_aot",
            "storage_layout_lowering",
        }
        for name, _ in timings
    )
    phases = tuple(
        PlanPhaseTiming(name, duration)
        for name, duration in timings
        if not (nested_capture and name == "capture_lowering")
    )
    measured = sum(item.duration_ns for item in phases)
    if measured > elapsed:
        raise RuntimeError(
            "plan phase intervals overlap: measured phase time exceeds wall time"
        )
    diagnostics = PlanDiagnostics(
        phases=phases,
        total_wall_time_ns=elapsed,
        unattributed_overhead_ns=elapsed - measured,
        profile_unique_keys=profiles.unique_keys,
        profile_cache_hits=profiles.cache_hits,
        profile_cache_misses=profiles.cache_misses,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        recomputation_cache_hits=int(recomputation_cache_hit),
        recomputation_cache_misses=int(not recomputation_cache_hit),
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_profiles=tuple(
            PlanCompilerProfile(
                structural_abi_key,
                tuple(PlanPhaseTiming(name, duration) for name, duration in values),
            )
            for structural_abi_key, values in compiler_phase_timings_by_abi
        ),
        cache_directories=cache_directories,
        cache_artifacts=touched_cache_artifacts,
        profiling_metadata=tuple(
            PlanProfilingMetadata(index, item.digest, item.canonical_json)
            for index, item in enumerate(profiling_metadata)
        ),
    )
    return PlanReport(
        mode="forward",
        capture_identity=capture_identity,
        execution_plan=execution_plan,
        task_profiles=execution_plan.program.profiles,
        transfer_actions=actions,
        transfer_bytes_evicted=sum(
            sizes[item.alias_group_id]
            for item in actions
            if item.kind is MemoryActionKind.OFFLOAD
        ),
        transfer_bytes_fetched=sum(
            sizes[item.alias_group_id]
            for item in actions
            if item.kind is MemoryActionKind.PREFETCH
        ),
        profile_unique_keys=profiles.unique_keys,
        profile_cache_hits=profiles.cache_hits,
        profile_cache_misses=profiles.cache_misses,
        profiling_provenance=tuple(
            dict.fromkeys(item.provenance for item in profiles.measurements)
        ),
        phase_timings_ns=(*timings, ("total", elapsed)),
        recomputation_cache_hits=int(recomputation_cache_hit),
        recomputation_cache_misses=int(not recomputation_cache_hit),
        fixed_slab_bytes=profiles.fixed_slab_bytes,
        pressurefit_results=pressurefit_results,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        diagnostics=diagnostics,
        execution_pool=memory.execution.name,
        spill_pool=memory.spill.name,
        execution_budget_bytes=memory.execution_budget,
        spill_budget_bytes=memory.spill_budget,
        execution_device=memory.execution_device,
        transfer_capabilities=memory.transfers,
    )


def build_training_report(
    signature_digests: tuple[str, ...],
    execution_plan: ExecutionPlan,
    profiles: ProfilingResult,
    timings: tuple[tuple[str, int], ...],
    started: int,
    *,
    initial_execution_plan: ExecutionPlan | None,
    recomputation_cache_hits: int,
    recomputation_cache_misses: int,
    captured_stage_count: int,
    aot_unique_stage_abis: int,
    aot_graph_pair_cache_hits: int,
    aot_graph_pair_cache_misses: int,
    pressurefit_results: tuple[PressureFitResult, ...],
    task_stage_map: tuple[PlanTaskStage, ...],
    unique_stages: tuple[PlanUniqueStage, ...],
    compiler_phase_timings_ns: tuple[tuple[str, int], ...],
    compiler_phase_timings_by_abi: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
    cache_directories: tuple[tuple[str, str], ...],
    touched_cache_artifacts: tuple[PlanCacheArtifact, ...],
    profiling_metadata: tuple[ProfilingMetadata, ...],
    optimizer_ordering: str,
    memory: PlanMemory,
) -> PlanReport:
    """Build complete accumulated-training planning evidence without writing it."""

    identity = {
        "mode": "training",
        "signatures": signature_digests,
        "artifacts": [item.abi_digest for item in execution_plan.entrypoints],
        "optimizer_ordering": optimizer_ordering,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = build_forward_report(
        digest,
        execution_plan,
        profiles,
        timings,
        started,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_phase_timings_by_abi=compiler_phase_timings_by_abi,
        cache_directories=cache_directories,
        touched_cache_artifacts=touched_cache_artifacts,
        profiling_metadata=profiling_metadata,
        memory=memory,
    )
    base = report.diagnostics
    diagnostics = replace(
        base,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        recomputation_cache_hits=recomputation_cache_hits,
        recomputation_cache_misses=recomputation_cache_misses,
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        cache_directories=cache_directories,
    )
    return replace(
        report,
        mode="training",
        capture_identity=digest,
        initial_execution_plan=initial_execution_plan,
        recomputation_cache_hits=recomputation_cache_hits,
        recomputation_cache_misses=recomputation_cache_misses,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_abis=aot_unique_stage_abis,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        pressurefit_results=pressurefit_results,
        diagnostics=diagnostics,
        optimizer_ordering=optimizer_ordering,
    )


def publish_plan_report(
    model: nn.Module,
    report: PlanReport,
    cache: PlanningCache,
    *,
    started: int,
) -> PlanReport:
    """Persist plan lineage and return the final immutable report."""

    archival_started = time.perf_counter_ns()
    artifacts_before_plan = cache_artifacts(cache)
    cache.archive_plan(
        model_label=f"{type(model).__module__}.{type(model).__qualname__}",
        capture_identity=report.capture_identity,
        execution_plan=report.execution_plan,
        initial_execution_plan=report.initial_execution_plan,
        manifest={
            "mode": report.mode,
            "execution_pool": report.execution_pool,
            "spill_pool": report.spill_pool,
            "execution_budget_bytes": report.execution_budget_bytes,
            "spill_budget_bytes": report.spill_budget_bytes,
            "execution_device": report.execution_device,
            "implementation_revision": cache.implementation_revision,
            "phase_timings_ns": [list(item) for item in report.phase_timings_ns],
            "artifacts": [item.as_dict() for item in artifacts_before_plan],
        },
    )
    archival_duration = time.perf_counter_ns() - archival_started
    elapsed = time.perf_counter_ns() - started
    phases = (
        *report.diagnostics.phases,
        PlanPhaseTiming("plan_archival", archival_duration),
    )
    measured = sum(item.duration_ns for item in phases)
    if measured > elapsed:
        raise RuntimeError(
            "plan phase intervals overlap after archival: measured time exceeds wall"
        )
    diagnostics = replace(
        report.diagnostics,
        phases=phases,
        total_wall_time_ns=elapsed,
        unattributed_overhead_ns=elapsed - measured,
        cache_artifacts=cache_artifacts(cache),
    )
    phase_timings = tuple(
        item for item in report.phase_timings_ns if item[0] != "total"
    )
    return replace(
        report,
        diagnostics=diagnostics,
        phase_timings_ns=(
            *phase_timings,
            ("plan_archival", archival_duration),
            ("total", elapsed),
        ),
    )


__all__ = [
    "build_forward_report",
    "build_training_report",
    "cache_artifacts",
    "publish_plan_report",
]
