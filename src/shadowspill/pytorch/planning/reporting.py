"""PlanReport construction and persistent lineage publication."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import replace

import torch.nn as nn

from shadowspill.ir import ExecutionPlan, MemoryActionKind
from shadowspill.planner import PressureFitResult
from shadowspill.pytorch.profiling import ProfilingMetadata, ProfilingResult

from ..cache import PlanningCache
from ..diagnostics import (
    PlanCacheArtifact,
    PlanCompilerProfile,
    PlanDiagnostics,
    PlanFixedLayoutAttempt,
    PlanPhaseTiming,
    PlanPhysicalLayout,
    PlanProfilingMetadata,
    PlanReport,
    PlanTaskMemoryEnvelope,
    PlanTaskStage,
    PlanUniqueStage,
)
from ..runtime_adapter import PlanMemory
from .admission import FixedLayoutSelection, SelectedAdmission
from .common import fixed_execution_bytes


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
    aot_unique_stage_contracts: int = 0,
    aot_graph_pair_cache_hits: int = 0,
    aot_graph_pair_cache_misses: int = 0,
    task_stage_map: tuple[PlanTaskStage, ...] = (),
    unique_stages: tuple[PlanUniqueStage, ...] = (),
    compiler_phase_timings_ns: tuple[tuple[str, int], ...] = (),
    compiler_phase_timings_by_contract: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ] = (),
    cache_directories: tuple[tuple[str, str], ...] = (),
    touched_cache_artifacts: tuple[PlanCacheArtifact, ...] = (),
    profiling_metadata: tuple[ProfilingMetadata, ...] = (),
    physical_layouts: tuple[PlanPhysicalLayout, ...] = (),
    memory: PlanMemory,
) -> PlanReport:
    """Build complete forward planning evidence without writing it."""

    elapsed = time.perf_counter_ns() - started
    phases = _report_phases(timings, elapsed)
    diagnostics = _forward_diagnostics(
        phases,
        elapsed,
        profiles,
        recomputation_cache_hit=recomputation_cache_hit,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_contracts=aot_unique_stage_contracts,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_phase_timings_by_contract=compiler_phase_timings_by_contract,
        cache_directories=cache_directories,
        touched_cache_artifacts=touched_cache_artifacts,
        profiling_metadata=profiling_metadata,
        pressurefit_results=pressurefit_results,
        physical_layouts=physical_layouts,
    )
    return _forward_report(
        _forward_capture_identity(
            signature_digest,
            execution_plan,
            profiling_metadata,
        ),
        execution_plan,
        profiles,
        timings,
        elapsed,
        diagnostics,
        recomputation_cache_hit=recomputation_cache_hit,
        pressurefit_results=pressurefit_results,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_contracts=aot_unique_stage_contracts,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        memory=memory,
    )


def _forward_diagnostics(
    phases: tuple[PlanPhaseTiming, ...],
    elapsed: int,
    profiles: ProfilingResult,
    *,
    recomputation_cache_hit: bool,
    captured_stage_count: int,
    aot_unique_stage_contracts: int,
    aot_graph_pair_cache_hits: int,
    aot_graph_pair_cache_misses: int,
    task_stage_map: tuple[PlanTaskStage, ...],
    unique_stages: tuple[PlanUniqueStage, ...],
    compiler_phase_timings_ns: tuple[tuple[str, int], ...],
    compiler_phase_timings_by_contract: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ],
    cache_directories: tuple[tuple[str, str], ...],
    touched_cache_artifacts: tuple[PlanCacheArtifact, ...],
    profiling_metadata: tuple[ProfilingMetadata, ...],
    pressurefit_results: tuple[PressureFitResult, ...],
    physical_layouts: tuple[PlanPhysicalLayout, ...],
) -> PlanDiagnostics:
    return PlanDiagnostics(
        phases=phases,
        total_wall_time_ns=elapsed,
        unattributed_overhead_ns=elapsed - sum(item.duration_ns for item in phases),
        profile_unique_keys=profiles.unique_keys,
        profile_cache_hits=profiles.cache_hits,
        profile_cache_misses=profiles.cache_misses,
        allocation_probe_seeds=profiles.allocation_probe_seeds,
        allocation_probe_repetitions=profiles.allocation_probe_repetitions,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_contracts=aot_unique_stage_contracts,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        recomputation_cache_hits=int(recomputation_cache_hit),
        recomputation_cache_misses=int(not recomputation_cache_hit),
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_profiles=tuple(
            PlanCompilerProfile(
                structural_contract_key,
                tuple(PlanPhaseTiming(name, duration) for name, duration in values),
            )
            for structural_contract_key, values in compiler_phase_timings_by_contract
        ),
        cache_directories=cache_directories,
        cache_artifacts=touched_cache_artifacts,
        profiling_metadata=tuple(
            PlanProfilingMetadata(index, item.digest, item.canonical_json)
            for index, item in enumerate(profiling_metadata)
        ),
        pressurefit_runs=tuple(item.diagnostics for item in pressurefit_results),
        physical_layouts=physical_layouts,
    )


def _forward_report(
    capture_identity: str,
    execution_plan: ExecutionPlan,
    profiles: ProfilingResult,
    timings: tuple[tuple[str, int], ...],
    elapsed: int,
    diagnostics: PlanDiagnostics,
    *,
    recomputation_cache_hit: bool,
    pressurefit_results: tuple[PressureFitResult, ...],
    captured_stage_count: int,
    aot_unique_stage_contracts: int,
    aot_graph_pair_cache_hits: int,
    aot_graph_pair_cache_misses: int,
    memory: PlanMemory,
) -> PlanReport:
    actions = execution_plan.schedule.actions
    return PlanReport(
        mode="forward",
        capture_identity=capture_identity,
        execution_plan=execution_plan,
        task_profiles=execution_plan.program.profiles,
        transfer_actions=actions,
        transfer_bytes_evicted=_transfer_bytes(
            execution_plan, MemoryActionKind.OFFLOAD
        ),
        transfer_bytes_fetched=_transfer_bytes(
            execution_plan, MemoryActionKind.PREFETCH
        ),
        profile_unique_keys=profiles.unique_keys,
        profile_cache_hits=profiles.cache_hits,
        profile_cache_misses=profiles.cache_misses,
        allocation_probe_seeds=profiles.allocation_probe_seeds,
        allocation_probe_repetitions=profiles.allocation_probe_repetitions,
        profiling_provenance=tuple(
            dict.fromkeys(item.provenance for item in profiles.measurements)
        ),
        phase_timings_ns=(*timings, ("total", elapsed)),
        recomputation_cache_hits=int(recomputation_cache_hit),
        recomputation_cache_misses=int(not recomputation_cache_hit),
        fixed_slab_bytes=fixed_execution_bytes(memory, profiles),
        pressurefit_results=pressurefit_results,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_contracts=aot_unique_stage_contracts,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        diagnostics=diagnostics,
        execution_pool=memory.execution.name,
        spill_pool=memory.spill.name,
        execution_budget_bytes=memory.execution_budget,
        spill_budget_bytes=memory.spill_budget,
        requested_dynamic_scratch_reserve_bytes=(memory.dynamic_scratch_reserve_bytes),
        execution_device=memory.execution_device,
        transfer_capabilities=memory.transfers,
    )


def _forward_capture_identity(
    signature_digest: str,
    execution_plan: ExecutionPlan,
    profiling_metadata: tuple[ProfilingMetadata, ...],
) -> str:
    identity = {
        "mode": "forward",
        "signature": signature_digest,
        "artifacts": [item.contract_digest for item in execution_plan.entrypoints],
        "profiling_metadata": [item.digest for item in profiling_metadata],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _report_phases(
    timings: tuple[tuple[str, int], ...],
    elapsed: int,
) -> tuple[PlanPhaseTiming, ...]:
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
    if sum(item.duration_ns for item in phases) > elapsed:
        raise RuntimeError(
            "plan phase intervals overlap: measured phase time exceeds wall time"
        )
    return phases


def _transfer_bytes(
    execution_plan: ExecutionPlan,
    kind: MemoryActionKind,
) -> int:
    sizes = {
        item.alias_group_id: item.size_bytes
        for item in execution_plan.program.alias_groups
    }
    return sum(
        sizes[action.alias_group_id]
        for action in execution_plan.schedule.actions
        if action.kind is kind
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
    aot_unique_stage_contracts: int,
    aot_graph_pair_cache_hits: int,
    aot_graph_pair_cache_misses: int,
    pressurefit_results: tuple[PressureFitResult, ...],
    task_stage_map: tuple[PlanTaskStage, ...],
    unique_stages: tuple[PlanUniqueStage, ...],
    compiler_phase_timings_ns: tuple[tuple[str, int], ...],
    compiler_phase_timings_by_contract: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ],
    cache_directories: tuple[tuple[str, str], ...],
    touched_cache_artifacts: tuple[PlanCacheArtifact, ...],
    profiling_metadata: tuple[ProfilingMetadata, ...],
    physical_layouts: tuple[PlanPhysicalLayout, ...],
    optimizer_ordering: str,
    memory: PlanMemory,
) -> PlanReport:
    """Build complete accumulated-training planning evidence without writing it."""

    identity = {
        "mode": "training",
        "signatures": signature_digests,
        "artifacts": [item.contract_digest for item in execution_plan.entrypoints],
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
        aot_unique_stage_contracts=aot_unique_stage_contracts,
        aot_graph_pair_cache_hits=aot_graph_pair_cache_hits,
        aot_graph_pair_cache_misses=aot_graph_pair_cache_misses,
        compiler_phase_timings_ns=compiler_phase_timings_ns,
        compiler_phase_timings_by_contract=compiler_phase_timings_by_contract,
        cache_directories=cache_directories,
        touched_cache_artifacts=touched_cache_artifacts,
        profiling_metadata=profiling_metadata,
        pressurefit_results=pressurefit_results,
        physical_layouts=physical_layouts,
        memory=memory,
    )
    base = report.diagnostics
    diagnostics = replace(
        base,
        captured_stage_count=captured_stage_count,
        aot_unique_stage_contracts=aot_unique_stage_contracts,
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
        aot_unique_stage_contracts=aot_unique_stage_contracts,
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
            "requested_dynamic_scratch_reserve_bytes": (
                report.requested_dynamic_scratch_reserve_bytes
            ),
            "allocation_probe_seeds": report.allocation_probe_seeds,
            "allocation_probe_repetitions": report.allocation_probe_repetitions,
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


def fixed_layout_diagnostic(
    plan_role: str,
    selection: FixedLayoutSelection,
    admitted: SelectedAdmission,
) -> PlanPhysicalLayout:
    """Build public immutable evidence for one certified physical layout."""

    layout = admitted.fixed_layout
    if layout is None:
        raise ValueError("fixed-layout diagnostics require a fixed admission")
    purposes = Counter(item.purpose.value for item in layout.placements)
    envelopes = tuple(
        PlanTaskMemoryEnvelope(
            task_id,
            envelope.maximum_requested_allocation_bytes,
            envelope.maximum_charged_allocation_bytes,
            envelope.live_requested_allocation_limit_bytes,
            envelope.live_charged_allocation_limit_bytes,
            envelope.dynamic_scratch_maximum_allocation_bytes,
            envelope.dynamic_scratch_live_limit_bytes,
            (
                None
                if envelope.allocation_contract is None
                else envelope.allocation_contract.compatibility_digest
            ),
            (
                0
                if envelope.allocation_contract is None
                else len(envelope.allocation_contract.steps)
            ),
            envelope.allocation_path_digests,
        )
        for task_id, envelope in admitted.task_envelopes
    )
    return PlanPhysicalLayout(
        plan_role=plan_role,
        strategy="fixed",
        layout_digest=layout.digest,
        program_digest=layout.program_digest,
        schedule_digest=layout.schedule_digest,
        topology_digest=layout.topology_digest,
        pool_capacity_bytes=layout.pool_capacity_bytes,
        original_object_capacity_bytes=selection.original_object_capacity_bytes,
        effective_object_capacity_bytes=selection.topology.object_capacity_bytes,
        object_capacity_reduction_bytes=selection.capacity_reduction_bytes,
        fixed_slice_bytes=layout.fixed_slice_bytes,
        dynamic_reserve_bytes=layout.dynamic_reserve_bytes,
        scratch_reserve_bytes=layout.scratch_reserve_bytes,
        required_bytes=layout.required_bytes,
        slack_bytes=layout.slack_bytes,
        placement_count=len(layout.placements),
        dynamic_lifetime_count=len(layout.dynamic_lifetimes),
        reuse_dependency_count=len(layout.reuse_dependencies),
        placements_by_purpose=tuple(sorted(purposes.items())),
        attempts=tuple(
            PlanFixedLayoutAttempt(
                item.requested_object_capacity_bytes,
                item.effective_object_capacity_bytes,
                item.required_bytes,
                item.pool_capacity_bytes,
                item.accepted,
                item.pressurefit_wall_time_ns,
                item.physical_admission_wall_time_ns,
                item.pressurefit_diagnostics,
            )
            for item in selection.attempts
        ),
        task_memory_envelopes=envelopes,
    )


__all__ = [
    "build_forward_report",
    "build_training_report",
    "cache_artifacts",
    "fixed_layout_diagnostic",
    "publish_plan_report",
]
