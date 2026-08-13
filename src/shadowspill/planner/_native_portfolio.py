"""One-call dense C evaluation for a resolved recomputation context."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ResidencySpec,
)
from shadowspill.simulator._compiled import CompiledSimulationTemplate
from shadowspill.simulator._diagnostics import (
    simulation_failure_detail,
    simulation_status_kind,
)

from ._capi import (
    ABI_VERSION,
    NO_INDEX,
    CPressureFitContext,
    CPressureFitContextOptions,
    CPressureFitContextResult,
    load_planner_library,
)
from ._dense_residency import CompiledResidencyTemplate
from .model import CandidateDiagnostic, PressureFitOptions

_STRATEGY_CODE = {
    "headroom-stall": 0,
    "headroom-transfer": 1,
    "tight-stall": 2,
    "tight-transfer": 3,
    "relaxed-stall": 4,
}
_RULE_CODE = {
    "packed-fifo": 0,
    "packed-fit": 1,
    "interval-entry": 2,
    "latest-safe": 3,
}
_ACTION_KIND = {
    0: MemoryActionKind.RELEASE,
    1: MemoryActionKind.OFFLOAD,
    2: MemoryActionKind.PREFETCH,
}
_LOCATION = {0: MemoryLocation.DEVICE, 1: MemoryLocation.HOST}


@dataclass(frozen=True, slots=True)
class NativeCandidateDiagnostic:
    status: int
    strategy: str
    rule: str
    coalesced: bool
    repair_attempts: int
    simulation_status: int
    makespan_ns: int
    schedule_digest: str | None
    error_task: int
    error_alias: int
    error_device: int
    error_location: int
    error_boundary: int
    error_time_ns: int
    error_capacity_bytes: int
    error_used_bytes: int
    error_requested_bytes: int
    error_required_bytes: int

    @property
    def candidate_id(self) -> str:
        suffix = "-coalesced" if self.coalesced else ""
        return f"{self.strategy}/{self.rule}{suffix}"


@dataclass(frozen=True, slots=True)
class NativeContextResult:
    selected_candidate_index: int | None
    selected_makespan_ns: int | None
    selected_schedule: MemorySchedule | None
    candidates: tuple[NativeCandidateDiagnostic, ...]
    residency_cache_hits: int
    residency_cache_misses: int
    schedule_emissions: int
    schedule_cache_hits: int
    simulation_calls: int
    simulation_cache_hits: int
    residency_time_ns: int
    schedule_time_ns: int
    simulation_time_ns: int
    digest_time_ns: int


def decode_candidate_diagnostic(
    value: NativeCandidateDiagnostic,
    *,
    selection_id: str,
    simulation: CompiledSimulationTemplate,
) -> CandidateDiagnostic:
    """Convert one dense diagnostic without changing its semantic fields."""

    if value.status == 0:
        return CandidateDiagnostic(
            candidate_id=value.candidate_id,
            selection_id=selection_id,
            status="valid",
            makespan_ns=value.makespan_ns,
            schedule_digest=value.schedule_digest,
            repair_attempts=value.repair_attempts,
        )
    if value.status == 1:
        device_id = simulation.device_ids[value.error_device]
        detail = (
            "no legal residency cut can relieve "
            f"{value.error_required_bytes} bytes at boundary "
            f"{value.error_boundary} on '{device_id}'; capacity is "
            f"{value.error_capacity_bytes}"
        )
        return CandidateDiagnostic(
            candidate_id=value.candidate_id,
            selection_id=selection_id,
            status="infeasible",
            failure_kind="analytic_capacity",
            failure_detail=detail,
            repair_attempts=value.repair_attempts,
        )
    if value.status != 2:
        raise RuntimeError(
            f"compiled PressureFit candidate {value.candidate_id!r} "
            f"returned internal status {value.status}"
        )
    kind = simulation_status_kind(value.simulation_status)
    detail = simulation_failure_detail(
        value.simulation_status,
        time_ns=value.error_time_ns,
        error_device=value.error_device,
        error_location=value.error_location,
        capacity_bytes=value.error_capacity_bytes,
        used_bytes=value.error_used_bytes,
        requested_bytes=value.error_requested_bytes,
        device_ids=simulation.device_ids,
    )
    return CandidateDiagnostic(
        candidate_id=value.candidate_id,
        selection_id=selection_id,
        status="infeasible",
        failure_kind=kind,
        failure_detail=detail,
        repair_attempts=value.repair_attempts,
    )


def _escaped_identifier(value: str) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return encoded[1:-1].encode("utf-8")


def _decode_schedule(
    result: CPressureFitContextResult,
    simulation: CompiledSimulationTemplate,
) -> MemorySchedule:
    value = result.selected_schedule
    return MemorySchedule(
        initial_residency=tuple(
            ResidencySpec(
                simulation.alias_ids[int(value.initial_aliases[index])],
                _LOCATION[int(value.initial_locations[index])],
            )
            for index in range(int(value.initial_count))
        ),
        actions=tuple(
            MemoryAction(
                simulation.task_ids[int(value.action_trigger_tasks[index])],
                simulation.alias_ids[int(value.action_aliases[index])],
                _ACTION_KIND[int(value.action_kinds[index])],
            )
            for index in range(int(value.action_count))
        ),
        final_residency=tuple(
            ResidencySpec(
                simulation.alias_ids[int(value.final_aliases[index])],
                _LOCATION[int(value.final_locations[index])],
            )
            for index in range(int(value.final_count))
        ),
    )


def evaluate_context_compiled(
    residency: CompiledResidencyTemplate,
    simulation: CompiledSimulationTemplate,
    options: PressureFitOptions,
) -> NativeContextResult:
    """Evaluate all policy variants while retaining dense records in C."""

    alias_payloads = tuple(_escaped_identifier(value) for value in simulation.alias_ids)
    task_payloads = tuple(_escaped_identifier(value) for value in simulation.task_ids)
    alias_names = (ctypes.c_char_p * len(alias_payloads))(*alias_payloads)
    task_names = (ctypes.c_char_p * len(task_payloads))(*task_payloads)
    strategy_names = tuple(options.residency_strategies)
    rule_names = tuple(options.prefetch_rules)
    strategies = (ctypes.c_uint8 * len(strategy_names))(
        *(_STRATEGY_CODE[value] for value in strategy_names)
    )
    rules = (ctypes.c_uint8 * len(rule_names))(
        *(_RULE_CODE[value] for value in rule_names)
    )
    context = CPressureFitContext(
        abi_version=ABI_VERSION,
        residency=ctypes.pointer(residency.problem),
        simulation=ctypes.pointer(simulation.program),
        seed_resident=residency.seed_resident,
        seed_breaks=residency.seed_breaks,
        alias_json_names=alias_names,
        task_json_names=task_names,
    )
    native_options = CPressureFitContextOptions(
        residency_strategies=strategies,
        residency_strategy_count=len(strategy_names),
        prefetch_rules=rules,
        prefetch_rule_count=len(rule_names),
        evaluate_coalesced=int(options.evaluate_coalesced),
        max_repair_attempts=options.max_repair_attempts,
    )
    native_result = CPressureFitContextResult()
    library = load_planner_library()
    status = int(
        library.shadowspill_evaluate_pressurefit_context(
            ctypes.byref(context),
            ctypes.byref(native_options),
            ctypes.byref(native_result),
        )
    )
    try:
        if status not in (0, 3):
            encoded = library.shadowspill_planner_status_string(status)
            message = encoded.decode("utf-8") if encoded else f"planner status {status}"
            raise RuntimeError(message)
        candidates: list[NativeCandidateDiagnostic] = []
        for index in range(int(native_result.candidate_count)):
            value = native_result.candidates[index]
            variants_per_rule = 2 if options.evaluate_coalesced else 1
            strategy = strategy_names[
                index // (len(rule_names) * variants_per_rule)
            ]
            within_strategy = index % (
                len(rule_names) * (2 if options.evaluate_coalesced else 1)
            )
            divisor = 2 if options.evaluate_coalesced else 1
            rule = rule_names[within_strategy // divisor]
            digest = (
                bytes(value.schedule_digest).hex()
                if int(value.status) == 0
                else None
            )
            candidates.append(
                NativeCandidateDiagnostic(
                    status=int(value.status),
                    strategy=strategy,
                    rule=rule,
                    coalesced=bool(value.coalesced),
                    repair_attempts=int(value.repair_attempts),
                    simulation_status=int(value.simulation_status),
                    makespan_ns=int(value.makespan_ns),
                    schedule_digest=digest,
                    error_task=int(value.error_task),
                    error_alias=int(value.error_alias),
                    error_device=int(value.error_device),
                    error_location=int(value.error_location),
                    error_boundary=int(value.error_boundary),
                    error_time_ns=int(value.error_time_ns),
                    error_capacity_bytes=int(value.error_capacity_bytes),
                    error_used_bytes=int(value.error_used_bytes),
                    error_requested_bytes=int(value.error_requested_bytes),
                    error_required_bytes=int(value.error_required_bytes),
                )
            )
        selected = int(native_result.selected_candidate_index)
        return NativeContextResult(
            selected_candidate_index=None if selected == NO_INDEX else selected,
            selected_makespan_ns=(
                None
                if selected == NO_INDEX
                else int(native_result.selected_makespan_ns)
            ),
            selected_schedule=(
                None
                if selected == NO_INDEX
                else _decode_schedule(native_result, simulation)
            ),
            candidates=tuple(candidates),
            residency_cache_hits=int(native_result.residency_cache_hits),
            residency_cache_misses=int(native_result.residency_cache_misses),
            schedule_emissions=int(native_result.schedule_emissions),
            schedule_cache_hits=int(native_result.schedule_cache_hits),
            simulation_calls=int(native_result.simulation_calls),
            simulation_cache_hits=int(native_result.simulation_cache_hits),
            residency_time_ns=int(native_result.residency_time_ns),
            schedule_time_ns=int(native_result.schedule_time_ns),
            simulation_time_ns=int(native_result.simulation_time_ns),
            digest_time_ns=int(native_result.digest_time_ns),
        )
    finally:
        library.shadowspill_pressurefit_context_result_destroy(
            ctypes.byref(native_result)
        )


__all__ = [
    "NativeCandidateDiagnostic",
    "NativeContextResult",
    "decode_candidate_diagnostic",
    "evaluate_context_compiled",
]
