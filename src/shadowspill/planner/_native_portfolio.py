"""One-call dense C evaluation for a resolved recomputation context."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from functools import lru_cache

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

from ._admission import CompiledAdmissionTopology
from ._capi import (
    ABI_VERSION,
    NO_INDEX,
    CPressureFitContext,
    CPressureFitContextOptions,
    CPressureFitContextResult,
    CPressureFitProgramContext,
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
    "demand": 4,
}
_ACTION_KIND = {
    0: MemoryActionKind.RELEASE,
    1: MemoryActionKind.OFFLOAD,
    2: MemoryActionKind.PREFETCH,
}
_LOCATION = {0: MemoryLocation.DEVICE, 1: MemoryLocation.HOST}
_INITIAL_PLACEMENT = {"required": 0, "greedy": 1}


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
class NativeDenseSchedule:
    """Copied dense indices for one context winner.

    Keeping the context-local winner dense avoids constructing and validating
    thousands of Python IR records that will be discarded when a different
    recomputation context wins the global portfolio.
    """

    action_trigger_tasks: tuple[int, ...]
    action_aliases: tuple[int, ...]
    action_kinds: tuple[int, ...]
    initial_aliases: tuple[int, ...]
    initial_locations: tuple[int, ...]
    final_aliases: tuple[int, ...]
    final_locations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeContextResult:
    selected_candidate_index: int | None
    selected_makespan_ns: int | None
    selected_schedule: NativeDenseSchedule | None
    candidates: tuple[NativeCandidateDiagnostic, ...]
    residency_cache_hits: int
    residency_cache_misses: int
    schedule_emissions: int
    schedule_cache_hits: int
    simulation_calls: int
    simulation_cache_hits: int
    admission_calls: int
    residency_time_ns: int
    schedule_time_ns: int
    simulation_time_ns: int
    admission_time_ns: int
    digest_time_ns: int


class NativeContextPreparationError(RuntimeError):
    """The compiled topology could not be normalized into planner facts."""


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
    if value.status == 3:
        device_id = simulation.device_ids[value.error_device]
        detail = (
            "dynamic MemoryPool admission cannot place a compatible range: "
            f"device={device_id!r}, capacity={value.error_capacity_bytes}, "
            f"used={value.error_used_bytes}, "
            f"request={value.error_requested_bytes}, "
            f"additional_slack={value.error_required_bytes}"
        )
        return CandidateDiagnostic(
            candidate_id=value.candidate_id,
            selection_id=selection_id,
            status="infeasible",
            failure_kind="physical_admission",
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


@lru_cache(maxsize=131_072)
def _escaped_identifier(value: str) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return encoded[1:-1].encode("utf-8")


def _copy_schedule(result: CPressureFitContextResult) -> NativeDenseSchedule:
    value = result.selected_schedule
    return NativeDenseSchedule(
        action_trigger_tasks=tuple(
            int(value.action_trigger_tasks[index])
            for index in range(int(value.action_count))
        ),
        action_aliases=tuple(
            int(value.action_aliases[index])
            for index in range(int(value.action_count))
        ),
        action_kinds=tuple(
            int(value.action_kinds[index])
            for index in range(int(value.action_count))
        ),
        initial_aliases=tuple(
            int(value.initial_aliases[index])
            for index in range(int(value.initial_count))
        ),
        initial_locations=tuple(
            int(value.initial_locations[index])
            for index in range(int(value.initial_count))
        ),
        final_aliases=tuple(
            int(value.final_aliases[index])
            for index in range(int(value.final_count))
        ),
        final_locations=tuple(
            int(value.final_locations[index])
            for index in range(int(value.final_count))
        ),
    )


def decode_schedule(
    value: NativeDenseSchedule,
    simulation: CompiledSimulationTemplate,
) -> MemorySchedule:
    return MemorySchedule(
        initial_residency=tuple(
            ResidencySpec(
                simulation.alias_ids[alias],
                _LOCATION[location],
            )
            for alias, location in zip(
                value.initial_aliases,
                value.initial_locations,
                strict=True,
            )
        ),
        actions=tuple(
            MemoryAction(
                simulation.task_ids[task],
                simulation.alias_ids[alias],
                _ACTION_KIND[kind],
            )
            for task, alias, kind in zip(
                value.action_trigger_tasks,
                value.action_aliases,
                value.action_kinds,
                strict=True,
            )
        ),
        final_residency=tuple(
            ResidencySpec(
                simulation.alias_ids[alias],
                _LOCATION[location],
            )
            for alias, location in zip(
                value.final_aliases,
                value.final_locations,
                strict=True,
            )
        ),
    )


def _evaluate_native_context(
    simulation: CompiledSimulationTemplate,
    options: PressureFitOptions,
    *,
    residency: CompiledResidencyTemplate | None,
    admission: CompiledAdmissionTopology | None,
) -> NativeContextResult | None:
    """Invoke and decode either compiled context-entry path."""

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
    native_options = CPressureFitContextOptions(
        residency_strategies=strategies,
        residency_strategy_count=len(strategy_names),
        prefetch_rules=rules,
        prefetch_rule_count=len(rule_names),
        evaluate_coalesced=int(options.evaluate_coalesced),
        max_repair_attempts=options.max_repair_attempts,
        initial_placement=_INITIAL_PLACEMENT[options.initial_placement.value],
    )
    native_result = CPressureFitContextResult()
    library = load_planner_library()
    if residency is not None:
        context: CPressureFitContext | CPressureFitProgramContext = (
            CPressureFitContext(
                abi_version=ABI_VERSION,
                residency=ctypes.pointer(residency.problem),
                simulation=ctypes.pointer(simulation.program),
                seed_resident=residency.seed_resident,
                seed_breaks=residency.seed_breaks,
                admission=(
                    ctypes.pointer(admission.value)
                    if admission is not None
                    else None
                ),
                alias_json_names=alias_names,
                task_json_names=task_names,
            )
        )
        status = int(
            library.shadowspill_evaluate_pressurefit_context(
                ctypes.byref(context),
                ctypes.byref(native_options),
                ctypes.byref(native_result),
            )
        )
    else:
        device_ranks = {
            device_id: rank
            for rank, device_id in enumerate(sorted(simulation.device_ids))
        }
        priorities = (ctypes.c_uint32 * len(simulation.device_ids))(
            *(device_ranks[value] for value in simulation.device_ids)
        )
        context = CPressureFitProgramContext(
            abi_version=ABI_VERSION,
            simulation=ctypes.pointer(simulation.program),
            device_priority=priorities,
            admission=(
                ctypes.pointer(admission.value) if admission is not None else None
            ),
            alias_json_names=alias_names,
            task_json_names=task_names,
        )
        status = int(
            library.shadowspill_evaluate_pressurefit_program_context(
                ctypes.byref(context),
                ctypes.byref(native_options),
                ctypes.byref(native_result),
            )
        )
    try:
        if residency is None and status == 5:
            return None
        if residency is None and status == 1:
            raise NativeContextPreparationError(
                "compiled PressureFit context rejected the selected topology"
            )
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
                else _copy_schedule(native_result)
            ),
            candidates=tuple(candidates),
            residency_cache_hits=int(native_result.residency_cache_hits),
            residency_cache_misses=int(native_result.residency_cache_misses),
            schedule_emissions=int(native_result.schedule_emissions),
            schedule_cache_hits=int(native_result.schedule_cache_hits),
            simulation_calls=int(native_result.simulation_calls),
            simulation_cache_hits=int(native_result.simulation_cache_hits),
            admission_calls=int(native_result.admission_calls),
            residency_time_ns=int(native_result.residency_time_ns),
            schedule_time_ns=int(native_result.schedule_time_ns),
            simulation_time_ns=int(native_result.simulation_time_ns),
            admission_time_ns=int(native_result.admission_time_ns),
            digest_time_ns=int(native_result.digest_time_ns),
        )
    finally:
        library.shadowspill_pressurefit_context_result_destroy(
            ctypes.byref(native_result)
        )


def evaluate_context_compiled(
    residency: CompiledResidencyTemplate,
    simulation: CompiledSimulationTemplate,
    options: PressureFitOptions,
) -> NativeContextResult:
    """Evaluate a caller-prepared dense context in C."""

    result = _evaluate_native_context(
        simulation, options, residency=residency, admission=None
    )
    assert result is not None
    return result


def evaluate_program_context_compiled(
    simulation: CompiledSimulationTemplate,
    options: PressureFitOptions,
    *,
    admission: CompiledAdmissionTopology | None = None,
) -> NativeContextResult | None:
    """Derive dense planning facts and evaluate the portfolio entirely in C."""

    return _evaluate_native_context(
        simulation, options, residency=None, admission=admission
    )


__all__ = [
    "NativeCandidateDiagnostic",
    "NativeContextPreparationError",
    "NativeContextResult",
    "NativeDenseSchedule",
    "decode_candidate_diagnostic",
    "decode_schedule",
    "evaluate_context_compiled",
    "evaluate_program_context_compiled",
]
