"""One-call indexed C evaluation for a resolved recomputation context."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from functools import lru_cache

from shadowspill._status import ABI_VERSION, Status
from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ResidencySpec,
)
from shadowspill.simulator._diagnostics import (
    simulation_failure_detail,
    simulation_status_kind,
)
from shadowspill.simulator._indexed import IndexedSimulationTemplate

from ._admission import IndexedAdmissionTopology
from ._capi import (
    NO_INDEX,
    CPressureFitContextOptions,
    CPressureFitContextResult,
    CPressureFitPreflightResult,
    CPressureFitProgramContext,
    CPressureFitRepairDiagnostics,
    CPressureFitWorkDiagnostics,
    load_planner_library,
)
from .diagnostics import PressureFitRepairDiagnostics, PressureFitWorkDiagnostics
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
_LOCATION = {0: MemoryLocation.DEVICE, 1: MemoryLocation.SPILL}
_INITIAL_PLACEMENT = {"required": 0, "greedy": 1}
_PREFLIGHT_WORKSPACE_CAPACITY = 1
_PREFLIGHT_REQUIRED_CAPACITY = 2
_PREFLIGHT_MISSING_INITIAL_RESIDENCY = 3


@dataclass(frozen=True, slots=True)
class CCandidateDiagnostic:
    status: int
    strategy: str
    rule: str
    coalesced: bool
    repairs: PressureFitRepairDiagnostics
    work: PressureFitWorkDiagnostics
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

    @property
    def repair_attempts(self) -> int:
        return self.repairs.total_attempts


@dataclass(frozen=True, slots=True)
class CompiledIndexedSchedule:
    """Copied contiguous indices for one context winner.

    Keeping the context-local winner indexed avoids constructing and validating
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
class CContextResult:
    selected_candidate_index: int | None
    selected_makespan_ns: int | None
    selected_schedule: CompiledIndexedSchedule | None
    candidates: tuple[CCandidateDiagnostic, ...]
    repairs: PressureFitRepairDiagnostics
    work: PressureFitWorkDiagnostics

    def __post_init__(self) -> None:
        repairs = PressureFitRepairDiagnostics()
        candidate_work = PressureFitWorkDiagnostics()
        for candidate in self.candidates:
            repairs += candidate.repairs
            candidate_work += candidate.work
        if repairs != self.repairs:
            raise RuntimeError(
                "PressureFit context repair counters do not reconcile"
            )
        for name in candidate_work.__dataclass_fields__:
            if getattr(candidate_work, name) > getattr(self.work, name):
                raise RuntimeError(
                    f"PressureFit candidate work exceeds context work: {name}"
                )


class ContextPreparationError(RuntimeError):
    """The topology could not be normalized into planner facts."""


@dataclass(frozen=True, slots=True)
class CPreflightResult:
    """Structured semantic-feasibility result from the planner."""

    failure_kind: str | None
    error_device: int | None
    error_alias: int | None
    error_boundary: int | None
    required_bytes: int | None
    capacity_bytes: int | None

    @property
    def valid(self) -> bool:
        return self.failure_kind is None


def decode_candidate_diagnostic(
    value: CCandidateDiagnostic,
    *,
    selection_id: str,
    simulation: IndexedSimulationTemplate,
) -> CandidateDiagnostic:
    """Convert one indexed diagnostic without changing its semantic fields."""

    if value.status == 0:
        return CandidateDiagnostic(
            candidate_id=value.candidate_id,
            selection_id=selection_id,
            status="valid",
            makespan_ns=value.makespan_ns,
            schedule_digest=value.schedule_digest,
            repairs=value.repairs,
            work=value.work,
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
            repairs=value.repairs,
            work=value.work,
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
            repairs=value.repairs,
            work=value.work,
        )
    if value.status == 5:
        if value.simulation_status == 0:
            device_id = simulation.device_ids[value.error_device]
            last_result = (
                "dynamic MemoryPool admission cannot place a compatible "
                f"range: device={device_id!r}, "
                f"capacity={value.error_capacity_bytes}, "
                f"used={value.error_used_bytes}, "
                f"request={value.error_requested_bytes}, "
                f"additional_slack={value.error_required_bytes}"
            )
        else:
            last_result = simulation_failure_detail(
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
            status="exhausted",
            failure_kind="repair_budget_exhausted",
            failure_detail=(
                "candidate repair budget exhausted after "
                f"{value.repair_attempts} monotonic repairs; last result: "
                f"{last_result}"
            ),
            repairs=value.repairs,
            work=value.work,
        )
    if value.status != 2:
        raise RuntimeError(
            f"PressureFit candidate {value.candidate_id!r} "
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
        repairs=value.repairs,
        work=value.work,
    )


def _decode_repairs(
    value: CPressureFitRepairDiagnostics,
) -> PressureFitRepairDiagnostics:
    return PressureFitRepairDiagnostics(
        admission_prefetch_advance_attempts=int(
            value.admission_prefetch_advance_attempts
        ),
        admission_prefetch_delay_attempts=int(value.admission_prefetch_delay_attempts),
        admission_pressure_boundary_attempts=int(
            value.admission_pressure_boundary_attempts
        ),
        simulation_prefetch_delay_attempts=int(
            value.simulation_prefetch_delay_attempts
        ),
        simulation_pressure_boundary_attempts=int(
            value.simulation_pressure_boundary_attempts
        ),
    )


def _decode_work(value: CPressureFitWorkDiagnostics) -> PressureFitWorkDiagnostics:
    return PressureFitWorkDiagnostics(
        evaluation_time_ns=int(value.evaluation_time_ns),
        residency_cache_hits=int(value.residency_cache_hits),
        residency_cache_misses=int(value.residency_cache_misses),
        schedule_emissions=int(value.schedule_emissions),
        schedule_cache_hits=int(value.schedule_cache_hits),
        simulation_calls=int(value.simulation_calls),
        simulation_cache_hits=int(value.simulation_cache_hits),
        admission_calls=int(value.admission_calls),
        residency_time_ns=int(value.residency_time_ns),
        schedule_time_ns=int(value.schedule_time_ns),
        simulation_time_ns=int(value.simulation_time_ns),
        admission_time_ns=int(value.admission_time_ns),
        digest_time_ns=int(value.digest_time_ns),
    )


@lru_cache(maxsize=131_072)
def _escaped_identifier(value: str) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return encoded[1:-1].encode("utf-8")


def _name_arrays(
    simulation: IndexedSimulationTemplate,
) -> tuple[
    ctypes.Array[ctypes.c_char_p],
    ctypes.Array[ctypes.c_char_p],
]:
    alias_payloads = tuple(_escaped_identifier(value) for value in simulation.alias_ids)
    task_payloads = tuple(_escaped_identifier(value) for value in simulation.task_ids)
    alias_names = (ctypes.c_char_p * max(1, len(alias_payloads)))(*alias_payloads)
    task_names = (ctypes.c_char_p * max(1, len(task_payloads)))(*task_payloads)
    return alias_names, task_names


def _program_context(
    simulation: IndexedSimulationTemplate,
    admission: IndexedAdmissionTopology | None,
) -> tuple[CPressureFitProgramContext, tuple[object, ...]]:
    alias_names, task_names = _name_arrays(simulation)
    device_ranks = {
        device_id: rank for rank, device_id in enumerate(sorted(simulation.device_ids))
    }
    priorities = (ctypes.c_uint32 * max(1, len(simulation.device_ids)))(
        *(device_ranks[value] for value in simulation.device_ids)
    )
    context = CPressureFitProgramContext(
        abi_version=ABI_VERSION,
        simulation=ctypes.pointer(simulation.program),
        device_priority=priorities,
        admission=ctypes.pointer(admission.value) if admission is not None else None,
        alias_json_names=alias_names,
        task_json_names=task_names,
    )
    return context, (alias_names, task_names, priorities)


def _context_options(
    options: PressureFitOptions,
) -> tuple[
    CPressureFitContextOptions,
    tuple[str, ...],
    tuple[str, ...],
    tuple[object, ...],
]:
    strategy_names = tuple(options.residency_strategies)
    rule_names = tuple(options.prefetch_rules)
    strategies = (ctypes.c_uint8 * len(strategy_names))(
        *(_STRATEGY_CODE[value] for value in strategy_names)
    )
    rules = (ctypes.c_uint8 * len(rule_names))(
        *(_RULE_CODE[value] for value in rule_names)
    )
    compiled = CPressureFitContextOptions(
        residency_strategies=strategies,
        residency_strategy_count=len(strategy_names),
        prefetch_rules=rules,
        prefetch_rule_count=len(rule_names),
        evaluate_coalesced=int(options.evaluate_coalesced),
        max_repair_attempts=options.max_repair_attempts,
        initial_placement=_INITIAL_PLACEMENT[options.initial_placement.value],
    )
    return compiled, strategy_names, rule_names, (strategies, rules)


def validate_program_context(
    simulation: IndexedSimulationTemplate,
    *,
    admission: IndexedAdmissionTopology | None = None,
) -> CPreflightResult:
    """Validate one selected topology using the planner authority."""

    context, _buffers = _program_context(simulation, admission)
    result = CPressureFitPreflightResult()
    library = load_planner_library()
    status = int(
        library.shadowspill_validate_pressurefit_program_context(
            ctypes.byref(context),
            ctypes.byref(result),
        )
    )
    if status != int(result.status):
        raise RuntimeError(
            "PressureFit preflight returned inconsistent status"
        )
    if status == 0:
        return CPreflightResult(None, None, None, None, None, None)
    failure_kind = int(result.failure_kind)
    if failure_kind not in {
        _PREFLIGHT_WORKSPACE_CAPACITY,
        _PREFLIGHT_REQUIRED_CAPACITY,
        _PREFLIGHT_MISSING_INITIAL_RESIDENCY,
    }:
        encoded = library.shadowspill_status_string(status)
        message = encoded.decode("utf-8") if encoded else f"planner status {status}"
        raise ContextPreparationError(message)
    failure_names = {
        _PREFLIGHT_WORKSPACE_CAPACITY: "workspace_capacity",
        _PREFLIGHT_REQUIRED_CAPACITY: "required_capacity",
        _PREFLIGHT_MISSING_INITIAL_RESIDENCY: "missing_initial_residency",
    }
    return CPreflightResult(
        failure_kind=failure_names[failure_kind],
        error_device=(
            None if int(result.error_device) == NO_INDEX else int(result.error_device)
        ),
        error_alias=(
            None if int(result.error_alias) == NO_INDEX else int(result.error_alias)
        ),
        error_boundary=(
            None
            if int(result.error_boundary) == -(1 << 31)
            else int(result.error_boundary)
        ),
        required_bytes=int(result.required_bytes),
        capacity_bytes=int(result.capacity_bytes),
    )


def _copy_schedule(result: CPressureFitContextResult) -> CompiledIndexedSchedule:
    value = result.selected_schedule
    return CompiledIndexedSchedule(
        action_trigger_tasks=tuple(
            int(value.action_trigger_tasks[index])
            for index in range(int(value.action_count))
        ),
        action_aliases=tuple(
            int(value.action_aliases[index]) for index in range(int(value.action_count))
        ),
        action_kinds=tuple(
            int(value.action_kinds[index]) for index in range(int(value.action_count))
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
            int(value.final_aliases[index]) for index in range(int(value.final_count))
        ),
        final_locations=tuple(
            int(value.final_locations[index]) for index in range(int(value.final_count))
        ),
    )


def decode_schedule(
    value: CompiledIndexedSchedule,
    simulation: IndexedSimulationTemplate,
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


def _evaluate_context(
    simulation: IndexedSimulationTemplate,
    options: PressureFitOptions,
    *,
    admission: IndexedAdmissionTopology | None,
) -> CContextResult | None:
    """Invoke and decode one compiled program-context evaluation."""

    context_options, strategy_names, rule_names, _option_buffers = (
        _context_options(options)
    )
    context_result = CPressureFitContextResult()
    library = load_planner_library()
    context, _context_buffers = _program_context(simulation, admission)
    status = int(
        library.shadowspill_evaluate_pressurefit_program_context(
            ctypes.byref(context),
            ctypes.byref(context_options),
            ctypes.byref(context_result),
        )
    )
    try:
        if status == Status.ANALYTIC_INFEASIBLE:
            return None
        if status == Status.INVALID_ARGUMENT:
            raise ContextPreparationError(
                "PressureFit context rejected the selected topology"
            )
        if status not in (Status.OK, Status.NO_FEASIBLE_CANDIDATE):
            encoded = library.shadowspill_status_string(status)
            message = encoded.decode("utf-8") if encoded else f"planner status {status}"
            raise RuntimeError(message)
        candidates: list[CCandidateDiagnostic] = []
        for index in range(int(context_result.candidate_count)):
            value = context_result.candidates[index]
            variants_per_rule = 2 if options.evaluate_coalesced else 1
            strategy = strategy_names[index // (len(rule_names) * variants_per_rule)]
            within_strategy = index % (
                len(rule_names) * (2 if options.evaluate_coalesced else 1)
            )
            divisor = 2 if options.evaluate_coalesced else 1
            rule = rule_names[within_strategy // divisor]
            digest = (
                bytes(value.schedule_digest).hex() if int(value.status) == 0 else None
            )
            candidates.append(
                CCandidateDiagnostic(
                    status=int(value.status),
                    strategy=strategy,
                    rule=rule,
                    coalesced=bool(value.coalesced),
                    repairs=_decode_repairs(value.repairs),
                    work=_decode_work(value.work),
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
        selected = int(context_result.selected_candidate_index)
        return CContextResult(
            selected_candidate_index=None if selected == NO_INDEX else selected,
            selected_makespan_ns=(
                None
                if selected == NO_INDEX
                else int(context_result.selected_makespan_ns)
            ),
            selected_schedule=(
                None if selected == NO_INDEX else _copy_schedule(context_result)
            ),
            candidates=tuple(candidates),
            repairs=_decode_repairs(context_result.repairs),
            work=_decode_work(context_result.work),
        )
    finally:
        library.shadowspill_pressurefit_context_result_destroy(
            ctypes.byref(context_result)
        )


def evaluate_program_context(
    simulation: IndexedSimulationTemplate,
    options: PressureFitOptions,
    *,
    admission: IndexedAdmissionTopology | None = None,
) -> CContextResult | None:
    """Derive indexed planning facts and evaluate the portfolio entirely in C."""

    return _evaluate_context(simulation, options, admission=admission)


__all__ = [
    "CCandidateDiagnostic",
    "CContextResult",
    "CPreflightResult",
    "CompiledIndexedSchedule",
    "ContextPreparationError",
    "decode_candidate_diagnostic",
    "decode_schedule",
    "evaluate_program_context",
    "validate_program_context",
]
