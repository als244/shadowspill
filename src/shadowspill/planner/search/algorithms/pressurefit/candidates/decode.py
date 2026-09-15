"""What the search answered, read back out of those records."""

from __future__ import annotations

import ctypes
import json
from dataclasses import replace
from functools import lru_cache

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ResidencySpec,
)
from shadowspill.simulator.diagnostics import (
    simulation_failure_detail,
    simulation_status_kind,
)
from shadowspill.simulator.indexing import IndexedSimulationTemplate
from shadowspill.status import Status

from .....admission.indexing import IndexedMemorySchedule
from .....capi import (
    NO_INDEX,
    CIndexedSchedule,
)
from .....diagnostics import (
    CandidateDiagnostic,
    PlanningRepairDiagnostics,
    PlanningSectionTiming,
    PlanningWorkDiagnostics,
    ReductionStep,
)
from .....diagnostics.counters import STEP_OUTCOMES
from ..capi import (
    CandidateStatus,
    CPressureFitCandidateDiagnostic,
    CPressureFitRepairDiagnostics,
    CPressureFitResult,
    CPressureFitSectionTiming,
    CPressureFitWorkDiagnostics,
)
from .records import (
    CCandidateDiagnostic,
    CIncumbentOutcome,
    CProblemResult,
    ProblemPreparationError,
)

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
_STRATEGY_NAME = {code: name for name, code in _STRATEGY_CODE.items()}
_RULE_NAME = {code: name for name, code in _RULE_CODE.items()}
_ACTION_KIND = {
    0: MemoryActionKind.RELEASE,
    1: MemoryActionKind.EVICT,
    2: MemoryActionKind.FETCH,
    3: MemoryActionKind.WRITE_BACK,
}
_LOCATION = {0: MemoryLocation.DEVICE, 1: MemoryLocation.SPILL}
_INITIAL_PLACEMENT = {"required": 0, "greedy": 1}
_PREFLIGHT_WORKSPACE_CAPACITY = 1
_PREFLIGHT_REQUIRED_CAPACITY = 2
_PREFLIGHT_RESIDENT_SLICE_CAPACITY = 4
_PREFLIGHT_MISSING_INITIAL_RESIDENCY = 3


_SECTION_FIELDS = tuple(name for name, *_ in CPressureFitSectionTiming._fields_)
_STEP_FLAGS = tuple((name, 1 << bit) for bit, name in enumerate(STEP_OUTCOMES))


def decode_candidate_diagnostic(
    value: CCandidateDiagnostic,
    *,
    selection_id: str,
    simulation: IndexedSimulationTemplate,
) -> CandidateDiagnostic:
    """Convert one indexed diagnostic without changing its semantic fields."""

    return replace(
        _decode_candidate_status(
            value, selection_id=selection_id, simulation=simulation
        ),
        started_ns=value.started_ns,
        finished_ns=value.finished_ns,
    )


def _decode_candidate_status(
    value: CCandidateDiagnostic,
    *,
    selection_id: str,
    simulation: IndexedSimulationTemplate,
) -> CandidateDiagnostic:
    """The shape of the diagnostic, which depends on how the candidate ended."""

    if value.status == 0:
        return CandidateDiagnostic(
            candidate_id=value.candidate_id,
            selection_id=selection_id,
            status="valid",
            makespan_ns=value.makespan_ns,
            capacity_violation_count=value.capacity_violation_count,
            placements_attempted=value.placements_attempted,
            placements_admitted=value.placements_admitted,
            capacity_refinements=value.capacity_refinements,
            repairs_at_best=value.repairs_at_best,
            pressure_escalations=value.pressure_escalations,
            escalations_taken_back=value.escalations_taken_back,
            schedule_digest=value.schedule_digest,
            repairs=value.repairs,
            work=value.work,
            steps=value.steps,
        )
    if value.status == CandidateStatus.UNPLACEABLE:
        return CandidateDiagnostic(
            candidate_id=value.candidate_id,
            selection_id=selection_id,
            status="infeasible",
            failure_kind="unplaceable",
            failure_detail=(
                "every plan this candidate reached needed more contiguous "
                "pool than the pool has; it was reduced "
                f"{value.capacity_refinements} times and measured "
                f"{value.placements_attempted}"
            ),
            capacity_violation_count=value.capacity_violation_count,
            placements_attempted=value.placements_attempted,
            placements_admitted=value.placements_admitted,
            capacity_refinements=value.capacity_refinements,
            pressure_escalations=value.pressure_escalations,
            escalations_taken_back=value.escalations_taken_back,
            repairs=value.repairs,
            work=value.work,
            steps=value.steps,
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
            steps=value.steps,
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
            steps=value.steps,
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
            steps=value.steps,
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
) -> PlanningRepairDiagnostics:
    return PlanningRepairDiagnostics(
        admission_fetch_advance_attempts=int(value.admission_fetch_advance_attempts),
        admission_fetch_delay_attempts=int(value.admission_fetch_delay_attempts),
        admission_pressure_boundary_attempts=int(
            value.admission_pressure_boundary_attempts
        ),
        simulation_fetch_delay_attempts=int(value.simulation_fetch_delay_attempts),
        simulation_pressure_boundary_attempts=int(
            value.simulation_pressure_boundary_attempts
        ),
    )


def _decode_sections(
    value: CPressureFitSectionTiming,
) -> PlanningSectionTiming:
    return PlanningSectionTiming(
        **{name: int(getattr(value, name)) for name in _SECTION_FIELDS}
    )


def _decode_work(value: CPressureFitWorkDiagnostics) -> PlanningWorkDiagnostics:
    return PlanningWorkDiagnostics(
        schedule_emissions=int(value.schedule_emissions),
        schedule_cache_hits=int(value.schedule_cache_hits),
        simulation_calls=int(value.simulation_calls),
        simulation_cache_hits=int(value.simulation_cache_hits),
        admission_calls=int(value.admission_calls),
        sections=_decode_sections(value.sections),
    )


def _decode_steps(
    value: CPressureFitCandidateDiagnostic,
) -> tuple[ReductionStep, ...]:
    """Copy one candidate's trajectory out of planner-owned memory.

    Empty unless the caller asked for a trajectory, in which case the arrays
    below belong to the result and stop existing when it does.
    """

    if not value.steps or value.step_count == 0:
        return ()
    aliases = value.cut_aliases[: value.cut_count] if value.cut_aliases else []
    return tuple(
        ReductionStep(
            makespan_ns=int(step.makespan_ns),
            required_bytes=int(step.required_bytes),
            capacity_bytes=int(step.capacity_bytes),
            cut_aliases=tuple(
                int(alias)
                for alias in aliases[step.cut_offset : step.cut_offset + step.cut_count]
            ),
            repairs=int(step.repairs),
            simulation_status=int(step.simulation_status),
            capacity_violations=int(step.capacity_violations),
            **{name: bool(step.flags & bit) for name, bit in _STEP_FLAGS},
        )
        for step in value.steps[: value.step_count]
    )


@lru_cache(maxsize=131_072)
def _escaped_identifier(value: str) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return encoded[1:-1].encode("utf-8")


def _decode_problem_result(
    library: ctypes.CDLL,
    status: int,
    problem_result: CPressureFitResult,
    simulation: IndexedSimulationTemplate,
) -> CProblemResult | None:
    """Copy one evaluation out of planner-owned memory and release it."""

    try:
        if status == Status.ANALYTIC_INFEASIBLE:
            return None
        if status == Status.INVALID_ARGUMENT:
            raise ProblemPreparationError(
                "PressureFit problem rejected the selected facts"
            )
        if status not in (Status.OK, Status.NO_FEASIBLE_CANDIDATE):
            encoded = library.shadowspill_status_string(status)
            message = encoded.decode("utf-8") if encoded else f"planner status {status}"
            raise RuntimeError(message)
        candidates: list[CCandidateDiagnostic] = []
        for index in range(int(problem_result.candidate_count)):
            value = problem_result.candidates[index]
            strategy = _STRATEGY_NAME[int(value.residency_strategy)]
            rule = _RULE_NAME[int(value.fetch_rule)]
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
                    capacity_violation_count=int(value.capacity_violation_count),
                    placements_attempted=int(value.placements_attempted),
                    placements_admitted=int(value.placements_admitted),
                    capacity_refinements=int(value.capacity_refinements),
                    repairs_at_best=(
                        None
                        if value.repairs_at_best == 0xFFFFFFFF
                        else int(value.repairs_at_best)
                    ),
                    pressure_escalations=int(value.pressure_escalations),
                    escalations_taken_back=int(value.escalations_taken_back),
                    started_ns=int(value.started_ns),
                    finished_ns=int(value.finished_ns),
                    steps=_decode_steps(value),
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
        selected = int(problem_result.selected_candidate_index)
        incumbent = (
            CIncumbentOutcome(
                status=int(problem_result.incumbent_status),
                makespan_ns=int(problem_result.incumbent_makespan_ns),
                required_bytes=int(problem_result.incumbent_required_bytes),
                selected=bool(problem_result.incumbent_selected),
            )
            if problem_result.incumbent_given
            else None
        )
        answered = selected != NO_INDEX or (
            incumbent is not None and incumbent.selected
        )
        return CProblemResult(
            selected_candidate_index=None if selected == NO_INDEX else selected,
            selected_makespan_ns=(
                int(problem_result.selected_makespan_ns) if answered else None
            ),
            selected_schedule=_copy_schedule(problem_result) if answered else None,
            candidates=tuple(candidates),
            repairs=_decode_repairs(problem_result.repairs),
            work=_decode_work(problem_result.work),
            started_ns=int(problem_result.started_ns),
            finished_ns=int(problem_result.finished_ns),
            evict_ineligible_aliases=int(problem_result.evict_ineligible_aliases),
            evict_ineligible_bytes=int(problem_result.evict_ineligible_bytes),
            resident_slice_bytes=(
                int(problem_result.resident_slice_bytes[0])
                if problem_result.resident_slice_bytes
                else 0
            ),
            resident_aliases=(
                tuple(
                    index
                    for index in range(len(simulation.alias_ids))
                    if problem_result.alias_evict_eligible[index] == 0
                )
                if problem_result.alias_evict_eligible
                else ()
            ),
            incumbent=incumbent,
        )
    finally:
        library.shadowspill_pressurefit_result_destroy(ctypes.byref(problem_result))


def decode_schedule(
    value: IndexedMemorySchedule,
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


def _copy_schedule(result: CPressureFitResult) -> IndexedMemorySchedule:
    value = result.selected_schedule
    return IndexedMemorySchedule(
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


def _indexed_schedule(
    schedule: IndexedMemorySchedule,
) -> tuple[CIndexedSchedule, tuple[object, ...]]:
    """A schedule as the library reads it, with the arrays it borrows."""

    def u32(values: tuple[int, ...]) -> object:
        return (ctypes.c_uint32 * max(1, len(values)))(*values)

    def u8(values: tuple[int, ...]) -> object:
        return (ctypes.c_uint8 * max(1, len(values)))(*values)

    buffers = (
        u32(schedule.action_trigger_tasks),
        u32(schedule.action_aliases),
        u8(schedule.action_kinds),
        u32(schedule.initial_aliases),
        u8(schedule.initial_locations),
        u32(schedule.final_aliases),
        u8(schedule.final_locations),
    )
    value = CIndexedSchedule(
        action_count=len(schedule.action_kinds),
        action_trigger_tasks=buffers[0],
        action_aliases=buffers[1],
        action_kinds=buffers[2],
        initial_count=len(schedule.initial_aliases),
        initial_aliases=buffers[3],
        initial_locations=buffers[4],
        final_count=len(schedule.final_aliases),
        final_aliases=buffers[5],
        final_locations=buffers[6],
    )
    return value, buffers
