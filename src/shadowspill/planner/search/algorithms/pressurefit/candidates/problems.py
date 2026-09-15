"""One resolved problem described to the C search, and what it must satisfy."""

import ctypes

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
)
from shadowspill.simulator.indexing import IndexedSimulationTemplate
from shadowspill.status import ABI_VERSION

from .....admission.indexing import IndexedAdmissionFacts, IndexedMemorySchedule
from .....capi import (
    NO_INDEX,
    CIndexedProblem,
    CScheduleContext,
)
from .....request import GenericPlanningOptions
from ..capi import (
    CPressureFitOptions,
    CPressureFitPreflightResult,
    pressurefit_api,
)
from ..options import PressureFitOptions
from .decode import _escaped_identifier, _indexed_schedule
from .records import CPreflightResult, ProblemPreparationError

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


def _program_problem(
    simulation: IndexedSimulationTemplate,
    admission: IndexedAdmissionFacts | None,
    placement: IndexedAdmissionFacts | None = None,
    incumbent: IndexedMemorySchedule | None = None,
) -> tuple[CIndexedProblem, tuple[object, ...]]:
    alias_names, task_names = _name_arrays(simulation)
    carried: tuple[object, ...] = ()
    incumbent_value = None
    if incumbent is not None:
        incumbent_value, carried = _indexed_schedule(incumbent)
    device_ranks = {
        device_id: rank for rank, device_id in enumerate(sorted(simulation.device_ids))
    }
    priorities = (ctypes.c_uint32 * max(1, len(simulation.device_ids)))(
        *(device_ranks[value] for value in simulation.device_ids)
    )
    problem = CIndexedProblem(
        abi_version=ABI_VERSION,
        context=CScheduleContext(
            simulation=ctypes.pointer(simulation.program),
            admission=(
                ctypes.pointer(admission.value) if admission is not None else None
            ),
            # Placement measures layouts during the search; it does not
            # prefilter through the dynamic-pool replay, which `admission`
            # above would switch on.
            placement=(
                ctypes.pointer(placement.value) if placement is not None else None
            ),
            alias_json_names=alias_names,
            task_json_names=task_names,
        ),
        device_priority=priorities,
        incumbent=(
            ctypes.pointer(incumbent_value) if incumbent_value is not None else None
        ),
    )
    return problem, (alias_names, task_names, priorities, incumbent_value, carried)


def _problem_options(
    generic: GenericPlanningOptions,
    search_options: PressureFitOptions,
    *,
    best_placed: int = 0,
) -> tuple[CPressureFitOptions, tuple[object, ...]]:
    strategy_names = tuple(search_options.residency_strategies)
    rule_names = tuple(search_options.fetch_rules)
    strategies = (ctypes.c_uint8 * len(strategy_names))(
        *(_STRATEGY_CODE[value] for value in strategy_names)
    )
    rules = (ctypes.c_uint8 * len(rule_names))(
        *(_RULE_CODE[value] for value in rule_names)
    )
    # The library takes the modes as a list like the other two axes; the
    # option a caller sets is the bool.
    mode_values = (0, 1) if search_options.evaluate_coalesced else (0,)
    modes = (ctypes.c_uint8 * len(mode_values))(*mode_values)
    compiled = CPressureFitOptions(
        residency_strategies=strategies,
        residency_strategy_count=len(strategy_names),
        fetch_rules=rules,
        fetch_rule_count=len(rule_names),
        coalescing_modes=modes,
        coalescing_mode_count=len(mode_values),
        max_repair_attempts=search_options.max_repair_attempts,
        initial_placement=_INITIAL_PLACEMENT[search_options.initial_placement.value],
        capacity_refinement_bytes=search_options.capacity_refinement_bytes,
        record_reduction_steps=int(search_options.record_reduction_steps),
        best_placed=best_placed or None,
        deterministic=int(generic.deterministic),
        minimum_object_bytes_evict_eligible=generic.minimum_object_bytes_evict_eligible,
    )
    return compiled, (strategies, rules, modes)


def validate_program_problem(
    simulation: IndexedSimulationTemplate,
    *,
    admission: IndexedAdmissionFacts | None = None,
) -> CPreflightResult:
    """Validate one selected facts using the planner authority."""

    problem, _buffers = _program_problem(simulation, admission)
    result = CPressureFitPreflightResult()
    library = pressurefit_api()
    status = int(
        library.shadowspill_pressurefit_preflight(
            ctypes.byref(problem),
            ctypes.byref(result),
        )
    )
    if status != int(result.status):
        raise RuntimeError("PressureFit preflight returned inconsistent status")
    if status == 0:
        return CPreflightResult(None, None, None, None, None, None)
    failure_kind = int(result.failure_kind)
    if failure_kind not in {
        _PREFLIGHT_WORKSPACE_CAPACITY,
        _PREFLIGHT_REQUIRED_CAPACITY,
        _PREFLIGHT_MISSING_INITIAL_RESIDENCY,
        _PREFLIGHT_RESIDENT_SLICE_CAPACITY,
    }:
        encoded = library.shadowspill_status_string(status)
        message = encoded.decode("utf-8") if encoded else f"planner status {status}"
        raise ProblemPreparationError(message)
    failure_names = {
        _PREFLIGHT_WORKSPACE_CAPACITY: "workspace_capacity",
        _PREFLIGHT_REQUIRED_CAPACITY: "required_capacity",
        _PREFLIGHT_MISSING_INITIAL_RESIDENCY: "missing_initial_residency",
        _PREFLIGHT_RESIDENT_SLICE_CAPACITY: "resident_slice_capacity",
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
