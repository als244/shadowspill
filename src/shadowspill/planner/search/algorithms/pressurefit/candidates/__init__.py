"""Several resolved problems evaluated by the C search at once.

The records they come back in are in ``records``, reading an answer out of
them in ``decode``, and describing a problem to the search in ``problems``.
"""

import ctypes

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
)
from shadowspill.simulator.indexing import IndexedSimulationTemplate
from shadowspill.status import Status

from .....admission.indexing import IndexedAdmissionFacts, IndexedMemorySchedule
from .....capi import (
    CIndexedProblem,
)
from .....request import GenericPlanningOptions
from ..capi import (
    CPressureFitResult,
    pressurefit_api,
)
from ..options import PressureFitOptions
from .decode import (
    _decode_problem_result,
    decode_candidate_diagnostic,
    decode_schedule,
)
from .problems import (
    _problem_options,
    _program_problem,
    validate_program_problem,
)
from .records import (
    CCandidateDiagnostic,
    CIncumbentOutcome,
    CPreflightResult,
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


def evaluate_program_problems(
    problems: tuple[
        tuple[
            IndexedSimulationTemplate,
            IndexedAdmissionFacts | None,
            IndexedAdmissionFacts | None,
            IndexedMemorySchedule | None,
        ],
        ...,
    ],
    generic: GenericPlanningOptions,
    search_options: PressureFitOptions,
    *,
    workers: int = 0,
    best_placed: int = 0,
) -> tuple[CProblemResult | None, ...]:
    """Evaluate several resolved programs on one set of worker threads.

    The library owns the threads and hands a candidate of a problem to
    whichever worker is free, so worker count and problem count are
    independent -- `workers` sizes the threads whether there is one
    resolved program here or five. Sharing one call is also what shares the
    placement record between them: a plan placed under any of these bounds
    the search under every other. Each problem may carry the plan to beat,
    which the library measures before any candidate runs.
    """

    if not problems:
        return ()
    library = pressurefit_api()
    problem_options, _option_buffers = _problem_options(
        generic, search_options, best_placed=best_placed
    )
    problem_options.workers = workers
    compiled = (CIndexedProblem * len(problems))()
    # Held until the call returns: the library borrows every array in them.
    buffers: list[object] = []
    for index, (simulation, admission, placement, incumbent) in enumerate(problems):
        value, held = _program_problem(simulation, admission, placement, incumbent)
        compiled[index] = value
        buffers.append(held)
    results = (CPressureFitResult * len(problems))()
    status = int(
        library.shadowspill_pressurefit_search(
            compiled,
            len(problems),
            ctypes.byref(problem_options),
            results,
        )
    )
    if status == Status.INVALID_ARGUMENT:
        raise ProblemPreparationError("PressureFit problem rejected the selected facts")
    # Every problem carries its own status; the call's status is only the
    # summary, so each is decoded on its own terms.
    return tuple(
        _decode_problem_result(
            library, int(results[index].status), results[index], simulation
        )
        for index, (simulation, _admission, _placement, _incumbent) in enumerate(
            problems
        )
    )


__all__ = [
    "CCandidateDiagnostic",
    "CIncumbentOutcome",
    "CPreflightResult",
    "CProblemResult",
    "IndexedMemorySchedule",
    "ProblemPreparationError",
    "decode_candidate_diagnostic",
    "decode_schedule",
    "evaluate_program_problems",
    "validate_program_problem",
]
