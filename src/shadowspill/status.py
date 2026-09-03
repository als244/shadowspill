"""The library's ABI version and status vocabulary, mirroring the C headers.

Every entry point returns one of these. The three shared codes mean
the same thing wherever they come from; each component's own codes occupy a
band, so a status decodes to exactly one meaning without knowing which
component produced it.

Keep this in step with the C header. The values are the contract.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

#: Everything in libshadowspill ships together and versions together; see
#: `<shadowspill/shadowspill.h>`. Backends and the PyTorch adapter are
#: compiled separately and keep their own.
ABI_VERSION: Final = 1


class Status(IntEnum):
    """A status returned by any compiled ShadowSpill entry point."""

    OK = 0
    INVALID_ARGUMENT = 1
    #: The library failed at something that was not the caller's request:
    #: memory it could not obtain, or an invariant it could not hold.
    INTERNAL_FAILURE = 2

    # Planning, 10-19.
    NO_FEASIBLE_CANDIDATE = 10
    PLANNER_INTERNAL_ERROR = 11
    ANALYTIC_INFEASIBLE = 12

    # Simulation, 20-39.
    INITIAL_DEVICE_CAPACITY = 20
    INITIAL_SPILL_CAPACITY = 21
    TASK_INPUT_DEADLOCK = 22
    TASK_DEVICE_CAPACITY = 23
    FETCH_DEVICE_CAPACITY = 24
    EVICT_SPILL_CAPACITY = 25
    TRANSFER_DEADLOCK = 26
    INVALID_RELEASE = 27
    RELEASE_TRANSFER_CONFLICT = 28
    INVALID_EVICT = 29
    INVALID_FETCH = 30
    FINAL_RESIDENCY = 31
    SIMULATION_INTERNAL_ERROR = 32

    # Execution, 40-79.
    OUT_OF_MEMORY = 40
    NO_PROGRESS = 41
    INVALID_STATE = 42
    PLAN_VIOLATION = 43
    BACKEND_FAILURE = 44
    WORKER_FAILURE = 45
    CLOSED = 46
    TASK_ALLOCATION_ENVELOPE_EXCEEDED = 47
    TASK_ALLOCATION_CONTRACT_MISMATCH = 48

    # Replaying a schedule's operations, 80-89.
    REPLAY_INFEASIBLE = 80
    INVALID_OPERATIONS = 81


__all__ = ["ABI_VERSION", "Status"]
