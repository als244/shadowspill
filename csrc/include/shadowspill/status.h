#ifndef SHADOWSPILL_STATUS_H
#define SHADOWSPILL_STATUS_H

#include <stdint.h>

#if defined(_WIN32)
#define SHADOWSPILL_STATUS_API __declspec(dllexport)
#else
#define SHADOWSPILL_STATUS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One status vocabulary for the whole library.
 *
 * There used to be four: the simulator, the planner, the runtime and the
 * admission replay each declared their own, and each agreed that 0 was OK, 1
 * was a bad argument and 2 was an internal failure before diverging. Four
 * enums meant four functions spelling out the same three strings, and it made
 * a status meaningless without knowing which component produced it - 3 was
 * "initial device capacity" to the simulator and "out of memory" to the
 * runtime.
 *
 * The three shared codes keep the values every component already used, and
 * each component's own codes occupy a band of their own, so a status decodes
 * to exactly one meaning wherever it came from.
 *
 * Component prefixes remain as aliases below, because a planner function
 * returning a planner-named code reads better at the call site than a
 * general one, and because the compiler should not have to be told about
 * two thousand unchanged expressions to make the values disjoint.
 */
typedef enum ShadowSpillStatus {
    SHADOWSPILL_STATUS_OK = 0,
    SHADOWSPILL_STATUS_INVALID_ARGUMENT = 1,
    /* The library failed at something that was not the caller's request:
     * memory it could not obtain, or an invariant it could not hold. */
    SHADOWSPILL_STATUS_INTERNAL_FAILURE = 2,

    /* Planning, 10-19. */
    SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE = 10,
    SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR = 11,
    SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE = 12,

    /* Simulation, 20-39. */
    SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY = 20,
    SHADOWSPILL_STATUS_INITIAL_HOST_CAPACITY = 21,
    SHADOWSPILL_STATUS_TASK_INPUT_DEADLOCK = 22,
    SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY = 23,
    SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY = 24,
    SHADOWSPILL_STATUS_OFFLOAD_HOST_CAPACITY = 25,
    SHADOWSPILL_STATUS_TRANSFER_DEADLOCK = 26,
    SHADOWSPILL_STATUS_INVALID_RELEASE = 27,
    SHADOWSPILL_STATUS_RELEASE_TRANSFER_CONFLICT = 28,
    SHADOWSPILL_STATUS_INVALID_OFFLOAD = 29,
    SHADOWSPILL_STATUS_INVALID_PREFETCH = 30,
    SHADOWSPILL_STATUS_FINAL_RESIDENCY = 31,
    SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR = 32,

    /* Execution, 40-79. */
    SHADOWSPILL_STATUS_OUT_OF_MEMORY = 40,
    SHADOWSPILL_STATUS_NO_PROGRESS = 41,
    SHADOWSPILL_STATUS_INVALID_STATE = 42,
    SHADOWSPILL_STATUS_PLAN_VIOLATION = 43,
    SHADOWSPILL_STATUS_BACKEND_FAILURE = 44,
    SHADOWSPILL_STATUS_WORKER_FAILURE = 45,
    SHADOWSPILL_STATUS_CLOSED = 46,
    SHADOWSPILL_STATUS_TASK_ALLOCATION_ENVELOPE_EXCEEDED = 47,
    SHADOWSPILL_STATUS_TASK_ALLOCATION_CONTRACT_MISMATCH = 48,

    /* Replaying a schedule's operations, 80-89. */
    SHADOWSPILL_STATUS_REPLAY_INFEASIBLE = 80,
    SHADOWSPILL_STATUS_INVALID_OPERATIONS = 81,
} ShadowSpillStatus;

/* One sentence for any status, from any component. */
SHADOWSPILL_STATUS_API const char *shadowspill_status_string(
    ShadowSpillStatus status
);

#ifdef __cplusplus
}
#endif

#endif
