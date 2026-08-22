#ifndef SHADOWSPILL_ADMISSION_REPLAY_H
#define SHADOWSPILL_ADMISSION_REPLAY_H

#include <stdint.h>

#include <shadowspill/status.h>

#if defined(_WIN32)
#define SHADOWSPILL_ADMISSION_REPLAY_API __declspec(dllexport)
#else
#define SHADOWSPILL_ADMISSION_REPLAY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_ADMISSION_REPLAY_ABI_VERSION 4U
#define SHADOWSPILL_ADMISSION_REPLAY_NO_ID UINT64_MAX

/* Replay names for the shared statuses; see <shadowspill/status.h>. */
typedef ShadowSpillStatus ShadowSpillAdmissionReplayStatus;
#define SHADOWSPILL_ADMISSION_REPLAY_OK SHADOWSPILL_STATUS_OK
#define SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT \
    SHADOWSPILL_STATUS_INVALID_ARGUMENT
#define SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE \
    SHADOWSPILL_STATUS_INTERNAL_FAILURE
#define SHADOWSPILL_ADMISSION_REPLAY_INFEASIBLE SHADOWSPILL_STATUS_REPLAY_INFEASIBLE
#define SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS \
    SHADOWSPILL_STATUS_INVALID_OPERATIONS

/*
 * Operations describe ownership transitions, not transfer semantics. A route,
 * task, or other consumer may reserve a lease and later acquire it.
 */
typedef enum ShadowSpillAdmissionReplayOperationKind {
    SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE = 0,
    SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT = 1,
    SHADOWSPILL_ADMISSION_REPLAY_PUBLISH_DEPENDENCY = 2,
    SHADOWSPILL_ADMISSION_REPLAY_RESERVE = 3,
    SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED = 4,
    SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT = 5,
    SHADOWSPILL_ADMISSION_REPLAY_RELEASE = 6,
} ShadowSpillAdmissionReplayOperationKind;

typedef enum ShadowSpillAdmissionReplayLeaseState {
    SHADOWSPILL_ADMISSION_REPLAY_LEASE_FREE = 0,
    SHADOWSPILL_ADMISSION_REPLAY_LEASE_IN_USE = 1,
    SHADOWSPILL_ADMISSION_REPLAY_LEASE_RETIRE_PENDING = 2,
    SHADOWSPILL_ADMISSION_REPLAY_LEASE_RESERVED = 3,
    SHADOWSPILL_ADMISSION_REPLAY_LEASE_SUCCESSOR_RESERVED = 4,
    SHADOWSPILL_ADMISSION_REPLAY_LEASE_PREDECESSOR_TRANSFERRED = 5,
} ShadowSpillAdmissionReplayLeaseState;

typedef struct ShadowSpillAdmissionReplayOperation {
    uint64_t sequence;
    uint64_t lease_id;
    uint64_t dependency_id;
    uint64_t bytes;
    uint64_t alignment;
    uint8_t kind;
    uint8_t dependency_expected;
} ShadowSpillAdmissionReplayOperation;

typedef struct ShadowSpillAdmissionReplayProgram {
    uint32_t abi_version;
    uint64_t capacity_bytes;
    uint64_t minimum_alignment;
    /*
     * Zero preserves ordinary low-address best fit.  Otherwise requests at
     * least this large split the selected free range from its high end.  The
     * policy keeps one globally coalescing arena; it does not partition or
     * reserve capacity for either size class.
     */
    uint64_t large_request_threshold_bytes;
    uint64_t lease_count;
    uint64_t dependency_count;
    const ShadowSpillAdmissionReplayOperation *operations;
    uint64_t operation_count;
} ShadowSpillAdmissionReplayProgram;

/* One deterministic allocator decision, aligned one-to-one with an operation. */
typedef struct ShadowSpillAdmissionReplayDecision {
    uint64_t operation_index;
    uint64_t sequence;
    uint64_t lease_id;
    uint64_t predecessor_lease_id;
    uint64_t dependency_id;
    uint64_t offset;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    int64_t physical_bytes_delta;
    uint8_t resulting_state;
} ShadowSpillAdmissionReplayDecision;

/* A consumer must not use the successor until this dependency completes. */
typedef struct ShadowSpillAdmissionReuseDependency {
    uint64_t predecessor_lease_id;
    uint64_t successor_lease_id;
    uint64_t dependency_id;
    uint64_t consumer_operation_index;
} ShadowSpillAdmissionReuseDependency;

/* Exact physical allocation ledger at the first infeasible operation. */
typedef struct ShadowSpillAdmissionReplayLiveLease {
    uint64_t lease_id;
    uint64_t offset;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint8_t state;
} ShadowSpillAdmissionReplayLiveLease;

typedef struct ShadowSpillAdmissionReplayResult {
    uint32_t status;
    uint64_t error_operation_index;
    uint64_t error_lease_id;
    uint64_t error_requested_bytes;
    uint64_t error_free_bytes;
    uint64_t error_largest_free_range_bytes;
    uint64_t peak_allocated_bytes;
    uint64_t peak_reserved_bytes;
    uint64_t peak_fragmentation_bytes;
    uint64_t final_allocated_bytes;
    uint64_t final_reserved_bytes;
    uint64_t final_largest_free_range_bytes;
    uint64_t decision_digest;

    ShadowSpillAdmissionReplayDecision *decisions;
    uint64_t decision_capacity;
    uint64_t decision_count;
    ShadowSpillAdmissionReuseDependency *dependencies;
    uint64_t dependency_capacity;
    uint64_t dependency_result_count;
    ShadowSpillAdmissionReplayLiveLease *live_leases;
    uint64_t live_lease_capacity;
    uint64_t live_lease_count;
} ShadowSpillAdmissionReplayResult;

/*
 * Opaque reusable scratch storage for repeated replay. A workspace owns no
 * backend arena and performs no I/O. It is not thread-safe: one caller may
 * use a workspace at a time, while distinct workspaces are independent.
 */
typedef struct ShadowSpillAdmissionReplayWorkspace
    ShadowSpillAdmissionReplayWorkspace;

/*
 * Replays one ordered operation sequence through the production MemoryPool
 * policy. Input
 * and output buffers are borrowed for the call. Lease and dependency IDs are
 * contiguous zero-based indices bounded by their respective counts. The function
 * performs no backend operations and owns no storage after it returns.
 */
SHADOWSPILL_ADMISSION_REPLAY_API uint32_t shadowspill_admission_replay_abi_version(
    void
);

SHADOWSPILL_ADMISSION_REPLAY_API ShadowSpillAdmissionReplayStatus
shadowspill_admission_replay_run(
    const ShadowSpillAdmissionReplayProgram *program,
    ShadowSpillAdmissionReplayResult *result
);

/*
 * Allocate all lease, dependency, synchronization, and range-node scratch
 * used by repeated replay. Subsequent run_reusing calls perform no heap
 * allocation when the supplied program fits these capacities.
 */
SHADOWSPILL_ADMISSION_REPLAY_API ShadowSpillAdmissionReplayStatus
shadowspill_admission_replay_workspace_create(
    uint64_t lease_capacity,
    uint64_t dependency_capacity,
    ShadowSpillAdmissionReplayWorkspace **workspace
);

SHADOWSPILL_ADMISSION_REPLAY_API ShadowSpillAdmissionReplayStatus
shadowspill_admission_replay_run_reusing(
    const ShadowSpillAdmissionReplayProgram *program,
    ShadowSpillAdmissionReplayResult *result,
    ShadowSpillAdmissionReplayWorkspace *workspace
);

SHADOWSPILL_ADMISSION_REPLAY_API void
shadowspill_admission_replay_workspace_destroy(
    ShadowSpillAdmissionReplayWorkspace *workspace
);

SHADOWSPILL_ADMISSION_REPLAY_API const char *shadowspill_admission_replay_status_string(
    ShadowSpillAdmissionReplayStatus status
);

#ifdef __cplusplus
}
#endif

#endif
