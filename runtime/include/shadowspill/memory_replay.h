#ifndef SHADOWSPILL_MEMORY_REPLAY_H
#define SHADOWSPILL_MEMORY_REPLAY_H

#include <stdint.h>

#if defined(_WIN32)
#define SHADOWSPILL_MEMORY_REPLAY_API __declspec(dllexport)
#else
#define SHADOWSPILL_MEMORY_REPLAY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_MEMORY_REPLAY_ABI_VERSION 1U
#define SHADOWSPILL_MEMORY_REPLAY_NO_ID UINT64_MAX

typedef enum ShadowSpillMemoryReplayStatus {
    SHADOWSPILL_MEMORY_REPLAY_OK = 0,
    SHADOWSPILL_MEMORY_REPLAY_INVALID_ARGUMENT = 1,
    SHADOWSPILL_MEMORY_REPLAY_ALLOCATION_FAILURE = 2,
    SHADOWSPILL_MEMORY_REPLAY_INFEASIBLE = 3,
    SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT = 4,
} ShadowSpillMemoryReplayStatus;

/*
 * Operations describe ownership transitions, not transfer semantics. A route,
 * task, or other consumer may reserve a lease and later acquire it.
 */
typedef enum ShadowSpillMemoryReplayOperationKind {
    SHADOWSPILL_MEMORY_REPLAY_ACQUIRE = 0,
    SHADOWSPILL_MEMORY_REPLAY_BEGIN_RETIREMENT = 1,
    SHADOWSPILL_MEMORY_REPLAY_PUBLISH_DEPENDENCY = 2,
    SHADOWSPILL_MEMORY_REPLAY_RESERVE = 3,
    SHADOWSPILL_MEMORY_REPLAY_ACQUIRE_RESERVED = 4,
    SHADOWSPILL_MEMORY_REPLAY_COMPLETE_RETIREMENT = 5,
    SHADOWSPILL_MEMORY_REPLAY_RELEASE = 6,
} ShadowSpillMemoryReplayOperationKind;

typedef enum ShadowSpillMemoryReplayLeaseState {
    SHADOWSPILL_MEMORY_REPLAY_LEASE_FREE = 0,
    SHADOWSPILL_MEMORY_REPLAY_LEASE_IN_USE = 1,
    SHADOWSPILL_MEMORY_REPLAY_LEASE_RETIRE_PENDING = 2,
    SHADOWSPILL_MEMORY_REPLAY_LEASE_RESERVED = 3,
    SHADOWSPILL_MEMORY_REPLAY_LEASE_SUCCESSOR_RESERVED = 4,
    SHADOWSPILL_MEMORY_REPLAY_LEASE_PREDECESSOR_TRANSFERRED = 5,
} ShadowSpillMemoryReplayLeaseState;

typedef struct ShadowSpillMemoryReplayOperation {
    uint64_t sequence;
    uint64_t lease_id;
    uint64_t dependency_id;
    uint64_t bytes;
    uint64_t alignment;
    uint8_t kind;
    uint8_t dependency_expected;
} ShadowSpillMemoryReplayOperation;

typedef struct ShadowSpillMemoryReplayProgram {
    uint32_t abi_version;
    uint64_t capacity_bytes;
    uint64_t minimum_alignment;
    uint64_t lease_count;
    uint64_t dependency_count;
    const ShadowSpillMemoryReplayOperation *operations;
    uint64_t operation_count;
} ShadowSpillMemoryReplayProgram;

/* One deterministic allocator decision, aligned one-to-one with an operation. */
typedef struct ShadowSpillMemoryReplayDecision {
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
} ShadowSpillMemoryReplayDecision;

/* A consumer must not use the successor until this dependency completes. */
typedef struct ShadowSpillMemoryReuseDependency {
    uint64_t predecessor_lease_id;
    uint64_t successor_lease_id;
    uint64_t dependency_id;
    uint64_t consumer_operation_index;
} ShadowSpillMemoryReuseDependency;

typedef struct ShadowSpillMemoryReplayResult {
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

    ShadowSpillMemoryReplayDecision *decisions;
    uint64_t decision_capacity;
    uint64_t decision_count;
    ShadowSpillMemoryReuseDependency *dependencies;
    uint64_t dependency_capacity;
    uint64_t dependency_result_count;
} ShadowSpillMemoryReplayResult;

/*
 * Replays one ordered script through the production MemoryPool policy. Input
 * and output buffers are borrowed for the call. Lease and dependency IDs are
 * dense zero-based indices bounded by their respective counts. The function
 * performs no backend operations and owns no storage after it returns.
 */
SHADOWSPILL_MEMORY_REPLAY_API uint32_t shadowspill_memory_replay_abi_version(
    void
);

SHADOWSPILL_MEMORY_REPLAY_API ShadowSpillMemoryReplayStatus
shadowspill_memory_replay_run(
    const ShadowSpillMemoryReplayProgram *program,
    ShadowSpillMemoryReplayResult *result
);

SHADOWSPILL_MEMORY_REPLAY_API const char *shadowspill_memory_replay_status_string(
    ShadowSpillMemoryReplayStatus status
);

#ifdef __cplusplus
}
#endif

#endif
