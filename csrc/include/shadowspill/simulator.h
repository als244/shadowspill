#ifndef SHADOWSPILL_SIMULATOR_H
#define SHADOWSPILL_SIMULATOR_H

#include <stddef.h>
#include <stdint.h>
#include <shadowspill/shadowspill.h>

#if defined(_WIN32)
#define SHADOWSPILL_SIMULATOR_API __declspec(dllexport)
#else
#define SHADOWSPILL_SIMULATOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_SIMULATOR_ABI_VERSION SHADOWSPILL_ABI_VERSION
#define SHADOWSPILL_SIMULATOR_NO_INDEX UINT32_MAX

/* Simulation names for the shared statuses; see <shadowspill/status.h>. */
typedef ShadowSpillStatus ShadowSpillSimulationStatus;
#define SHADOWSPILL_SIMULATION_OK SHADOWSPILL_STATUS_OK
#define SHADOWSPILL_SIMULATION_INVALID_ARGUMENT SHADOWSPILL_STATUS_INVALID_ARGUMENT
#define SHADOWSPILL_SIMULATION_ALLOCATION_FAILURE SHADOWSPILL_STATUS_INTERNAL_FAILURE
#define SHADOWSPILL_SIMULATION_INITIAL_DEVICE_CAPACITY SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY
#define SHADOWSPILL_SIMULATION_INITIAL_HOST_CAPACITY SHADOWSPILL_STATUS_INITIAL_HOST_CAPACITY
#define SHADOWSPILL_SIMULATION_TASK_INPUT_DEADLOCK SHADOWSPILL_STATUS_TASK_INPUT_DEADLOCK
#define SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY
#define SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY
#define SHADOWSPILL_SIMULATION_OFFLOAD_HOST_CAPACITY SHADOWSPILL_STATUS_OFFLOAD_HOST_CAPACITY
#define SHADOWSPILL_SIMULATION_TRANSFER_DEADLOCK SHADOWSPILL_STATUS_TRANSFER_DEADLOCK
#define SHADOWSPILL_SIMULATION_INVALID_RELEASE SHADOWSPILL_STATUS_INVALID_RELEASE
#define SHADOWSPILL_SIMULATION_RELEASE_TRANSFER_CONFLICT SHADOWSPILL_STATUS_RELEASE_TRANSFER_CONFLICT
#define SHADOWSPILL_SIMULATION_INVALID_OFFLOAD SHADOWSPILL_STATUS_INVALID_OFFLOAD
#define SHADOWSPILL_SIMULATION_INVALID_PREFETCH SHADOWSPILL_STATUS_INVALID_PREFETCH
#define SHADOWSPILL_SIMULATION_FINAL_RESIDENCY SHADOWSPILL_STATUS_FINAL_RESIDENCY
#define SHADOWSPILL_SIMULATION_INTERNAL_ERROR SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR

typedef enum ShadowSpillMemoryLocation {
    SHADOWSPILL_MEMORY_DEVICE = 0,
    SHADOWSPILL_MEMORY_HOST = 1,
} ShadowSpillMemoryLocation;

typedef enum ShadowSpillMemoryActionKind {
    SHADOWSPILL_MEMORY_RELEASE = 0,
    SHADOWSPILL_MEMORY_OFFLOAD = 1,
    SHADOWSPILL_MEMORY_PREFETCH = 2,
} ShadowSpillMemoryActionKind;

typedef enum ShadowSpillTransferDirection {
    SHADOWSPILL_TRANSFER_FETCH = 0,
    SHADOWSPILL_TRANSFER_EVICT = 1,
} ShadowSpillTransferDirection;

enum {
    SHADOWSPILL_STALL_INPUT_RESIDENCY = 1U << 0U,
    SHADOWSPILL_STALL_DEVICE_CAPACITY = 1U << 1U,
    SHADOWSPILL_STALL_SOURCE_READINESS = 1U << 2U,
    SHADOWSPILL_STALL_HOST_CAPACITY = 1U << 3U,
    SHADOWSPILL_STALL_MEMORY_REUSE = 1U << 4U,
};

typedef struct ShadowSpillSimulationDevice {
    uint64_t capacity_bytes;
    uint64_t fetch_bandwidth_bytes_per_second;
    uint64_t evict_bandwidth_bytes_per_second;
    uint64_t fetch_latency_ns;
    uint64_t evict_latency_ns;
} ShadowSpillSimulationDevice;

typedef struct ShadowSpillSimulationProgram {
    uint32_t abi_version;
    uint32_t device_count;
    uint32_t alias_count;
    uint32_t task_count;
    uint32_t action_count;
    uint32_t initial_count;
    uint32_t final_count;
    uint32_t dependency_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t mutation_count;
    uint32_t reuse_dependency_count;
    uint32_t use_admission_accounting;
    uint64_t host_capacity_bytes;

    const ShadowSpillSimulationDevice *devices;
    const uint32_t *alias_device;
    const uint64_t *alias_size_bytes;
    const uint64_t *alias_initial_version;
    const uint8_t *alias_retain_spill_copy;

    const uint32_t *task_device;
    const uint8_t *task_resource_kind;
    const uint32_t *task_resource_lane;
    const uint64_t *task_runtime_ns;
    const uint64_t *task_workspace_bytes;
    const int64_t *task_start_physical_deltas;
    const int64_t *task_completion_physical_deltas;
    const uint32_t *dependency_offsets;
    const uint32_t *dependencies;
    const uint32_t *input_offsets;
    const uint32_t *input_aliases;
    const uint32_t *output_offsets;
    const uint32_t *output_aliases;
    const uint32_t *mutation_offsets;
    const uint32_t *mutation_aliases;
    const uint64_t *mutation_version_deltas;

    const uint32_t *action_trigger_tasks;
    const uint32_t *action_aliases;
    const uint8_t *action_kinds;
    const int64_t *action_trigger_physical_deltas;
    const int64_t *action_completion_physical_deltas;
    const uint32_t *initial_aliases;
    const uint8_t *initial_locations;
    const uint64_t *initial_physical_bytes;
    const uint32_t *final_aliases;
    const uint8_t *final_locations;
    const uint32_t *reuse_predecessor_actions;
    const uint32_t *reuse_successor_tasks;
    const uint32_t *reuse_successor_actions;
} ShadowSpillSimulationProgram;

typedef struct ShadowSpillTaskInterval {
    uint32_t task;
    uint64_t ready_ns;
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t workspace_bytes;
    uint32_t stall_mask;
} ShadowSpillTaskInterval;

typedef struct ShadowSpillTransferInterval {
    uint32_t alias;
    uint32_t trigger_task;
    uint32_t device;
    uint8_t direction;
    uint32_t sequence;
    uint64_t ready_ns;
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t bytes;
    uint32_t stall_mask;
} ShadowSpillTransferInterval;

typedef struct ShadowSpillDevicePeak {
    uint64_t object_bytes;
    uint64_t workspace_bytes;
    uint64_t total_bytes;
} ShadowSpillDevicePeak;

typedef struct ShadowSpillSimulationResult {
    uint32_t status;
    uint32_t error_task;
    uint32_t error_alias;
    uint32_t error_device;
    uint8_t error_location;
    uint64_t error_time_ns;
    uint64_t error_capacity_bytes;
    uint64_t error_used_bytes;
    uint64_t error_requested_bytes;
    uint64_t makespan_ns;
    uint64_t host_peak_bytes;

    ShadowSpillTaskInterval *task_intervals;
    uint32_t task_interval_capacity;
    uint32_t task_interval_count;
    ShadowSpillTransferInterval *transfer_intervals;
    uint32_t transfer_interval_capacity;
    uint32_t transfer_interval_count;
    ShadowSpillDevicePeak *device_peaks;
    uint32_t device_peak_capacity;
} ShadowSpillSimulationResult;

/*
 * All pointers in `program` and output buffers in `result` are borrowed for
 * the duration of this call. The function owns no external storage, performs
 * no I/O, uses no global mutable state, and is safe to call concurrently with
 * distinct result buffers. On failure, `status` and diagnostic fields are
 * populated and all caller-owned buffers remain caller-owned.
 */
SHADOWSPILL_SIMULATOR_API ShadowSpillSimulationStatus shadowspill_simulate(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationResult *result
);

SHADOWSPILL_SIMULATOR_API uint32_t shadowspill_simulator_abi_version(void);

SHADOWSPILL_SIMULATOR_API const char *shadowspill_simulation_status_string(
    ShadowSpillSimulationStatus status
);

#ifdef __cplusplus
}
#endif

#endif
