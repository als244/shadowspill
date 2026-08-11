#ifndef SHADOWSPILL_SIMULATOR_H
#define SHADOWSPILL_SIMULATOR_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define SHADOWSPILL_SIMULATOR_API __declspec(dllexport)
#else
#define SHADOWSPILL_SIMULATOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_SIMULATOR_ABI_VERSION 1U
#define SHADOWSPILL_SIMULATOR_NO_INDEX UINT32_MAX

typedef enum ShadowSpillSimulationStatus {
    SHADOWSPILL_SIMULATION_OK = 0,
    SHADOWSPILL_SIMULATION_INVALID_ARGUMENT = 1,
    SHADOWSPILL_SIMULATION_ALLOCATION_FAILURE = 2,
    SHADOWSPILL_SIMULATION_INITIAL_DEVICE_CAPACITY = 3,
    SHADOWSPILL_SIMULATION_INITIAL_HOST_CAPACITY = 4,
    SHADOWSPILL_SIMULATION_TASK_INPUT_DEADLOCK = 5,
    SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY = 6,
    SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY = 7,
    SHADOWSPILL_SIMULATION_OFFLOAD_HOST_CAPACITY = 8,
    SHADOWSPILL_SIMULATION_TRANSFER_DEADLOCK = 9,
    SHADOWSPILL_SIMULATION_INVALID_RELEASE = 10,
    SHADOWSPILL_SIMULATION_RELEASE_TRANSFER_CONFLICT = 11,
    SHADOWSPILL_SIMULATION_INVALID_OFFLOAD = 12,
    SHADOWSPILL_SIMULATION_INVALID_PREFETCH = 13,
    SHADOWSPILL_SIMULATION_FINAL_RESIDENCY = 14,
    SHADOWSPILL_SIMULATION_INTERNAL_ERROR = 15,
} ShadowSpillSimulationStatus;

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
    SHADOWSPILL_TRANSFER_HOST_TO_DEVICE = 0,
    SHADOWSPILL_TRANSFER_DEVICE_TO_HOST = 1,
} ShadowSpillTransferDirection;

enum {
    SHADOWSPILL_STALL_INPUT_RESIDENCY = 1U << 0U,
    SHADOWSPILL_STALL_DEVICE_CAPACITY = 1U << 1U,
    SHADOWSPILL_STALL_SOURCE_READINESS = 1U << 2U,
    SHADOWSPILL_STALL_HOST_CAPACITY = 1U << 3U,
};

typedef struct ShadowSpillSimulationDevice {
    uint64_t capacity_bytes;
    uint64_t h2d_bandwidth_bytes_per_second;
    uint64_t d2h_bandwidth_bytes_per_second;
    uint64_t h2d_latency_ns;
    uint64_t d2h_latency_ns;
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
    uint64_t host_capacity_bytes;

    const ShadowSpillSimulationDevice *devices;
    const uint32_t *alias_device;
    const uint64_t *alias_size_bytes;
    const uint64_t *alias_initial_version;
    const uint8_t *alias_retain_host_backing;

    const uint32_t *task_device;
    const uint8_t *task_resource_kind;
    const uint32_t *task_resource_lane;
    const uint64_t *task_runtime_ns;
    const uint64_t *task_workspace_bytes;
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
    const uint32_t *initial_aliases;
    const uint8_t *initial_locations;
    const uint32_t *final_aliases;
    const uint8_t *final_locations;
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
