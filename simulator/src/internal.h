#ifndef SHADOWSPILL_SIMULATOR_INTERNAL_H
#define SHADOWSPILL_SIMULATOR_INTERNAL_H

#include "shadowspill/simulator.h"

enum {
    SHADOWSPILL_TASK_UNLAUNCHED = 0,
    SHADOWSPILL_TASK_ACTIVE = 1,
    SHADOWSPILL_TASK_COMPLETE = 2,
    SHADOWSPILL_TRANSFER_UNUSED = 0,
    SHADOWSPILL_TRANSFER_QUEUED = 1,
    SHADOWSPILL_TRANSFER_ACTIVE = 2,
    SHADOWSPILL_TRANSFER_COMPLETE = 3,
};

typedef struct ShadowSpillAliasState {
    uint8_t device_allocated;
    uint8_t device_ready;
    uint8_t host_allocated;
    uint8_t host_ready;
    uint8_t fetch_pending;
    uint8_t evict_pending;
    uint64_t device_version;
    uint64_t host_version;
} ShadowSpillAliasState;

typedef struct ShadowSpillTaskState {
    uint8_t state;
    uint8_t ready_set;
    uint64_t ready_ns;
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t stall_mask;
} ShadowSpillTaskState;

typedef struct ShadowSpillTransferState {
    uint8_t state;
    uint8_t direction;
    uint32_t alias;
    uint32_t trigger_task;
    uint32_t device;
    uint32_t sequence;
    uint32_t stall_mask;
    uint64_t ready_ns;
    uint64_t start_ns;
    uint64_t end_ns;
} ShadowSpillTransferState;

typedef struct ShadowSpillSimulationWork {
    ShadowSpillAliasState *aliases;
    ShadowSpillTaskState *tasks;
    ShadowSpillTransferState *transfers;
    int32_t *active_fetch;
    int32_t *active_evict;
    uint32_t *fetch_sequence;
    uint32_t *evict_sequence;
    uint64_t *device_object_bytes;
    uint64_t *device_workspace_bytes;
    uint64_t *device_object_peaks;
    uint64_t *device_workspace_peaks;
    uint64_t *device_total_peaks;
    uint64_t host_bytes;
    uint64_t host_peak_bytes;
    uint64_t now_ns;
    uint32_t completed_tasks;
    uint32_t submitted_actions;
} ShadowSpillSimulationWork;

int shadowspill_add_overflow_u64(
    uint64_t left,
    uint64_t right,
    uint64_t *result
);

int shadowspill_validate_program(
    const ShadowSpillSimulationProgram *program
);

void shadowspill_initialize_result(ShadowSpillSimulationResult *result);

void shadowspill_set_error(
    ShadowSpillSimulationResult *result,
    ShadowSpillSimulationStatus status,
    const ShadowSpillSimulationWork *work,
    uint32_t task,
    uint32_t alias,
    uint32_t device
);

void shadowspill_set_capacity_error(
    ShadowSpillSimulationResult *result,
    ShadowSpillSimulationStatus status,
    const ShadowSpillSimulationWork *work,
    uint32_t task,
    uint32_t alias,
    uint32_t device,
    uint8_t location,
    uint64_t capacity,
    uint64_t used,
    uint64_t requested
);

int shadowspill_allocate_work(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
);

void shadowspill_free_work(ShadowSpillSimulationWork *work);

void shadowspill_update_peaks(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
);

int shadowspill_initialize_memory(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
);

int shadowspill_task_dependencies_complete(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t task,
    uint64_t *ready_ns
);

int shadowspill_inputs_ready(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t task
);

int shadowspill_try_launch_tasks(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
);

int shadowspill_complete_task(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t task
);

int shadowspill_submit_ready_actions(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
);

int shadowspill_try_start_transfers(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
);

int shadowspill_complete_transfer(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t device,
    uint8_t direction
);

int shadowspill_has_pending_work(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
);

uint64_t shadowspill_next_event_time(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
);

int shadowspill_complete_events(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
);

int shadowspill_report_deadlock(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
);

int shadowspill_check_final_residency(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
);

#endif
