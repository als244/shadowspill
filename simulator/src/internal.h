#ifndef SHADOWSPILL_SIMULATOR_INTERNAL_H
#define SHADOWSPILL_SIMULATOR_INTERNAL_H

#include "shadowspill/simulator.h"

typedef struct ShadowSpillAliasState {
    uint8_t device_allocated;
    uint8_t device_ready;
    uint8_t host_allocated;
    uint8_t host_ready;
    uint8_t h2d_pending;
    uint8_t d2h_pending;
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
    int32_t *active_h2d;
    int32_t *active_d2h;
    uint32_t *h2d_sequence;
    uint32_t *d2h_sequence;
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

#endif
