#include "internal.h"
#include "../failure/internal.h"

#include <pthread.h>

ShadowSpillStatus shadowspill_pytorch_physical_admission(
    ShadowSpillPytorchPhysicalAdmission *admission
) {
    if (admission == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime == NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_CLOSED;
    }
    *admission = adapter.admission;
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_pytorch_physical_memory(
    ShadowSpillBackendPhysicalMemory *memory
) {
    if (memory == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    const ShadowSpillBackend backend = adapter.backend.table;
    pthread_mutex_unlock(&adapter.mutex);
    if (backend.state == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    return backend.physical_memory(backend.state, memory) == 0
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_BACKEND_FAILURE;
}

ShadowSpillStatus shadowspill_pytorch_check_physical_budget(void) {
    ShadowSpillBackendPhysicalMemory memory = {0};
    pthread_mutex_lock(&adapter.mutex);
    const ShadowSpillBackend backend = adapter.backend.table;
    ShadowSpillPytorchPhysicalAdmission admission = adapter.admission;
    pthread_mutex_unlock(&adapter.mutex);
    if (backend.state == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    if (backend.physical_memory(backend.state, &memory) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    uint64_t base = admission.baseline_bytes + admission.allocator_pool_bytes;
    uint64_t external = memory.process_bytes > base
        ? memory.process_bytes - base
        : 0U;
    ShadowSpillStatus status =
        memory.process_bytes <= admission.device_budget_bytes &&
            external <= admission.provider_headroom_bytes
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_PLAN_VIOLATION;
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.physical_checks;
    if (memory.process_bytes > adapter.peak_process_physical_bytes) {
        adapter.peak_process_physical_bytes = memory.process_bytes;
    }
    if (external > adapter.observed_external_high_water_bytes) {
        adapter.observed_external_high_water_bytes = external;
    }
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_pytorch_failure_latch_physical_locked(
            status,
            memory.process_bytes,
            admission.device_budget_bytes > memory.process_bytes
                ? admission.device_budget_bytes - memory.process_bytes
                : 0U
        );
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_seal_physical_budget(
    uint64_t required_provider_headroom_bytes,
    uint64_t runtime_record_reserve
) {
    pthread_mutex_lock(&adapter.mutex);
    const ShadowSpillBackend backend = adapter.backend.table;
    ShadowSpillRuntime *runtime = adapter.runtime;
    pthread_mutex_unlock(&adapter.mutex);
    if (backend.state == NULL || runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus reserve_status =
        shadowspill_runtime_reserve_event_leases(runtime, runtime_record_reserve);
    if (reserve_status != SHADOWSPILL_STATUS_OK) {
        return reserve_status;
    }
    reserve_status = shadowspill_runtime_reserve_retirement_records(
        runtime, runtime_record_reserve
    );
    if (reserve_status != SHADOWSPILL_STATUS_OK) {
        return reserve_status;
    }
    for (uint32_t pool_id = 0U;
         pool_id < adapter.admission.pool_count;
         ++pool_id) {
        reserve_status = shadowspill_runtime_reserve_memory_lease_records(
            runtime, pool_id, runtime_record_reserve
        );
        if (reserve_status != SHADOWSPILL_STATUS_OK) {
            return reserve_status;
        }
    }
    ShadowSpillStatus status =
        shadowspill_pytorch_check_physical_budget();
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (required_provider_headroom_bytes >
        adapter.admission.provider_headroom_bytes) {
        status = SHADOWSPILL_STATUS_PLAN_VIOLATION;
        shadowspill_pytorch_failure_latch_physical_locked(
            status,
            required_provider_headroom_bytes,
            adapter.admission.provider_headroom_bytes
        );
    } else {
        adapter.physical_budget_sealed = 1U;
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
}
