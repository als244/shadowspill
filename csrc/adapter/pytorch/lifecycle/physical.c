#include "internal.h"
#include "../failure/internal.h"

#include <pthread.h>
#include <stdio.h>

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
    const uint8_t within_budget =
        memory.process_bytes <= admission.device_budget_bytes &&
        external <= admission.external_headroom_bytes;
    const uint8_t report_only = admission.reject_overbudget == 0U;
    ShadowSpillStatus status = within_budget || report_only
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_PLAN_VIOLATION;
    pthread_mutex_lock(&adapter.mutex);
    const uint8_t report_growth = report_only && !within_budget &&
        (memory.process_bytes > adapter.peak_process_physical_bytes ||
         external > adapter.observed_external_high_water_bytes);
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
    if (report_growth) {
        (void)fprintf(
            stderr,
            "ShadowSpill: physical use is %llu bytes against a declared budget "
            "of %llu, with %llu bytes outside the baseline and allocator pool. "
            "reject_overbudget is false, so this is reported and not enforced.\n",
            (unsigned long long)memory.process_bytes,
            (unsigned long long)admission.device_budget_bytes,
            (unsigned long long)external
        );
        (void)fflush(stderr);
    }
    return status;
}

ShadowSpillStatus shadowspill_pytorch_seal_physical_budget(
    uint64_t required_external_headroom_bytes,
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
    /* The allowance sizes the pool; only this flag controls enforcement. */
    const uint64_t reported_requirement = !adapter.admission.reject_overbudget &&
        required_external_headroom_bytes > adapter.admission.external_headroom_bytes
        ? required_external_headroom_bytes
        : 0U;
    if (!adapter.admission.reject_overbudget) {
        adapter.physical_budget_sealed = 1U;
    } else if (required_external_headroom_bytes >
        adapter.admission.external_headroom_bytes) {
        status = SHADOWSPILL_STATUS_PLAN_VIOLATION;
        shadowspill_pytorch_failure_latch_physical_locked(
            status,
            required_external_headroom_bytes,
            adapter.admission.external_headroom_bytes
        );
    } else {
        adapter.physical_budget_sealed = 1U;
    }
    pthread_mutex_unlock(&adapter.mutex);
    if (reported_requirement != 0U) {
        (void)fprintf(
            stderr,
            "ShadowSpill: measured external memory needs a headroom of %llu "
            "bytes; reject_overbudget is false, so this is reported and not "
            "enforced.\n",
            (unsigned long long)reported_requirement
        );
        (void)fflush(stderr);
    }
    return status;
}
