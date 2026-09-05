#include "internal.h"

#include "../internal.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void append_failure_message(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    const char *format,
    ...
) {
    if (destination == NULL || destination_bytes == 0U ||
        *offset >= destination_bytes) {
        return;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = vsnprintf(
        destination + *offset,
        destination_bytes - *offset,
        format,
        arguments
    );
    va_end(arguments);
    if (written < 0) {
        return;
    }
    const size_t available = destination_bytes - *offset;
    *offset += (size_t)written < available ? (size_t)written : available - 1U;
}

static void append_failure_task(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    uint64_t task_id
) {
    const uint64_t profiling_base = SHADOWSPILL_PYTORCH_PROFILING_SCOPE_BASE;
    const uint64_t initial_actions_base =
        SHADOWSPILL_PYTORCH_INITIAL_ACTIONS_TASK_ID;
    if (task_id >= profiling_base) {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "planning_task: structural_profile_%06llu\n",
            (unsigned long long)(task_id - profiling_base)
        );
        return;
    }
    if (task_id >= initial_actions_base) {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "runtime_scope: initial_actions.invocation_%06llu\n",
            (unsigned long long)(task_id - initial_actions_base)
        );
        return;
    }
    char label[SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U] = {0};
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.failure_task_label[0] != '\0') {
        (void)snprintf(
            label, sizeof(label), "%s", adapter.failure_task_label
        );
    }
    pthread_mutex_unlock(&adapter.mutex);
    if (label[0] == '\0') {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "canonical_task: task_%06llu\n",
            (unsigned long long)task_id
        );
        return;
    }
    char *const separator = strchr(label, '.');
    if (separator != NULL) {
        *separator = '\0';
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "execution_task: %s\nsemantic_task: %s\n",
            label,
            separator + 1
        );
    } else {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "execution_task: %s\n",
            label
        );
    }
    append_failure_message(
        destination,
        destination_bytes,
        offset,
        "canonical_task: task_%06llu\n",
        (unsigned long long)task_id
    );
}

/* Bytes as a person reads them. Reports are read by people deciding what to
 * change, and "16273899520" does not tell them it exceeds a 16 GiB budget. */
static void append_bytes(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    const char *label,
    uint64_t value
) {
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    size_t unit = 0U;
    double scaled = (double)value;
    while (scaled >= 1024.0 && unit + 1U < sizeof(units) / sizeof(units[0])) {
        scaled /= 1024.0;
        ++unit;
    }
    if (unit == 0U) {
        append_failure_message(
            destination, destination_bytes, offset, "%s: %llu B\n",
            label, (unsigned long long)value
        );
        return;
    }
    append_failure_message(
        destination, destination_bytes, offset, "%s: %.2f %s (%llu bytes)\n",
        label, scaled, units[unit], (unsigned long long)value
    );
}

static const char *allocation_operation_name(uint8_t operation) {
    if (operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE) {
        return "allocate";
    }
    if (operation == SHADOWSPILL_TASK_ALLOCATION_FREE) {
        return "free";
    }
    if (operation == UINT8_MAX) {
        return "end_of_task";
    }
    return "unknown";
}

/* The failure the report describes: the call that just failed, not the
   first failure the runtime ever saw. A recovered failure latched earlier
   says nothing about this one, and reporting its operands beside this
   call's makes every number in the message unattributable. Says whether an
   earlier first failure is what this call is really failing on. */
static ShadowSpillStatus reported_failure(
    ShadowSpillPytorchAdapterFailure *failure,
    int *superseded_by_first_failure
) {
    ShadowSpillStatus status = shadowspill_pytorch_allocator_failure(failure);
    pthread_mutex_lock(&adapter.mutex);
    const int have_recent = adapter.recent_valid != 0U;
    const ShadowSpillPytorchAdapterFailure recent = adapter.recent;
    pthread_mutex_unlock(&adapter.mutex);
    *superseded_by_first_failure = 0;
    if (have_recent) {
        *superseded_by_first_failure =
            failure->status != SHADOWSPILL_STATUS_OK &&
            failure->status != recent.status;
        status = (ShadowSpillStatus)recent.status;
        failure->device_ordinal = recent.device_ordinal;
        failure->requested_bytes = recent.requested_bytes;
    }
    return status;
}

/* "Out of memory" means very different things for the device execution
   pool and the spill pool, and an internal failure belongs to neither. */
static const char *pool_name(uint32_t pool_id) {
    if (pool_id == UINT32_MAX) {
        return "no pool";
    }
    return pool_id == shadowspill_pytorch_allocator_pool_id()
        ? "execution pool"
        : "spill pool";
}

static void append_headline(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    ShadowSpillStatus status,
    const ShadowSpillPytorchAdapterFailure *failure
) {
    const char *pool = pool_name(failure->runtime.pool_id);
    if (status == SHADOWSPILL_STATUS_NO_PROGRESS) {
        append_failure_message(
            destination, destination_bytes, offset,
            "ShadowSpill out of memory in the %s (device %d), with nothing "
            "left to release\n",
            pool, failure->device_ordinal
        );
    } else if (status == SHADOWSPILL_STATUS_OUT_OF_MEMORY) {
        append_failure_message(
            destination, destination_bytes, offset,
            "ShadowSpill out of memory in the %s (device %d)\n",
            pool, failure->device_ordinal
        );
    } else {
        append_failure_message(
            destination, destination_bytes, offset,
            "ShadowSpill %s (device %d)\nstatus: %u (%s)\n",
            shadowspill_status_string(status),
            failure->device_ordinal,
            (unsigned int)status,
            shadowspill_status_string(status)
        );
    }
}

static void append_envelope(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    const ShadowSpillRuntimeFailure *runtime
) {
    append_failure_message(
        destination,
        destination_bytes,
        offset,
        "reason: TASK_ALLOCATION_ENVELOPE_EXCEEDED\n"
        "task_live_requested: %llu\n"
        "task_live_charged: %llu\n"
        "task_live_requested_limit: %llu\n"
        "task_live_charged_limit: %llu\n"
        "task_maximum_requested_allocation: %llu\n"
        "task_maximum_charged_allocation: %llu\n",
        (unsigned long long)runtime->task_live_requested_bytes,
        (unsigned long long)runtime->task_live_charged_bytes,
        (unsigned long long)runtime->task_live_requested_limit_bytes,
        (unsigned long long)runtime->task_live_charged_limit_bytes,
        (unsigned long long)
            runtime->task_maximum_requested_allocation_bytes,
        (unsigned long long)
            runtime->task_maximum_charged_allocation_bytes
    );
}

static void append_contract_mismatch(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    const ShadowSpillRuntimeFailure *runtime
) {
    append_failure_message(
        destination,
        destination_bytes,
        offset,
        "reason: TASK_ALLOCATION_CONTRACT_MISMATCH\n"
        "task_allocation_operation_index: %llu\n"
        "expected_operation: %s\nactual_operation: %s\n"
        "expected_ordinal: %llu\nactual_ordinal: %llu\n"
        "expected_requested: %llu\nactual_requested: %llu\n"
        "expected_charged: %llu\nactual_charged: %llu\n"
        "expected_alignment: %llu\nactual_alignment: %llu\n",
        (unsigned long long)runtime->task_allocation_operation_index,
        allocation_operation_name(
            runtime->task_allocation_expected_operation
        ),
        allocation_operation_name(
            runtime->task_allocation_actual_operation
        ),
        (unsigned long long)runtime->task_allocation_expected_ordinal,
        (unsigned long long)runtime->task_allocation_actual_ordinal,
        (unsigned long long)
            runtime->task_allocation_expected_requested_bytes,
        (unsigned long long)
            runtime->task_allocation_actual_requested_bytes,
        (unsigned long long)
            runtime->task_allocation_expected_charged_bytes,
        (unsigned long long)
            runtime->task_allocation_actual_charged_bytes,
        (unsigned long long)
            runtime->task_allocation_expected_alignment_bytes,
        (unsigned long long)
            runtime->task_allocation_actual_alignment_bytes
    );
}

ShadowSpillStatus shadowspill_pytorch_backend_malloc_failure_message(
    char *destination,
    size_t destination_bytes
) {
    if (destination == NULL || destination_bytes == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    destination[0] = '\0';
    ShadowSpillPytorchAdapterFailure failure = {0};
    int superseded = 0;
    const ShadowSpillStatus status = reported_failure(&failure, &superseded);
    size_t offset = 0U;
    if (status == SHADOWSPILL_STATUS_OK) {
        /* Nothing recorded this call. Say so rather than inventing a status:
         * a made-up diagnosis is worse than an admitted absence. */
        append_failure_message(
            destination,
            destination_bytes,
            &offset,
            "ShadowSpill allocator returned no memory and recorded no reason\n"
        );
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    const ShadowSpillRuntimeFailure *runtime = &failure.runtime;
    append_headline(destination, destination_bytes, &offset, status, &failure);
    if (runtime->reason != SHADOWSPILL_FAILURE_REASON_UNSPECIFIED) {
        append_failure_message(
            destination, destination_bytes, &offset, "reason: %s\n",
            shadowspill_failure_reason_string(
                (ShadowSpillFailureReason)runtime->reason
            )
        );
    }
    if (superseded) {
        append_failure_message(
            destination, destination_bytes, &offset,
            "note: an earlier failure (%s) had already stopped this runtime; "
            "later calls fail because of it, not on their own\n",
            shadowspill_status_string((ShadowSpillStatus)failure.status)
        );
    }
    if (runtime->task_id != UINT64_MAX) {
        append_failure_task(
            destination, destination_bytes, &offset, runtime->task_id
        );
    }
    append_bytes(
        destination, destination_bytes, &offset, "requested",
        failure.requested_bytes != 0U
            ? failure.requested_bytes
            : runtime->requested_bytes
    );
    if (runtime->free_bytes != 0U || runtime->largest_free_range_bytes != 0U) {
        append_bytes(destination, destination_bytes, &offset, "pool free",
            runtime->free_bytes);
        append_bytes(destination, destination_bytes, &offset,
            "largest free range", runtime->largest_free_range_bytes);
    }
    if (status == SHADOWSPILL_STATUS_TASK_ALLOCATION_ENVELOPE_EXCEEDED) {
        append_envelope(destination, destination_bytes, &offset, runtime);
    } else if (status == SHADOWSPILL_STATUS_TASK_ALLOCATION_CONTRACT_MISMATCH) {
        append_contract_mismatch(
            destination, destination_bytes, &offset, runtime
        );
    }
    return status;
}
