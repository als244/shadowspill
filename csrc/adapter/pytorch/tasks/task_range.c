#include "internal.h"

#include <stdio.h>

_Thread_local int task_range_active;
_Thread_local ShadowSpillProfilerRange task_range_id;
_Thread_local const char *active_task_label;

void shadowspill_pytorch_end_task_range(void) {
    if (task_range_active) {
        shadowspill_pytorch_profile_range_end(task_range_id);
        task_range_active = 0;
        task_range_id = 0;
    }
    active_task_label = NULL;
}

void shadowspill_pytorch_format_task_range_name(
    char *destination,
    size_t destination_bytes,
    const char *operation,
    const ShadowSpillTaskHandle *handle
) {
    const uint64_t task_id = shadowspill_task_id(handle);
    const char *label = shadowspill_task_trace_label(handle);
    if (label != NULL && label[0] != '\0') {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.%s.%s",
            operation,
            label
        );
    } else {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.%s.canonical_%llu",
            operation,
            (unsigned long long)task_id
        );
    }
}
