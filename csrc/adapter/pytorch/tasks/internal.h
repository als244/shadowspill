#ifndef SHADOWSPILL_PYTORCH_TASKS_INTERNAL_H
#define SHADOWSPILL_PYTORCH_TASKS_INTERNAL_H

/*
 * The task boundary on the dispatching thread: the range a task or an
 * allocation scope opens, the boundary calls between which a planned task
 * runs, and the pre-task action batch. One task range is open per thread at
 * a time, and the failure record reads its label.
 */

#include "../internal.h"

extern _Thread_local int task_range_active;
extern _Thread_local ShadowSpillProfilerRange task_range_id;
extern _Thread_local const char *active_task_label;

void shadowspill_pytorch_end_task_range(void);

void shadowspill_pytorch_format_task_range_name(
    char *destination,
    size_t destination_bytes,
    const char *operation,
    const ShadowSpillTaskHandle *handle
);

#endif
