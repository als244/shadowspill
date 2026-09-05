#ifndef SHADOWSPILL_PYTORCH_TASKS_INTERNAL_H
#define SHADOWSPILL_PYTORCH_TASKS_INTERNAL_H

/*
 * The task boundary on the dispatching thread: the range a task or an
 * allocation scope opens, the boundary calls between which a planned task
 * runs, and the pre-task action batch. One task range is open per thread at
 * a time, and the failure record reads its label. The storage operators
 * include this header, so it carries nothing C++ cannot parse.
 */

#include <stdint.h>

#include <shadowspill/pytorch_adapter.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open the thread's task range: a profiler range under name, when a name is
   given, and the task's label for the failure record, which a scope leaves
   NULL. The caller has checked that none is open. */
void shadowspill_pytorch_task_range_begin(const char *name, const char *label);

/* Close the range if one is open, ending its profiler range. */
void shadowspill_pytorch_task_range_end(void);

int shadowspill_pytorch_task_range_active(void);

/* The open range's label, or NULL. */
const char *shadowspill_pytorch_task_range_label(void);

/* Between before and after: the compute stream waits for the ranges the
   boundary reuses. The C half of the _wait_task_allocations operator. */
ShadowSpillStatus shadowspill_pytorch_wait_task_allocations(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
);

#ifdef __cplusplus
}
#endif

#endif
