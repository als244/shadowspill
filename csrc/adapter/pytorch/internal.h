#ifndef SHADOWSPILL_PYTORCH_ALLOCATOR_INTERNAL_H
#define SHADOWSPILL_PYTORCH_ALLOCATOR_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C runtime implementation wrapped by the exception-safe C++ callback. */
void *shadowspill_pytorch_cuda_malloc_impl(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
);

/*
 * Stamp one task-boundary timestamp. The frontend's task boundary is the whole
 * storage operation, not the runtime call inside it: acquiring a task also
 * rebinds its storages to the addresses the runtime returned, and publishing
 * one adopts, rebinds and dematerialises before the runtime hears about it.
 * Recording from there keeps before/invoke/after adjacent, with no frontend
 * work falling between them. Boundaries are before enter, before exit, after
 * enter, after exit.
 */
void shadowspill_pytorch_record_task_boundary(uint64_t task_id, uint8_t boundary);

/* Format the first allocator/runtime failure for the exception-safe wrapper. */
ShadowSpillStatus shadowspill_pytorch_cuda_malloc_failure_message(
    char *destination,
    size_t destination_bytes
);

#ifdef __cplusplus
}
#endif

#endif
