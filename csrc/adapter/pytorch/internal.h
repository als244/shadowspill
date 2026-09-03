#ifndef SHADOWSPILL_PYTORCH_ALLOCATOR_INTERNAL_H
#define SHADOWSPILL_PYTORCH_ALLOCATOR_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C runtime implementation wrapped by the exception-safe C++ callback. */
void *shadowspill_pytorch_backend_malloc_impl(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
);

/* Format the first allocator/runtime failure for the exception-safe wrapper. */
ShadowSpillStatus shadowspill_pytorch_backend_malloc_failure_message(
    char *destination,
    size_t destination_bytes
);

#ifdef __cplusplus
}
#endif

#endif
