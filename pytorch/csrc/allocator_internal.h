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

#ifdef __cplusplus
}
#endif

#endif
