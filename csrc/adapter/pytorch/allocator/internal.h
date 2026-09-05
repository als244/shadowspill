#ifndef SHADOWSPILL_PYTORCH_ALLOCATOR_INTERNAL_H
#define SHADOWSPILL_PYTORCH_ALLOCATOR_INTERNAL_H

/*
 * The three calls PyTorch's pluggable allocator makes -- malloc, free,
 * record_stream -- and the query that classifies a pointer. This is the hot
 * path and the only code that runs inside a PyTorch call; the C++ wrapper
 * includes this header, so it carries nothing C++ cannot parse.
 */

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/pytorch_adapter.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C allocation behind shadowspill_pytorch_backend_malloc: returns NULL
   and latches the reason where the C++ wrapper turns that into a throw. */
void *shadowspill_pytorch_backend_malloc_impl(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
