#ifndef SHADOWSPILL_PYTORCH_FAILURE_INTERNAL_H
#define SHADOWSPILL_PYTORCH_FAILURE_INTERNAL_H

/*
 * What is latched when a call fails, and how it is told. latch.c records --
 * the first failure, which stopped the runtime, and the most recent, which
 * explains the call that just failed -- and message.c renders the report a
 * person reads. The C++ wrapper includes this header, so it carries nothing
 * C++ cannot parse.
 */

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/pytorch_adapter.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forget the first failure; the caller holds the adapter's lock. */
void shadowspill_pytorch_failure_clear_locked(int32_t device_ordinal);

void shadowspill_pytorch_latch_failure(
    ShadowSpillStatus status,
    int32_t device_ordinal,
    const void *address,
    uint64_t requested_bytes
);

/* The report for the allocator call that just failed. */
ShadowSpillStatus shadowspill_pytorch_backend_malloc_failure_message(
    char *destination,
    size_t destination_bytes
);

#ifdef __cplusplus
}
#endif

#endif
