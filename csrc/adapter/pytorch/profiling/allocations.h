#ifndef SHADOWSPILL_PYTORCH_ALLOCATION_OBSERVER_H
#define SHADOWSPILL_PYTORCH_ALLOCATION_OBSERVER_H

#include <shadowspill/pytorch_adapter.h>

/* Observe the ordinary caching allocator without installing an allocator or
   opening runtime pools. Pointer-bearing records are process-local evidence. */
typedef struct ShadowSpillPytorchAllocationObservation {
    uint64_t address;
    uint64_t requested_bytes;
    uint64_t stream;
    uint64_t pool_first;
    uint64_t pool_second;
    int32_t device;
    int32_t action;
} ShadowSpillPytorchAllocationObservation;

#ifdef __cplusplus
extern "C" {
#endif

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_observe_allocations_begin(int32_t device, uint64_t capacity);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_observe_allocations_end(
    ShadowSpillPytorchAllocationObservation *records,
    uint64_t capacity,
    uint64_t *count,
    uint32_t *overflowed);

#ifdef __cplusplus
}
#endif
#endif
