#include "shadowspill/pytorch_adapter.h"

#include "allocator_internal.h"

#include <new>

void *shadowspill_pytorch_cuda_malloc(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
) {
    void *const address = shadowspill_pytorch_cuda_malloc_impl(
        bytes, device_ordinal, stream
    );
    if (address == nullptr && bytes > 0) {
        /*
         * CUDAPluggableAllocator does not reject a null pointer before it
         * constructs a tensor.  Throw while still inside PyTorch's C++
         * allocator call so no opaque operator can consume invalid storage.
         * The C runtime has already latched the structured first failure; the
         * frontend task boundary reads and reports it after this exception.
         */
        throw std::bad_alloc();
    }
    return address;
}
