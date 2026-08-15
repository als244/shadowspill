#include "shadowspill/pytorch_adapter.h"

#include "allocator_internal.h"

#ifdef SHADOWSPILL_PYTORCH_STORAGE_ADAPTER
#include <c10/util/Exception.h>
#endif

#include <stdexcept>

namespace {

[[noreturn]] void throw_allocator_failure() {
    char message[4096] = {0};
    const ShadowSpillRuntimeStatus status =
        shadowspill_pytorch_cuda_malloc_failure_message(
            message, sizeof(message)
        );
#ifdef SHADOWSPILL_PYTORCH_STORAGE_ADAPTER
    if (status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY ||
        status == SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        C10_THROW_ERROR(OutOfMemoryError, message);
    }
    C10_THROW_ERROR(Error, message);
#else
    throw std::runtime_error(message);
#endif
}

}  // namespace

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
         * The exception contains the C runtime's immutable first-failure
         * snapshot, including the current execution-task label.
         */
        throw_allocator_failure();
    }
    return address;
}
