#include "internal.h"
#include <c10/cuda/CUDAStream.h>

namespace shadowspill_pytorch_storage {

struct CallerLease {
  uint64_t allocation_id;
  int32_t device_ordinal;
};

void release_caller_lease(void* problem) {
  auto* lease = static_cast<CallerLease*>(problem);
  if (lease != nullptr) {
    const c10::cuda::CUDAStream stream =
        c10::cuda::getCurrentCUDAStream(lease->device_ordinal);
    (void)shadowspill_pytorch_release_caller_allocation(
        lease->allocation_id,
        reinterpret_cast<uintptr_t>(stream.stream()));
    delete lease;
  }
}

at::Tensor transfer_acquired_storage_to_caller(
    const at::Tensor& tensor,
    int64_t acquisition_handle,
    int64_t object_ordinal,
    int64_t generation,
    int64_t allocation_id
) {
  RangeGuard range_guard("shadowspill.pytorch.caller_lease");
  TORCH_CHECK(tensor.is_cuda(), "caller transfer requires a CUDA tensor");
  TORCH_CHECK(acquisition_handle > 0, "acquisition handle must be positive");
  TORCH_CHECK(object_ordinal >= 0, "object ordinal must be nonnegative");
  TORCH_CHECK(generation >= 0, "generation must be nonnegative");
  TORCH_CHECK(allocation_id >= 0, "allocation ID must be nonnegative");

  auto storage = tensor.storage();
  const uint64_t address = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(storage.data_ptr().get()));
  TORCH_CHECK(address != 0U, "caller output storage is dematerialized");
  ShadowSpillAllocation allocation = {};
  const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(
      static_cast<c10::DeviceIndex>(tensor.get_device()));
  const ShadowSpillStatus status =
      shadowspill_pytorch_transfer_acquired_object_to_caller(
      static_cast<uintptr_t>(acquisition_handle),
      static_cast<uint32_t>(object_ordinal),
      reinterpret_cast<uintptr_t>(stream.stream()),
      address,
      static_cast<uint64_t>(generation),
      static_cast<uint64_t>(allocation_id),
      &allocation);
  TORCH_CHECK(
      status == SHADOWSPILL_STATUS_OK,
      "caller output transfer failed: ",
      shadowspill_status_string(status));
  TORCH_CHECK(
      allocation.allocation_id == static_cast<uint64_t>(allocation_id) &&
          allocation.generation == static_cast<uint64_t>(generation) &&
          allocation.pointer == reinterpret_cast<void*>(
              static_cast<uintptr_t>(address)),
      "caller output allocation changed during transfer");
  auto* lease = new CallerLease{
      static_cast<uint64_t>(allocation_id), tensor.get_device()};
  c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
      reinterpret_cast<void*>(static_cast<uintptr_t>(address)),
      lease,
      release_caller_lease,
      tensor.device()));
  prior.clear();
  return tensor;
}

}  // namespace shadowspill_pytorch_storage

using namespace shadowspill_pytorch_storage;

TORCH_LIBRARY_FRAGMENT(shadowspill, library) {
  library.def(
      "_transfer_acquired_storage_to_caller(Tensor(a!) tensor, int "
      "acquisition_handle, int object_ordinal, int generation, int "
      "allocation_id) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl(
      "_transfer_acquired_storage_to_caller",
      TORCH_FN(transfer_acquired_storage_to_caller));
}
