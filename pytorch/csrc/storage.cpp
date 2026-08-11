#include <shadowspill/pytorch_adapter.h>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <torch/library.h>

#include <nvtx3/nvToolsExt.h>

#include <cstdint>
#include <utility>

namespace {

struct CallerLease {
  uint64_t allocation_id;
};

void release_caller_lease(void* context) {
  auto* lease = static_cast<CallerLease*>(context);
  if (lease != nullptr) {
    (void)shadowspill_pytorch_release_caller_allocation(lease->allocation_id);
    delete lease;
  }
}

at::Tensor rebind_storage(
    const at::Tensor& tensor,
    int64_t target_address,
    int64_t object_id,
    int64_t generation
) {
  nvtxRangePushA("shadowspill.pytorch.storage_rebind");
  struct RangeGuard {
    ~RangeGuard() { nvtxRangePop(); }
  } range_guard;
  TORCH_CHECK(tensor.is_cuda(), "storage rebinding requires a CUDA tensor");
  TORCH_CHECK(target_address >= 0, "storage address must be nonnegative");
  TORCH_CHECK(object_id >= 0, "object ID must be nonnegative");
  TORCH_CHECK(generation >= 0, "allocation generation must be nonnegative");

  auto storage = tensor.storage();
  const c10::DataPtr& current = storage.data_ptr();
  const uint64_t current_address =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(current.get()));
  if (current_address != 0U) {
    const ShadowSpillRuntimeStatus status =
        shadowspill_pytorch_validate_object_binding(
            static_cast<uint64_t>(object_id),
            current_address,
            static_cast<uint64_t>(generation));
    TORCH_CHECK(
        status == SHADOWSPILL_RUNTIME_OK,
        "current storage does not match the planned object generation: ",
        shadowspill_runtime_status_string(status));
  }
  if (target_address != 0) {
    const ShadowSpillRuntimeStatus status =
        shadowspill_pytorch_validate_object_binding(
            static_cast<uint64_t>(object_id),
            static_cast<uint64_t>(target_address),
            static_cast<uint64_t>(generation));
    TORCH_CHECK(
        status == SHADOWSPILL_RUNTIME_OK,
        "target storage does not match the planned object generation: ",
        shadowspill_runtime_status_string(status));
  }
  TORCH_CHECK(
      current_address != 0U || target_address != 0,
      "storage is already dematerialized");

  c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
      reinterpret_cast<void*>(static_cast<uintptr_t>(target_address)),
      tensor.device()));
  prior.clear();
  return tensor;
}

at::Tensor transfer_storage_to_caller(
    const at::Tensor& tensor,
    int64_t object_id,
    int64_t generation,
    int64_t allocation_id
) {
  nvtxRangePushA("shadowspill.pytorch.caller_lease");
  struct RangeGuard {
    ~RangeGuard() { nvtxRangePop(); }
  } range_guard;
  TORCH_CHECK(tensor.is_cuda(), "caller transfer requires a CUDA tensor");
  TORCH_CHECK(object_id >= 0, "object ID must be nonnegative");
  TORCH_CHECK(generation >= 0, "generation must be nonnegative");
  TORCH_CHECK(allocation_id >= 0, "allocation ID must be nonnegative");

  auto storage = tensor.storage();
  const uint64_t address = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(storage.data_ptr().get()));
  TORCH_CHECK(address != 0U, "caller output storage is dematerialized");
  ShadowSpillRuntimeStatus status = shadowspill_pytorch_validate_object_binding(
      static_cast<uint64_t>(object_id),
      address,
      static_cast<uint64_t>(generation));
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "caller output binding is invalid: ",
      shadowspill_runtime_status_string(status));

  auto* lease = new CallerLease{static_cast<uint64_t>(allocation_id)};
  c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
      reinterpret_cast<void*>(static_cast<uintptr_t>(address)),
      lease,
      release_caller_lease,
      tensor.device()));
  prior.clear();

  ShadowSpillAllocation allocation = {};
  status = shadowspill_pytorch_transfer_output_to_caller(
      static_cast<uint64_t>(object_id), &allocation);
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "caller output transfer failed: ",
      shadowspill_runtime_status_string(status));
  TORCH_CHECK(
      allocation.allocation_id == static_cast<uint64_t>(allocation_id) &&
          allocation.generation == static_cast<uint64_t>(generation) &&
          allocation.pointer == reinterpret_cast<void*>(
              static_cast<uintptr_t>(address)),
      "caller output allocation changed during transfer");
  return tensor;
}

}  // namespace

TORCH_LIBRARY(shadowspill, library) {
  library.def(
      "_rebind_storage(Tensor(a!) tensor, int address, int object_id, "
      "int generation) -> Tensor(a!)");
  library.def(
      "_transfer_storage_to_caller(Tensor(a!) tensor, int object_id, "
      "int generation, int allocation_id) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl("_rebind_storage", TORCH_FN(rebind_storage));
  library.impl(
      "_transfer_storage_to_caller", TORCH_FN(transfer_storage_to_caller));
}
