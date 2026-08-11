#include <shadowspill/pytorch_adapter.h>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <torch/library.h>

#include <nvtx3/nvToolsExt.h>

#include <cstdint>
#include <utility>

namespace {

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

}  // namespace

TORCH_LIBRARY(shadowspill, library) {
  library.def(
      "_rebind_storage(Tensor(a!) tensor, int address, int object_id, "
      "int generation) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl("_rebind_storage", TORCH_FN(rebind_storage));
}
