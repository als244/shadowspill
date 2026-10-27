#include "internal.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace {

at::Tensor make_runtime_cpu_storage(
    const at::Tensor& dispatch,
    int64_t pool_id,
    int64_t target_address,
    int64_t object_id,
    int64_t size_bytes
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_make_runtime_cpu");
  TORCH_CHECK(dispatch.device().is_cpu(), "runtime storage dispatch must be CPU");
  TORCH_CHECK(pool_id >= 0, "pool ID must be nonnegative");
  TORCH_CHECK(target_address > 0, "runtime address must be positive");
  TORCH_CHECK(object_id >= 0, "object ID must be nonnegative");
  TORCH_CHECK(size_bytes > 0, "spill storage size must be positive");
  const ShadowSpillStatus status =
      shadowspill_pytorch_validate_object_binding(
          static_cast<uint32_t>(pool_id),
          static_cast<uint64_t>(object_id),
          static_cast<uint64_t>(target_address),
          static_cast<uint64_t>(size_bytes));
  TORCH_CHECK(
      status == SHADOWSPILL_STATUS_OK,
      "CPU runtime storage does not name a current pool lease: ",
      shadowspill_status_string(status));
  return at::from_blob(
      reinterpret_cast<void*>(static_cast<uintptr_t>(target_address)),
      {size_bytes},
      [](void*) {},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(shadowspill, library) {
  library.def(
      "_make_runtime_cpu_storage(Tensor dispatch, int pool_id, int address, "
      "int object_id, int size) -> Tensor");
}

TORCH_LIBRARY_IMPL(shadowspill, CPU, library) {
  library.impl("_make_runtime_cpu_storage", TORCH_FN(make_runtime_cpu_storage));
}
