#include "internal.h"

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <torch/library.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace {

void import_cpu_storages(
    at::TensorList tensors,
    at::IntArrayRef pool_ids,
    at::IntArrayRef target_addresses,
    at::IntArrayRef object_ids,
    at::IntArrayRef sizes
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_import_cpu_batch");
  const size_t count = tensors.size();
  TORCH_CHECK(
      pool_ids.size() == count && target_addresses.size() == count &&
          object_ids.size() == count && sizes.size() == count,
      "CPU storage import fields must have equal lengths");
  std::vector<uint64_t> current_addresses;
  current_addresses.reserve(count);
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    TORCH_CHECK(tensor.device().is_cpu(), "storage import requires CPU tensors");
    TORCH_CHECK(pool_ids[index] >= 0, "pool ID must be nonnegative");
    TORCH_CHECK(target_addresses[index] >= 0, "pool address must be nonnegative");
    TORCH_CHECK(object_ids[index] >= 0, "object ID must be nonnegative");
    TORCH_CHECK(sizes[index] >= 0, "storage size must be nonnegative");
    TORCH_CHECK(
        static_cast<uint64_t>(tensor.storage().nbytes()) ==
            static_cast<uint64_t>(sizes[index]),
        "CPU storage size differs from its runtime lease");
    const ShadowSpillStatus status =
        shadowspill_pytorch_validate_object_binding(
            static_cast<uint32_t>(pool_ids[index]),
            static_cast<uint64_t>(object_ids[index]),
            static_cast<uint64_t>(target_addresses[index]),
            static_cast<uint64_t>(sizes[index]));
    TORCH_CHECK(
        status == SHADOWSPILL_STATUS_OK,
        "CPU storage import does not name a current runtime lease: ",
        shadowspill_status_string(status));
    current_addresses.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
        tensor.storage().data_ptr().get())));
  }
  for (const auto index : c10::irange(count)) {
    if (current_addresses[index] ==
        static_cast<uint64_t>(target_addresses[index])) {
      continue;
    }
    c10::Storage storage = tensors[index].storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(
            static_cast<uintptr_t>(target_addresses[index])),
        c10::Device(c10::DeviceType::CPU)));
    prior.clear();
  }
}

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

void export_cpu_storages(
    at::TensorList tensors,
    at::TensorList owners
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_export_cpu_batch");
  TORCH_CHECK(
      tensors.size() == owners.size(),
      "CPU storage export fields must have equal lengths");
  for (const auto index : c10::irange(tensors.size())) {
    const at::Tensor& tensor = tensors[index];
    const at::Tensor& owner = owners[index];
    TORCH_CHECK(
        tensor.device().is_cpu() && owner.device().is_cpu(),
        "storage export requires CPU tensors");
    TORCH_CHECK(
        tensor.storage().nbytes() == owner.storage().nbytes(),
        "destination CPU owner has the wrong storage size");
    TORCH_CHECK(
        owner.storage().data_ptr().get() != nullptr ||
            owner.storage().nbytes() == 0,
        "destination CPU owner has no allocation");
  }
  for (const auto index : c10::irange(tensors.size())) {
    c10::Storage source = owners[index].storage();
    c10::DataPtr replacement = source.set_data_ptr(c10::DataPtr(
        nullptr, c10::Device(c10::DeviceType::CPU)));
    c10::Storage destination = tensors[index].storage();
    c10::DataPtr prior = destination.set_data_ptr(std::move(replacement));
    prior.clear();
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(shadowspill, library) {
  library.def(
      "_import_cpu_storages(Tensor(a!)[] tensors, int[] pool_ids, "
      "int[] addresses, int[] object_ids, int[] sizes) -> ()");
  library.def(
      "_export_cpu_storages(Tensor(a!)[] tensors, Tensor[] owners) -> ()");
  library.def(
      "_make_runtime_cpu_storage(Tensor dispatch, int pool_id, int address, "
      "int object_id, int size) -> Tensor");
}

TORCH_LIBRARY_IMPL(shadowspill, CPU, library) {
  library.impl("_import_cpu_storages", TORCH_FN(import_cpu_storages));
  library.impl("_export_cpu_storages", TORCH_FN(export_cpu_storages));
  library.impl(
      "_make_runtime_cpu_storage", TORCH_FN(make_runtime_cpu_storage));
}
