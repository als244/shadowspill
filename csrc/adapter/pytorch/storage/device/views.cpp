#include "internal.h"
#include <c10/core/Allocator.h>
#include <vector>

namespace shadowspill_pytorch_storage {

at::Tensor make_unbacked_cuda_storage(const at::Tensor& dispatch, int64_t bytes) {
  TORCH_CHECK(dispatch.is_cuda(), "storage dispatch must be CUDA");
  TORCH_CHECK(bytes >= 0, "storage size must be nonnegative");
  c10::Storage storage(
      c10::Storage::use_byte_size_t(), static_cast<size_t>(bytes),
      c10::DataPtr(nullptr, dispatch.device()), nullptr, false);
  at::Tensor result = at::empty({0}, dispatch.options().dtype(at::kByte));
  result.set_(storage, 0, {bytes}, {1});
  return result;
}

void acquire_storages(
    at::TensorList tensors,
    at::IntArrayRef target_addresses
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_acquire_batch");
  const size_t count = tensors.size();
  TORCH_CHECK(
      target_addresses.size() == count,
      "storage acquisition batch fields must have equal lengths");
  std::vector<uint64_t> current_addresses;
  current_addresses.reserve(count);
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    const int64_t target_address = target_addresses[index];
    TORCH_CHECK(tensor.is_cuda(), "storage acquisition requires CUDA tensors");
    TORCH_CHECK(target_address > 0, "acquired address must be positive");
    const uint64_t current_address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    TORCH_CHECK(
        current_address == 0U ||
            current_address == static_cast<uint64_t>(target_address),
        "storage does not name its acquired runtime generation");
    current_addresses.push_back(current_address);
  }
  for (const auto index : c10::irange(count)) {
    if (current_addresses[index] ==
        static_cast<uint64_t>(target_addresses[index])) {
      continue;
    }
    const at::Tensor& tensor = tensors[index];
    c10::Storage storage = tensor.storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(
            static_cast<uintptr_t>(target_addresses[index])),
        tensor.device()));
    prior.clear();
  }
}

void dematerialize_storages(at::TensorList tensors) {
  RangeGuard range_guard("shadowspill.pytorch.storage_dematerialize_batch");
  for (const at::Tensor& tensor : tensors) {
    TORCH_CHECK(
        tensor.is_cuda(), "storage dematerialization requires CUDA tensors");
    TORCH_CHECK(
        tensor.storage().data_ptr().get() != nullptr,
        "storage is already dematerialized");
  }
  for (const at::Tensor& tensor : tensors) {
    c10::Storage storage = tensor.storage();
    c10::DataPtr prior =
        storage.set_data_ptr(c10::DataPtr(nullptr, tensor.device()));
    prior.clear();
  }
}

}  // namespace shadowspill_pytorch_storage

using namespace shadowspill_pytorch_storage;

TORCH_LIBRARY_FRAGMENT(shadowspill, library) {
  library.def("_make_unbacked_cuda_storage(Tensor dispatch, int bytes) -> Tensor");
  library.def("_acquire_storages(Tensor(a!)[] tensors, int[] addresses) -> ()");
  library.def("_dematerialize_storages(Tensor(a!)[] tensors) -> ()");
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl("_make_unbacked_cuda_storage", TORCH_FN(make_unbacked_cuda_storage));
  library.impl("_acquire_storages", TORCH_FN(acquire_storages));
  library.impl("_dematerialize_storages", TORCH_FN(dematerialize_storages));
}
