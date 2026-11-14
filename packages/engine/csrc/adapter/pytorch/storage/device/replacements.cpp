#include "internal.h"
#include <vector>

namespace shadowspill_pytorch_storage {

void rebind_replacement_views(
    at::TensorList replacement_tensors,
    at::IntArrayRef target_indices,
    at::TensorList adopted_tensors,
    at::IntArrayRef publication_ordinals,
    int64_t task_handle
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_replace_views_batch");
  const size_t count = replacement_tensors.size();
  TORCH_CHECK(
      target_indices.size() == count,
      "storage replacement fields must have equal lengths");
  TORCH_CHECK(
      publication_ordinals.size() == adopted_tensors.size(),
      "adopted storage fields must have equal lengths");
  if (count == 0U) {
    return;
  }

  uintptr_t runtime_handle = 0;
  TORCH_CHECK(
      shadowspill_pytorch_runtime_handle(&runtime_handle) ==
          SHADOWSPILL_STATUS_OK,
      "the ShadowSpill runtime is closed");
  auto *const runtime = reinterpret_cast<ShadowSpillRuntime *>(runtime_handle);

  // Validate every retired and successor address before rebinding any view.
  // The runtime has adopted the replacement, but has not yet published task
  // actions, so both leases remain stable throughout this transaction.
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = replacement_tensors[index];
    const int64_t target_index = target_indices[index];
    TORCH_CHECK(tensor.is_cuda(), "storage replacement requires CUDA tensors");
    TORCH_CHECK(
        target_index >= 0 &&
            static_cast<size_t>(target_index) < adopted_tensors.size(),
        "replacement target index is out of range");
    TORCH_CHECK(
        publication_ordinals[target_index] >= 0,
        "replacement target identity must be nonnegative");
    const at::Tensor& target_tensor = adopted_tensors[target_index];
    TORCH_CHECK(
        target_tensor.is_cuda() && tensor.get_device() == target_tensor.get_device(),
        "replacement views must be on the target CUDA device");
    const uint64_t target = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
        target_tensor.storage().data_ptr().get()));
    TORCH_CHECK(target != 0U, "replacement target is dematerialized");
    const uint64_t current = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    if (current != target) {
      const ShadowSpillStatus status =
          shadowspill_task_validate_replacement_binding(
              runtime,
              reinterpret_cast<const ShadowSpillTaskHandle *>(task_handle),
              static_cast<uint32_t>(publication_ordinals[target_index]),
              reinterpret_cast<const void *>(static_cast<uintptr_t>(current)),
              reinterpret_cast<const void *>(static_cast<uintptr_t>(target)));
      TORCH_CHECK(
          status == SHADOWSPILL_STATUS_OK,
          "existing storage does not match the retired object generation: ",
          shadowspill_status_string(status));
    }
  }
  // Every retained view of a logical object must share its StorageImpl.
  // Merely installing the same address in separate StorageImpls leaves those
  // other views stale when a later fetch rebinds the canonical representative.
  std::vector<int64_t> first_views(adopted_tensors.size(), -1);
  for (const auto index : c10::irange(count)) {
    const int64_t target_index = target_indices[index];
    int64_t& first_view = first_views[target_index];
    if (first_view >= 0) {
      replacement_tensors[index].unsafeGetTensorImpl()->set_storage_keep_dtype(
          replacement_tensors[first_view].storage());
      continue;
    }
    first_view = static_cast<int64_t>(index);
    const uint64_t target = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
        adopted_tensors[target_index].storage().data_ptr().get()));
    c10::Storage storage = replacement_tensors[index].storage();
    const uint64_t current = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(storage.data_ptr().get()));
    if (current != target) {
      c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
          reinterpret_cast<void*>(static_cast<uintptr_t>(target)),
          replacement_tensors[index].device()));
      prior.clear();
    }
  }
}

}  // namespace shadowspill_pytorch_storage
