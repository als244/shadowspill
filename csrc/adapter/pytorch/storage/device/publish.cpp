#include "internal.h"
#include <c10/cuda/CUDAStream.h>

namespace shadowspill_pytorch_storage {

void adopt_storages(
    at::TensorList tensors,
    at::IntArrayRef publication_ordinals,
    int64_t task_handle
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_adopt_batch");
  const size_t count = tensors.size();
  TORCH_CHECK(
      publication_ordinals.size() == count,
      "storage adoption batch fields must have equal lengths");
  TORCH_CHECK(task_handle > 0, "task handle must be positive");

  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    TORCH_CHECK(tensor.is_cuda(), "storage adoption requires CUDA tensors");
    TORCH_CHECK(
        publication_ordinals[index] >= 0,
        "publication ordinal must be nonnegative");
    const uint64_t address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    TORCH_CHECK(
        address != 0U,
        "adopted storage is dematerialized: batch index ",
        index,
        ", publication ordinal ",
        publication_ordinals[index],
        ", tensor numel ",
        tensor.numel());
  }

  uintptr_t runtime_handle = 0;
  TORCH_CHECK(
      shadowspill_pytorch_runtime_handle(&runtime_handle) ==
          SHADOWSPILL_STATUS_OK,
      "the ShadowSpill runtime is closed");
  auto *const runtime = reinterpret_cast<ShadowSpillRuntime *>(runtime_handle);

  for (const auto index : c10::irange(count)) {
    const uint64_t address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
        tensors[index].storage().data_ptr().get()));
    ShadowSpillObjectBinding binding = {};
    const ShadowSpillStatus status =
        shadowspill_task_publish_allocation(
            runtime,
            reinterpret_cast<const ShadowSpillTaskHandle *>(task_handle),
            static_cast<uint32_t>(publication_ordinals[index]),
            reinterpret_cast<const void *>(static_cast<uintptr_t>(address)),
            &binding);
    TORCH_CHECK(
        status == SHADOWSPILL_STATUS_OK,
        "storage adoption failed at batch index ",
        index,
        ", publication ordinal ",
        publication_ordinals[index],
        ", address ",
        address,
        ": ",
        shadowspill_status_string(status));
    TORCH_CHECK(
        binding.pointer == reinterpret_cast<void*>(
            static_cast<uintptr_t>(address)),
        "storage adoption changed the allocation address");
  }

  // Runtime adoption succeeds for the complete batch before any owning
  // DataPtr is replaced. Clearing the prior DataPtr then reports PyTorch's
  // logical free while the runtime keeps the plan-owned allocation alive.
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    const uint64_t address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
        tensor.storage().data_ptr().get()));
    c10::Storage storage = tensor.storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(static_cast<uintptr_t>(address)),
        tensor.device()));
    prior.clear();
  }
}

// Publish one task boundary to the runtime. The stream comes from the ordinal
// the caller names rather than from a tensor, which is what lets a boundary
// with no storage to adopt be published at all.
void after_task_boundary(int64_t task_handle, int64_t device_ordinal) {
  TORCH_CHECK(task_handle > 0, "task handle must be positive");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_ordinal));
  const ShadowSpillStatus status =
      shadowspill_pytorch_after_task_handle(
          static_cast<uintptr_t>(task_handle),
          reinterpret_cast<uintptr_t>(stream.stream()));
  TORCH_CHECK(
      status == SHADOWSPILL_STATUS_OK,
      "task publication failed: ",
      shadowspill_status_string(status));
}

void after_task_storages(
    at::TensorList adopted_tensors,
    at::IntArrayRef publication_ordinals,
    at::TensorList replacement_tensors,
    at::IntArrayRef replacement_target_indices,
    at::TensorList dematerialized_tensors,
    int64_t task_handle,
    int64_t device_ordinal
) {
  TORCH_CHECK(task_handle > 0, "task handle must be positive");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  adopt_storages(adopted_tensors, publication_ordinals, task_handle);
  rebind_replacement_views(
      replacement_tensors,
      replacement_target_indices,
      adopted_tensors,
      publication_ordinals,
      task_handle);
  dematerialize_storages(dematerialized_tensors);
  after_task_boundary(task_handle, device_ordinal);
}

}  // namespace shadowspill_pytorch_storage

using namespace shadowspill_pytorch_storage;

TORCH_LIBRARY_FRAGMENT(shadowspill, library) {
  library.def(
      "_after_task_boundary(int task_handle, int device_ordinal) -> ()",
      TORCH_FN(after_task_boundary));
  library.def(
      "_after_task_storages(Tensor(a!)[] adopted_tensors, int[] "
      "publication_ordinals, Tensor(a!)[] replacement_tensors, int[] "
      "replacement_target_indices, "
      "Tensor(a!)[] dematerialized_tensors, int task_handle, "
      "int device_ordinal) -> ()");
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl("_after_task_storages", TORCH_FN(after_task_storages));
}
