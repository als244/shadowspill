#include "internal.h"
#include "../../tasks/internal.h"
#include <c10/cuda/CUDAStream.h>

namespace shadowspill_pytorch_storage {

void wait_task_allocations(int64_t task_handle, int64_t device_ordinal) {
  TORCH_CHECK(task_handle > 0, "task handle must be positive");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_ordinal));
  const ShadowSpillStatus status =
      shadowspill_pytorch_wait_task_allocations(
          static_cast<uintptr_t>(task_handle),
          reinterpret_cast<uintptr_t>(stream.stream()));
  TORCH_CHECK(
      status == SHADOWSPILL_STATUS_OK,
      "task allocation dependency wait failed: ",
      shadowspill_status_string(status));
}

// Open one task boundary with the runtime. The stream comes from the ordinal
// the caller names rather than from a tensor, which is what lets a task that
// acquires no storage open at all.
void acquire_task_boundary(
    int64_t task_handle,
    int64_t device_ordinal,
    const ShadowSpillObjectBinding** bindings,
    uint32_t* binding_count
) {
  TORCH_CHECK(task_handle > 0, "task handle must be positive");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_ordinal));
  const ShadowSpillStatus status =
      shadowspill_pytorch_before_task_handle(
          static_cast<uintptr_t>(task_handle),
          reinterpret_cast<uintptr_t>(stream.stream()),
          bindings,
          binding_count);
  TORCH_CHECK(
      status == SHADOWSPILL_STATUS_OK,
      "task acquisition failed: ",
      shadowspill_status_string(status));
}

void before_task_boundary(int64_t task_handle, int64_t device_ordinal) {
  const ShadowSpillObjectBinding* bindings = nullptr;
  uint32_t binding_count = 0U;
  acquire_task_boundary(task_handle, device_ordinal, &bindings, &binding_count);
  TORCH_CHECK(
      binding_count == 0U,
      "task with no storage to acquire was given bindings");
}

void before_task_storages(
    at::TensorList tensors,
    int64_t task_handle,
    int64_t device_ordinal
) {
  TORCH_CHECK(task_handle > 0, "task handle must be positive");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  const size_t count = tensors.size();
  for (const at::Tensor& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda(), "task acquisition requires CUDA tensors");
    TORCH_CHECK(
        tensor.get_device() == device_ordinal,
        "task acquisition tensor is on the wrong CUDA device");
  }

  const ShadowSpillObjectBinding* bindings = nullptr;
  uint32_t binding_count = 0U;
  acquire_task_boundary(task_handle, device_ordinal, &bindings, &binding_count);
  TORCH_CHECK(
      binding_count == count && (count == 0U || bindings != nullptr),
      "task acquisition returned the wrong binding count");

  struct TaskScopeGuard {
    uintptr_t task_handle;
    bool active = true;
    ~TaskScopeGuard() {
      if (active) {
        (void)shadowspill_pytorch_abort_task_handle(task_handle);
      }
    }
  } scope_guard{static_cast<uintptr_t>(task_handle)};
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    const ShadowSpillObjectBinding& binding = bindings[index];
    const uint64_t target_address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(binding.pointer));
    TORCH_CHECK(target_address != 0U, "task acquired a null device address");
    const uint64_t current_address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    TORCH_CHECK(
        current_address == 0U || current_address == target_address,
        "storage does not name its acquired runtime generation");
  }
  for (const auto index : c10::irange(count)) {
    const uint64_t target_address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(bindings[index].pointer));
    const at::Tensor& tensor = tensors[index];
    const uint64_t current_address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    if (current_address == target_address) {
      continue;
    }
    c10::Storage storage = tensor.storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(static_cast<uintptr_t>(target_address)),
        tensor.device()));
    prior.clear();
  }
  scope_guard.active = false;
}

}  // namespace shadowspill_pytorch_storage

using namespace shadowspill_pytorch_storage;

TORCH_LIBRARY_FRAGMENT(shadowspill, library) {
  library.def(
      "_before_task_storages(Tensor(a!)[] tensors, int task_handle, "
      "int device_ordinal) -> ()");
  // Handle-only boundaries also support tasks with no tensor inputs.
  library.def(
      "_wait_task_allocations(int task_handle, int device_ordinal) -> ()",
      TORCH_FN(wait_task_allocations));
  library.def(
      "_before_task_boundary(int task_handle, int device_ordinal) -> ()",
      TORCH_FN(before_task_boundary));
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl("_before_task_storages", TORCH_FN(before_task_storages));
}
