#include "internal.h"
#include "../tasks/internal.h"

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

namespace {

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
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_ordinal));
  const ShadowSpillStatus status =
      shadowspill_pytorch_before_task_handle(
          static_cast<uintptr_t>(task_handle),
          reinterpret_cast<uintptr_t>(stream.stream()),
          &bindings,
          &binding_count);
  TORCH_CHECK(
      status == SHADOWSPILL_STATUS_OK,
      "task acquisition failed: ",
      shadowspill_status_string(status));
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
          shadowspill_pytorch_validate_task_replacement_binding(
              static_cast<uintptr_t>(task_handle),
              static_cast<uint32_t>(publication_ordinals[target_index]),
              current,
              target);
      TORCH_CHECK(
          status == SHADOWSPILL_STATUS_OK,
          "existing storage does not match the retired object generation: ",
          shadowspill_status_string(status));
    }
  }
  for (const auto index : c10::irange(count)) {
    const int64_t target_index = target_indices[index];
    const uint64_t target = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
        adopted_tensors[target_index].storage().data_ptr().get()));
    const uint64_t current = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
        replacement_tensors[index].storage().data_ptr().get()));
    if (current == target) {
      continue;
    }
    c10::Storage storage = replacement_tensors[index].storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(
            static_cast<uintptr_t>(target)),
        replacement_tensors[index].device()));
    prior.clear();
  }
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

}  // namespace

TORCH_LIBRARY_FRAGMENT(shadowspill, library) {
  library.def(
      "_acquire_storages(Tensor(a!)[] tensors, int[] addresses) -> ()");
  library.def(
      "_before_task_storages(Tensor(a!)[] tensors, int task_handle, "
      "int device_ordinal) -> ()");
  library.def(
      "_wait_task_allocations(int task_handle, int device_ordinal) -> ()",
      TORCH_FN(wait_task_allocations));
  library.def("_dematerialize_storages(Tensor(a!)[] tensors) -> ()");
  library.def(
      "_after_task_storages(Tensor(a!)[] adopted_tensors, int[] "
      "publication_ordinals, Tensor(a!)[] replacement_tensors, int[] "
      "replacement_target_indices, "
      "Tensor(a!)[] dematerialized_tensors, int task_handle, "
      "int device_ordinal) -> ()");
  library.def(
      "_transfer_acquired_storage_to_caller(Tensor(a!) tensor, int "
      "acquisition_handle, int object_ordinal, int generation, int "
      "allocation_id) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl("_acquire_storages", TORCH_FN(acquire_storages));
  library.impl(
      "_before_task_storages", TORCH_FN(before_task_storages));
  library.impl(
      "_dematerialize_storages", TORCH_FN(dematerialize_storages));
  library.impl(
      "_after_task_storages", TORCH_FN(after_task_storages));
  library.impl(
      "_transfer_acquired_storage_to_caller",
      TORCH_FN(transfer_acquired_storage_to_caller));
}
