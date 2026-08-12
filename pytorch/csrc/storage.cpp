#include <shadowspill/pytorch_adapter.h>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "profiler.h"

#include <cstdint>
#include <vector>
#include <utility>

namespace {

struct RangeGuard {
  explicit RangeGuard(const char* name)
      : range(shadowspill_pytorch_profile_range_begin(name)) {}
  ~RangeGuard() { shadowspill_pytorch_profile_range_end(range); }

  ShadowSpillProfilerRange range;
};

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
  RangeGuard range_guard("shadowspill.pytorch.storage_rebind");
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

void rebind_storages(
    at::TensorList tensors,
    at::IntArrayRef target_addresses,
    at::IntArrayRef object_ids,
    at::IntArrayRef generations
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_rebind_batch");
  const size_t count = tensors.size();
  TORCH_CHECK(
      target_addresses.size() == count && object_ids.size() == count &&
          generations.size() == count,
      "storage rebinding batch fields must have equal lengths");

  std::vector<uint64_t> current_addresses;
  current_addresses.reserve(count);

  // Validate the complete request before changing any Tensor storage. A bad
  // entry therefore cannot leave a partially rebound task boundary.
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    const int64_t target_address = target_addresses[index];
    const int64_t object_id = object_ids[index];
    const int64_t generation = generations[index];
    TORCH_CHECK(tensor.is_cuda(), "storage rebinding requires CUDA tensors");
    TORCH_CHECK(target_address >= 0, "storage address must be nonnegative");
    TORCH_CHECK(object_id >= 0, "object ID must be nonnegative");
    TORCH_CHECK(generation >= 0, "allocation generation must be nonnegative");

    const c10::DataPtr& current = tensor.storage().data_ptr();
    const uint64_t current_address =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(current.get()));
    current_addresses.push_back(current_address);
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
    if (target_address != 0 &&
        static_cast<uint64_t>(target_address) != current_address) {
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
  }

  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    if (current_addresses[index] ==
        static_cast<uint64_t>(target_addresses[index])) {
      continue;
    }
    c10::Storage storage = tensor.storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(
            static_cast<uintptr_t>(target_addresses[index])),
        tensor.device()));
    prior.clear();
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

std::vector<int64_t> before_execution_storages(
    at::TensorList tensors,
    int64_t execution_handle,
    int64_t task_id,
    int64_t device_ordinal
) {
  TORCH_CHECK(execution_handle > 0, "execution handle must be positive");
  TORCH_CHECK(task_id >= 0, "task ID must be nonnegative");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  const size_t count = tensors.size();
  for (const at::Tensor& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda(), "task acquisition requires CUDA tensors");
    TORCH_CHECK(
        tensor.get_device() == device_ordinal,
        "task acquisition tensor is on the wrong CUDA device");
  }

  std::vector<ShadowSpillObjectBinding> bindings(count);
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_ordinal));
  const ShadowSpillRuntimeStatus status =
      shadowspill_pytorch_before_execution_handle(
          static_cast<uintptr_t>(execution_handle),
          static_cast<uint64_t>(task_id),
          reinterpret_cast<uintptr_t>(stream.stream()),
          bindings.data(),
          static_cast<uint32_t>(count));
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "task acquisition failed: ",
      shadowspill_runtime_status_string(status));

  struct TaskScopeGuard {
    bool active = true;
    ~TaskScopeGuard() {
      if (active) {
        shadowspill_pytorch_abort_task_range();
      }
    }
  } scope_guard;
  std::vector<uint64_t> current_addresses;
  std::vector<int64_t> generations;
  current_addresses.reserve(count);
  generations.reserve(count);
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
    current_addresses.push_back(current_address);
    generations.push_back(static_cast<int64_t>(binding.generation));
  }
  for (const auto index : c10::irange(count)) {
    const uint64_t target_address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(bindings[index].pointer));
    if (current_addresses[index] == target_address) {
      continue;
    }
    const at::Tensor& tensor = tensors[index];
    c10::Storage storage = tensor.storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(static_cast<uintptr_t>(target_address)),
        tensor.device()));
    prior.clear();
  }
  scope_guard.active = false;
  return generations;
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

std::vector<int64_t> adopt_storages(
    at::TensorList tensors,
    at::IntArrayRef object_ids,
    at::IntArrayRef sizes,
    at::IntArrayRef modes
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_adopt_batch");
  const size_t count = tensors.size();
  TORCH_CHECK(
      object_ids.size() == count && sizes.size() == count &&
          modes.size() == count,
      "storage adoption batch fields must have equal lengths");

  std::vector<uint64_t> addresses;
  addresses.reserve(count);
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    TORCH_CHECK(tensor.is_cuda(), "storage adoption requires CUDA tensors");
    TORCH_CHECK(object_ids[index] >= 0, "object ID must be nonnegative");
    TORCH_CHECK(sizes[index] >= 0, "object size must be nonnegative");
    TORCH_CHECK(
        modes[index] >= 0 && modes[index] <= 2,
        "storage adoption mode must be promote, bind, or replace");
    const uint64_t address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    TORCH_CHECK(
        address != 0U,
        "adopted storage is dematerialized: batch index ",
        index,
        ", object ",
        object_ids[index],
        ", declared bytes ",
        sizes[index],
        ", tensor numel ",
        tensor.numel());
    addresses.push_back(address);
  }

  std::vector<ShadowSpillObjectBinding> bindings(count);
  std::vector<int64_t> generations;
  generations.reserve(count);
  for (const auto index : c10::irange(count)) {
    ShadowSpillRuntimeStatus status = modes[index] == 2
        ? shadowspill_pytorch_replace_registered_allocation(
              static_cast<uint64_t>(object_ids[index]),
              addresses[index],
              static_cast<uint64_t>(sizes[index]),
              &bindings[index])
        : modes[index] == 1
          ? shadowspill_pytorch_bind_registered_allocation(
              static_cast<uint64_t>(object_ids[index]),
              addresses[index],
              static_cast<uint64_t>(sizes[index]),
              &bindings[index])
          : shadowspill_pytorch_promote_allocation(
              static_cast<uint64_t>(object_ids[index]),
              addresses[index],
              static_cast<uint64_t>(sizes[index]),
              &bindings[index]);
    TORCH_CHECK(
        status == SHADOWSPILL_RUNTIME_OK,
        "storage adoption failed: ",
        shadowspill_runtime_status_string(status));
    TORCH_CHECK(
        bindings[index].pointer == reinterpret_cast<void*>(
            static_cast<uintptr_t>(addresses[index])),
        "storage adoption changed the allocation address");
    generations.push_back(static_cast<int64_t>(bindings[index].generation));
  }

  // Runtime adoption succeeds for the complete batch before any owning
  // DataPtr is replaced. Clearing the prior DataPtr then reports PyTorch's
  // logical free while the runtime keeps the plan-owned allocation alive.
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    c10::Storage storage = tensor.storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(static_cast<uintptr_t>(addresses[index])),
        tensor.device()));
    prior.clear();
  }
  return generations;
}

void replace_storages(
    at::TensorList tensors,
    int64_t object_id,
    int64_t previous_generation,
    int64_t target_address,
    int64_t target_generation
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_replace_batch");
  TORCH_CHECK(object_id >= 0, "object ID must be nonnegative");
  TORCH_CHECK(
      previous_generation >= 0 && target_generation >= 0,
      "storage generations must be nonnegative");
  TORCH_CHECK(target_address > 0, "replacement address must be positive");
  const uint64_t object = static_cast<uint64_t>(object_id);
  const uint64_t target = static_cast<uint64_t>(target_address);
  ShadowSpillRuntimeStatus status = shadowspill_pytorch_validate_object_binding(
      object, target, static_cast<uint64_t>(target_generation));
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "replacement storage does not match the new object generation: ",
      shadowspill_runtime_status_string(status));

  std::vector<uint64_t> current_addresses;
  current_addresses.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda(), "storage replacement requires CUDA tensors");
    const uint64_t current = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    if (current != target) {
      status = shadowspill_pytorch_validate_object_binding(
          object, current, static_cast<uint64_t>(previous_generation));
      TORCH_CHECK(
          status == SHADOWSPILL_RUNTIME_OK,
          "existing storage does not match the retired object generation: ",
          shadowspill_runtime_status_string(status));
    }
    current_addresses.push_back(current);
  }
  for (const auto index : c10::irange(tensors.size())) {
    if (current_addresses[index] == target) {
      continue;
    }
    c10::Storage storage = tensors[index].storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(static_cast<uintptr_t>(target)),
        tensors[index].device()));
    prior.clear();
  }
}

std::vector<int64_t> after_execution_storages(
    at::TensorList adopted_tensors,
    at::IntArrayRef object_ids,
    at::IntArrayRef sizes,
    at::IntArrayRef modes,
    at::TensorList dematerialized_tensors,
    int64_t execution_handle,
    int64_t task_id,
    int64_t device_ordinal
) {
  TORCH_CHECK(execution_handle > 0, "execution handle must be positive");
  TORCH_CHECK(task_id >= 0, "task ID must be nonnegative");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  std::vector<int64_t> generations = adopt_storages(
      adopted_tensors, object_ids, sizes, modes);
  dematerialize_storages(dematerialized_tensors);
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_ordinal));
  const ShadowSpillRuntimeStatus status =
      shadowspill_pytorch_after_execution_handle(
          static_cast<uintptr_t>(execution_handle),
          static_cast<uint64_t>(task_id),
          reinterpret_cast<uintptr_t>(stream.stream()));
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "task publication failed: ",
      shadowspill_runtime_status_string(status));
  return generations;
}

at::Tensor transfer_storage_to_caller(
    const at::Tensor& tensor,
    int64_t object_id,
    int64_t generation,
    int64_t allocation_id
) {
  RangeGuard range_guard("shadowspill.pytorch.caller_lease");
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
      "_rebind_storages(Tensor(a!)[] tensors, int[] addresses, "
      "int[] object_ids, int[] generations) -> ()");
  library.def(
      "_acquire_storages(Tensor(a!)[] tensors, int[] addresses) -> ()");
  library.def(
      "_before_execution_storages(Tensor(a!)[] tensors, int execution_handle, "
      "int task_id, int device_ordinal) -> int[]");
  library.def("_dematerialize_storages(Tensor(a!)[] tensors) -> ()");
  library.def(
      "_adopt_storages(Tensor(a!)[] tensors, int[] object_ids, int[] sizes, "
      "int[] modes) -> int[]");
  library.def(
      "_replace_storages(Tensor(a!)[] tensors, int object_id, int "
      "previous_generation, int target_address, int target_generation) -> ()");
  library.def(
      "_after_execution_storages(Tensor(a!)[] adopted_tensors, int[] "
      "object_ids, int[] sizes, int[] modes, Tensor(a!)[] "
      "dematerialized_tensors, int execution_handle, int task_id, int "
      "device_ordinal) -> int[]");
  library.def(
      "_transfer_storage_to_caller(Tensor(a!) tensor, int object_id, "
      "int generation, int allocation_id) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(shadowspill, CUDA, library) {
  library.impl("_rebind_storage", TORCH_FN(rebind_storage));
  library.impl("_rebind_storages", TORCH_FN(rebind_storages));
  library.impl("_acquire_storages", TORCH_FN(acquire_storages));
  library.impl(
      "_before_execution_storages", TORCH_FN(before_execution_storages));
  library.impl(
      "_dematerialize_storages", TORCH_FN(dematerialize_storages));
  library.impl("_adopt_storages", TORCH_FN(adopt_storages));
  library.impl("_replace_storages", TORCH_FN(replace_storages));
  library.impl(
      "_after_execution_storages", TORCH_FN(after_execution_storages));
  library.impl(
      "_transfer_storage_to_caller", TORCH_FN(transfer_storage_to_caller));
}
