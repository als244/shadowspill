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
  int32_t device_ordinal;
};

void import_cpu_storages(
    at::TensorList tensors,
    at::IntArrayRef target_addresses,
    at::IntArrayRef object_ids,
    at::IntArrayRef sizes
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_import_cpu_batch");
  const size_t count = tensors.size();
  TORCH_CHECK(
      target_addresses.size() == count && object_ids.size() == count &&
          sizes.size() == count,
      "CPU storage import fields must have equal lengths");
  std::vector<uint64_t> current_addresses;
  current_addresses.reserve(count);
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = tensors[index];
    TORCH_CHECK(tensor.device().is_cpu(), "storage import requires CPU tensors");
    TORCH_CHECK(target_addresses[index] >= 0, "spill address must be nonnegative");
    TORCH_CHECK(object_ids[index] >= 0, "object ID must be nonnegative");
    TORCH_CHECK(sizes[index] >= 0, "storage size must be nonnegative");
    TORCH_CHECK(
        static_cast<uint64_t>(tensor.storage().nbytes()) ==
            static_cast<uint64_t>(sizes[index]),
        "CPU storage size differs from its spill lease");
    const ShadowSpillRuntimeStatus status =
        shadowspill_pytorch_validate_spill_binding(
            static_cast<uint64_t>(object_ids[index]),
            static_cast<uint64_t>(target_addresses[index]),
            static_cast<uint64_t>(sizes[index]));
    TORCH_CHECK(
        status == SHADOWSPILL_RUNTIME_OK,
        "CPU storage import does not name a current spill lease: ",
        shadowspill_runtime_status_string(status));
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

at::Tensor make_spill_cpu_storage(
    const at::Tensor& dispatch,
    int64_t target_address,
    int64_t object_id,
    int64_t size_bytes
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_make_spill_cpu");
  TORCH_CHECK(dispatch.device().is_cpu(), "spill storage dispatch must be CPU");
  TORCH_CHECK(target_address > 0, "spill address must be positive");
  TORCH_CHECK(object_id >= 0, "object ID must be nonnegative");
  TORCH_CHECK(size_bytes > 0, "spill storage size must be positive");
  const ShadowSpillRuntimeStatus status =
      shadowspill_pytorch_validate_spill_binding(
          static_cast<uint64_t>(object_id),
          static_cast<uint64_t>(target_address),
          static_cast<uint64_t>(size_bytes));
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "CPU spill storage does not name a current spill lease: ",
      shadowspill_runtime_status_string(status));
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

void release_caller_lease(void* context) {
  auto* lease = static_cast<CallerLease*>(context);
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

std::vector<int64_t> before_task_storages(
    at::TensorList tensors,
    int64_t task_handle,
    int64_t task_id,
    int64_t device_ordinal
) {
  TORCH_CHECK(task_handle > 0, "task handle must be positive");
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
      shadowspill_pytorch_before_task_handle(
          static_cast<uintptr_t>(task_handle),
          static_cast<uint64_t>(task_id),
          reinterpret_cast<uintptr_t>(stream.stream()),
          bindings.data(),
          static_cast<uint32_t>(count));
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "task acquisition failed: ",
      shadowspill_runtime_status_string(status));

  struct TaskScopeGuard {
    uintptr_t task_handle;
    uint64_t task_id;
    bool active = true;
    ~TaskScopeGuard() {
      if (active) {
        (void)shadowspill_pytorch_abort_task_handle(task_handle, task_id);
      }
    }
  } scope_guard{
      static_cast<uintptr_t>(task_handle),
      static_cast<uint64_t>(task_id)};
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
    at::IntArrayRef publication_ordinals,
    int64_t task_handle
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_adopt_batch");
  const size_t count = tensors.size();
  TORCH_CHECK(
      publication_ordinals.size() == count,
      "storage adoption batch fields must have equal lengths");
  TORCH_CHECK(task_handle > 0, "task handle must be positive");

  std::vector<uint64_t> addresses;
  addresses.reserve(count);
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
    addresses.push_back(address);
  }

  std::vector<ShadowSpillObjectBinding> bindings(count);
  std::vector<int64_t> generations;
  generations.reserve(count);
  for (const auto index : c10::irange(count)) {
    const ShadowSpillRuntimeStatus status =
        shadowspill_pytorch_task_publish_allocation(
            static_cast<uintptr_t>(task_handle),
            static_cast<uint32_t>(publication_ordinals[index]),
            addresses[index],
            &bindings[index]);
    TORCH_CHECK(
        status == SHADOWSPILL_RUNTIME_OK,
        "storage adoption failed at batch index ",
        index,
        ", publication ordinal ",
        publication_ordinals[index],
        ", address ",
        addresses[index],
        ": ",
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

void rebind_replacement_views(
    at::TensorList replacement_tensors,
    at::IntArrayRef previous_generations,
    at::IntArrayRef target_indices,
    at::TensorList adopted_tensors,
    at::IntArrayRef publication_ordinals,
    int64_t task_handle,
    const std::vector<int64_t>& target_generations
) {
  RangeGuard range_guard("shadowspill.pytorch.storage_replace_views_batch");
  const size_t count = replacement_tensors.size();
  TORCH_CHECK(
      previous_generations.size() == count && target_indices.size() == count,
      "storage replacement fields must have equal lengths");
  TORCH_CHECK(
      publication_ordinals.size() == adopted_tensors.size() &&
          target_generations.size() == adopted_tensors.size(),
      "adopted storage fields must have equal lengths");

  std::vector<uint64_t> current_addresses;
  std::vector<uint64_t> target_addresses;
  current_addresses.reserve(count);
  target_addresses.reserve(count);

  // Validate every old and new generation before rebinding any frontend view.
  // The runtime has adopted the replacement, but has not yet published task
  // actions, so both generations remain valid throughout this transaction.
  for (const auto index : c10::irange(count)) {
    const at::Tensor& tensor = replacement_tensors[index];
    const int64_t target_index = target_indices[index];
    TORCH_CHECK(tensor.is_cuda(), "storage replacement requires CUDA tensors");
    TORCH_CHECK(
        previous_generations[index] >= 0,
        "previous storage generation must be nonnegative");
    TORCH_CHECK(
        target_index >= 0 &&
            static_cast<size_t>(target_index) < adopted_tensors.size(),
        "replacement target index is out of range");
    TORCH_CHECK(
        publication_ordinals[target_index] >= 0 &&
            target_generations[target_index] >= 0,
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
      const ShadowSpillRuntimeStatus status =
          shadowspill_pytorch_validate_task_publication_binding(
              static_cast<uintptr_t>(task_handle),
              static_cast<uint32_t>(publication_ordinals[target_index]),
              current,
              static_cast<uint64_t>(previous_generations[index]));
      TORCH_CHECK(
          status == SHADOWSPILL_RUNTIME_OK,
          "existing storage does not match the retired object generation: ",
          shadowspill_runtime_status_string(status));
    }
    current_addresses.push_back(current);
    target_addresses.push_back(target);
  }
  for (const auto index : c10::irange(count)) {
    if (current_addresses[index] == target_addresses[index]) {
      continue;
    }
    c10::Storage storage = replacement_tensors[index].storage();
    c10::DataPtr prior = storage.set_data_ptr(c10::DataPtr(
        reinterpret_cast<void*>(
            static_cast<uintptr_t>(target_addresses[index])),
        replacement_tensors[index].device()));
    prior.clear();
  }
}

std::vector<int64_t> after_task_storages(
    at::TensorList adopted_tensors,
    at::IntArrayRef publication_ordinals,
    at::TensorList replacement_tensors,
    at::IntArrayRef replacement_previous_generations,
    at::IntArrayRef replacement_target_indices,
    at::TensorList dematerialized_tensors,
    int64_t task_handle,
    int64_t task_id,
    int64_t device_ordinal
) {
  TORCH_CHECK(task_handle > 0, "task handle must be positive");
  TORCH_CHECK(task_id >= 0, "task ID must be nonnegative");
  TORCH_CHECK(device_ordinal >= 0, "device ordinal must be nonnegative");
  std::vector<int64_t> generations = adopt_storages(
      adopted_tensors, publication_ordinals, task_handle);
  rebind_replacement_views(
      replacement_tensors,
      replacement_previous_generations,
      replacement_target_indices,
      adopted_tensors,
      publication_ordinals,
      task_handle,
      generations);
  dematerialize_storages(dematerialized_tensors);
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_ordinal));
  const ShadowSpillRuntimeStatus status =
      shadowspill_pytorch_after_task_handle(
          static_cast<uintptr_t>(task_handle),
          static_cast<uint64_t>(task_id),
          reinterpret_cast<uintptr_t>(stream.stream()));
  TORCH_CHECK(
      status == SHADOWSPILL_RUNTIME_OK,
      "task publication failed: ",
      shadowspill_runtime_status_string(status));
  return generations;
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
  const ShadowSpillRuntimeStatus status =
      shadowspill_pytorch_transfer_acquired_object_to_caller(
      static_cast<uintptr_t>(acquisition_handle),
      static_cast<uint32_t>(object_ordinal),
      reinterpret_cast<uintptr_t>(stream.stream()),
      address,
      static_cast<uint64_t>(generation),
      static_cast<uint64_t>(allocation_id),
      &allocation);
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

TORCH_LIBRARY(shadowspill, library) {
  library.def(
      "_import_cpu_storages(Tensor(a!)[] tensors, int[] addresses, "
      "int[] object_ids, int[] sizes) -> ()");
  library.def(
      "_export_cpu_storages(Tensor(a!)[] tensors, Tensor[] owners) -> ()");
  library.def(
      "_make_spill_cpu_storage(Tensor dispatch, int address, int object_id, "
      "int size) -> Tensor");
  library.def(
      "_acquire_storages(Tensor(a!)[] tensors, int[] addresses) -> ()");
  library.def(
      "_before_task_storages(Tensor(a!)[] tensors, int task_handle, "
      "int task_id, int device_ordinal) -> int[]");
  library.def("_dematerialize_storages(Tensor(a!)[] tensors) -> ()");
  library.def(
      "_after_task_storages(Tensor(a!)[] adopted_tensors, int[] "
      "publication_ordinals, Tensor(a!)[] replacement_tensors, int[] "
      "replacement_previous_generations, int[] replacement_target_indices, "
      "Tensor(a!)[] dematerialized_tensors, int task_handle, int task_id, "
      "int device_ordinal) -> int[]");
  library.def(
      "_transfer_acquired_storage_to_caller(Tensor(a!) tensor, int "
      "acquisition_handle, int object_ordinal, int generation, int "
      "allocation_id) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(shadowspill, CPU, library) {
  library.impl("_import_cpu_storages", TORCH_FN(import_cpu_storages));
  library.impl("_export_cpu_storages", TORCH_FN(export_cpu_storages));
  library.impl("_make_spill_cpu_storage", TORCH_FN(make_spill_cpu_storage));
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
