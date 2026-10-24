/* Bounded telemetry from PyTorch's native allocator. No Python/GIL callbacks. */
#include "allocations.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <mutex>
#include <new>
#include <vector>

namespace {
using c10::cuda::CUDACachingAllocator::TraceEntry;

struct Observer {
  std::mutex mutex;
  std::once_flag attached;
  std::vector<ShadowSpillPytorchAllocationObservation> records;
  size_t count = 0;
  int32_t device = 0;
  bool enabled = false;
  bool overflowed = false;
};

Observer& observer() {
  // The allocator retains its callback until process exit. Keep its target alive
  // through allocator/module teardown, independently of static destructor order.
  static auto* value = new Observer;
  return *value;
}

void record(const TraceEntry& entry) {
  auto& state = observer();
  std::lock_guard<std::mutex> guard(state.mutex);
  if (!state.enabled || state.device != entry.device_) return;
  if (state.count == state.records.size()) {
    state.overflowed = true;
    return;
  }
  state.records[state.count++] = {
      static_cast<uint64_t>(entry.addr_),
      static_cast<uint64_t>(entry.size_),
      reinterpret_cast<uint64_t>(entry.stream_),
      entry.mempool_.first, entry.mempool_.second,
      entry.device_, static_cast<int32_t>(entry.action_)};
}
} // namespace

extern "C" SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_observe_allocations_begin(int32_t device, uint64_t capacity) {
  if (device < 0 || capacity == 0 || capacity > 1000000)
    return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
  try {
    auto& state = observer();
    // Attach outside our lock. Callback lock order is allocator -> observer;
    // no code holding the observer lock calls the allocator or acquires the GIL.
    std::call_once(state.attached, [] {
      c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(record);
    });
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.enabled) return SHADOWSPILL_STATUS_INVALID_STATE;
    state.records.resize(static_cast<size_t>(capacity));
    state.count = 0;
    state.device = device;
    state.overflowed = false;
    state.enabled = true;
    return SHADOWSPILL_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return SHADOWSPILL_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return SHADOWSPILL_STATUS_INVALID_STATE;
  }
}

extern "C" SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_observe_allocations_end(
    ShadowSpillPytorchAllocationObservation* records, uint64_t capacity,
    uint64_t* count, uint32_t* overflowed) {
  if (!records || !count || !overflowed) return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
  auto& state = observer();
  std::lock_guard<std::mutex> guard(state.mutex);
  if (!state.enabled) return SHADOWSPILL_STATUS_INVALID_STATE;
  state.enabled = false;
  *count = state.count;
  *overflowed = state.overflowed ? 1 : 0;
  if (capacity < state.count) return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
  std::copy_n(state.records.data(), state.count, records);
  return SHADOWSPILL_STATUS_OK;
}
