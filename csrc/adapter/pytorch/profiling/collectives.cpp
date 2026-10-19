/* Preparation-only collective intervals within one compiled task.
 * Use NCCL's work events, not host submission times or stream envelopes. */
#include <ATen/record_function.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/library.h>

#include <memory>
#include <string>
#include <vector>

namespace {
using Dict = c10::Dict<std::string, c10::IValue>;

Dict dictionary(std::initializer_list<std::pair<std::string, c10::IValue>> fields) {
  c10::impl::GenericDict result(c10::StringType::get(), c10::AnyType::get());
  for (const auto& field : fields) result.insert(field.first, field.second);
  return c10::impl::toTypedDict<std::string, c10::IValue>(result);
}

// A real derived copy retains the events that NCCL's event cache otherwise
// recycles. Keep this PyTorch-specific access out of the neutral simulator.
struct Events : c10d::ProcessGroupNCCL::WorkNCCL {
  explicit Events(const WorkNCCL& work) : WorkNCCL(work) {}
  auto start() const { return ncclStartEvent_; }
  auto end() const { return ncclEndEvent_; }
};

struct Record {
  std::string operation, group;
  std::shared_ptr<at::cuda::CUDAEvent> start, end;
};

struct Session {
  at::cuda::CUDAEvent anchor{cudaEventDefault};
  cudaEvent_t borrowed_anchor = nullptr;
  at::CallbackHandle callback = at::INVALID_CALLBACK_HANDLE;
  std::vector<std::unique_ptr<Record>> records;
  size_t capacity;
  int64_t errors = 0;
};

thread_local std::unique_ptr<Session> current;

struct Observation : at::ObserverContext {
  Record* record;
  explicit Observation(Record* value) : record(value) {}
};

bool device_value(const c10::IValue& value) {
  if (value.isTensor()) return value.toTensor().is_cuda();
  if (value.isList()) {
    for (const auto& item : value.toListRef()) {
      if (device_value(item)) return true;
    }
  }
  return false;
}

std::unique_ptr<at::ObserverContext> before(const at::RecordFunction& fn) {
  const std::string_view name(fn.name());
  if (!current || name.rfind("c10d::", 0) != 0) return nullptr;
  try {
    c10::intrusive_ptr<c10d::ProcessGroup> group;
    bool on_device = false;
    for (const auto& value : fn.inputs()) {
      on_device |= device_value(value);
      if (value.isCustomClass() &&
          value.type() == c10::getCustomClassType<c10::intrusive_ptr<c10d::ProcessGroup>>()) {
        group = value.toCustomClass<c10d::ProcessGroup>();
      }
    }
    if (!group || !on_device) return nullptr;
    auto backend = group->getBackend(c10::DeviceType::CUDA);
    if (!dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get())) return nullptr;
    if (current->records.size() == current->capacity) {
      ++current->errors;
      return nullptr;
    }
    backend->enableCollectivesTiming();
    auto record = std::make_unique<Record>();
    record->operation = name.substr(6);
    record->group = group->getGroupName();
    auto* pointer = record.get();
    current->records.push_back(std::move(record));
    return std::make_unique<Observation>(pointer);
  } catch (...) {
    ++current->errors;
    return nullptr;
  }
}

void after(const at::RecordFunction& fn, at::ObserverContext* context) {
  auto* observation = static_cast<Observation*>(context);
  if (!observation) return;
  try {
    for (const auto& value : fn.outputs()) {
      if (!value.isCustomClass() ||
          value.type() != c10::getCustomClassType<c10::intrusive_ptr<c10d::Work>>()) continue;
      auto work = value.toCustomClass<c10d::Work>();
      auto* nccl = dynamic_cast<c10d::ProcessGroupNCCL::WorkNCCL*>(work.get());
      if (nccl) {
        const Events events(*nccl);
        observation->record->start = events.start();
        observation->record->end = events.end();
      }
      return;
    }
  } catch (...) {
    ++current->errors;
  }
}

void begin(int64_t capacity, int64_t anchor) {
  TORCH_CHECK(capacity > 0, "collective profile capacity must be positive");
  TORCH_CHECK(!current, "a task collective profile is already active");
  auto owner = std::make_unique<Session>();
  owner->capacity = static_cast<size_t>(capacity);
  owner->records.reserve(owner->capacity);
  if (anchor) {
    owner->borrowed_anchor = reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(anchor));
  } else {
    owner->anchor.record(c10::cuda::getCurrentCUDAStream());
  }
  owner->callback = at::addThreadLocalCallback(
      at::RecordFunctionCallback(&before, &after).needsInputs(true).needsOutputs(true));
  current = std::move(owner);
}

double elapsed(const Session& owner, const at::cuda::CUDAEvent& event) {
  if (!owner.borrowed_anchor) return owner.anchor.elapsed_time(event);
  float milliseconds = 0;
  C10_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, owner.borrowed_anchor, event.event()));
  return milliseconds;
}

Dict end() {
  TORCH_CHECK(current, "no task collective profile is active");
  auto owner = std::move(current);
  at::removeCallback(owner->callback);
  at::cuda::CUDAEvent finished{cudaEventDefault};
  finished.record(c10::cuda::getCurrentCUDAStream());
  finished.synchronize();
  c10::impl::GenericList records(c10::AnyType::get());
  for (const auto& record : owner->records) {
    if (!record->start || !record->end) {
      ++owner->errors;
      continue;
    }
    record->end->synchronize();
    records.push_back(dictionary({
        {"operation", record->operation}, {"group", record->group},
        {"start_ms", elapsed(*owner, *record->start)},
        {"end_ms", elapsed(*owner, *record->end)}}));
  }
  return dictionary({
      {"duration_ms", elapsed(*owner, finished)},
      {"errors", owner->errors}, {"records", records}});
}
} // namespace

TORCH_LIBRARY_FRAGMENT(shadowspill, m) {
  m.def("_collective_profile_begin(int capacity, int anchor=0) -> ()", &begin);
  m.def("_collective_profile_end() -> Dict(str, Any)", &end);
}
