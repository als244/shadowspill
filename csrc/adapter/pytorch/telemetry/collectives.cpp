/* Trace c10d operations on both the task thread and the transfer worker.
 * Observe the dispatcher, rather than replacing collectives or their streams.
 * Only an explicitly armed runtime trace installs this global callback. */
#include "../storage/internal.h"

#include <ATen/record_function.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/library.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

using Dict = c10::Dict<std::string, c10::IValue>;

Dict dictionary(std::initializer_list<std::pair<std::string, c10::IValue>> items) {
  c10::impl::GenericDict result(c10::StringType::get(), c10::AnyType::get());
  for (const auto& item : items) {
    result.insert(item.first, item.second);
  }
  return c10::impl::toTypedDict<std::string, c10::IValue>(result);
}

// WorkNCCL's public copy constructor shares its timing events (the watchdog
// uses the same constructor). The adapter reads the protected event handles
// through a real derived copy, never by downcasting a base object to a fake
// subclass. Isolate this version-sensitive PyTorch detail here. Retaining the
// shared events also prevents NCCL's event cache from recycling them early.
struct NcclEvents : c10d::ProcessGroupNCCL::WorkNCCL {
  explicit NcclEvents(const WorkNCCL& work) : WorkNCCL(work) {}
  auto start() const { return ncclStartEvent_; }
  auto stop() const { return ncclEndEvent_; }
};

struct Record {
  ShadowSpillTraceContext context{};
  std::string operation;
  std::string group;
  int rank = 0;
  int size = 0;
  uint64_t sequence = 0;
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  std::shared_ptr<at::cuda::CUDAEvent> start;
  std::shared_ptr<at::cuda::CUDAEvent> stop;
};

struct HostCopyRecord {
  ShadowSpillTraceContext context{};
  uint64_t bytes = 0;
  std::shared_ptr<at::cuda::CUDAEvent> start;
  std::shared_ptr<at::cuda::CUDAEvent> stop;
};

struct Session {
  ShadowSpillRuntime* runtime;
  size_t capacity;
  std::atomic<uint64_t> dropped{0};
  std::atomic<uint64_t> errors{0};
  std::atomic<uint64_t> copy_dropped{0};
  std::atomic<uint64_t> copy_errors{0};
  std::mutex mutex;
  std::vector<std::unique_ptr<Record>> records;
  std::vector<std::unique_ptr<HostCopyRecord>> copies;
  at::CallbackHandle callback = at::INVALID_CALLBACK_HANDLE;
};

// Begin/end happen only at an idle invocation boundary. Callback contexts keep
// the session alive until the dispatcher has returned, including error paths.
std::shared_ptr<Session> session;

struct Observation : at::ObserverContext {
  std::shared_ptr<Session> owner;
  Record* record;
  Observation(std::shared_ptr<Session> owner_, Record* record_)
      : owner(std::move(owner_)), record(record_) {}
};

uint64_t tensor_bytes(const c10::IValue& value) {
  if (value.isTensor()) {
    return value.toTensor().nbytes();
  }
  uint64_t bytes = 0;
  if (value.isList()) {
    for (const auto& item : value.toListRef()) {
      bytes += tensor_bytes(item);
    }
  }
  return bytes;
}

bool has_device_value(const c10::IValue& value) {
  if (value.isTensor()) {
    return value.toTensor().is_cuda();
  }
  if (value.isList()) {
    for (const auto& item : value.toListRef()) {
      if (has_device_value(item)) {
        return true;
      }
    }
  }
  return false;
}

std::unique_ptr<at::ObserverContext> before(const at::RecordFunction& fn) {
  const std::string_view name(fn.name());
  if (name.rfind("c10d::", 0) != 0) {
    return nullptr;
  }
  auto owner = std::atomic_load(&session);
  if (!owner) {
    return nullptr;
  }
  try {
    ShadowSpillTraceContext context{};
    if (!shadowspill_trace_context(owner->runtime, &context)) {
      return nullptr;
    }
    c10::intrusive_ptr<c10d::ProcessGroup> group;
    for (const auto& value : fn.inputs()) {
      if (value.isCustomClass() &&
          value.type() == c10::getCustomClassType<c10::intrusive_ptr<c10d::ProcessGroup>>()) {
        group = value.toCustomClass<c10d::ProcessGroup>();
        break;
      }
    }
    bool device_operation = false;
    for (const auto& value : fn.inputs()) {
      device_operation |= has_device_value(value);
    }
    if (!group || !device_operation) {
      return nullptr; // c10d bookkeeping is not a collective.
    }
    auto backend = group->getBackend(c10::DeviceType::CUDA);
    if (!dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get())) {
      return nullptr;
    }
    // Supported PyTorch switch; no environment mutation and no replacement of
    // user hooks. PyTorch keeps timing enabled for this group's remaining life.
    backend->enableCollectivesTiming();
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (owner->records.size() == owner->capacity) {
      ++owner->dropped;
      return nullptr;
    }
    auto record = std::make_unique<Record>();
    record->context = context;
    record->operation = name.substr(6);
    record->group = group->getGroupName();
    record->rank = group->getRank();
    record->size = group->getSize();
    const auto inputs = fn.inputs();
    if (inputs.size() >= 2 &&
        (name.find("allgather") != name.npos || name.find("reduce_scatter") != name.npos ||
         name.find("alltoall") != name.npos)) {
      record->output_bytes = tensor_bytes(inputs[0]);
      record->input_bytes = tensor_bytes(inputs[1]);
    } else if (!inputs.empty()) {
      record->input_bytes = record->output_bytes = tensor_bytes(inputs[0]);
    }
    auto* pointer = record.get();
    owner->records.push_back(std::move(record));
    return std::make_unique<Observation>(owner, pointer);
  } catch (...) {
    ++owner->errors; // A diagnostic failure must not break training.
    return nullptr;
  }
}

void after(const at::RecordFunction& fn, at::ObserverContext* context) {
  auto* observation = static_cast<Observation*>(context);
  if (!observation) {
    return;
  }
  try {
    for (const auto& value : fn.outputs()) {
      if (!value.isCustomClass() ||
          value.type() != c10::getCustomClassType<c10::intrusive_ptr<c10d::Work>>()) {
        continue;
      }
      auto work = value.toCustomClass<c10d::Work>();
      auto* nccl = dynamic_cast<c10d::ProcessGroupNCCL::WorkNCCL*>(work.get());
      if (nccl) {
        const NcclEvents events(*nccl);
        observation->record->start = events.start();
        observation->record->stop = events.stop();
        observation->record->sequence = work->getSequencenumber();
      }
      return;
    }
  } catch (...) {
    ++observation->owner->errors;
  }
}

void begin(int64_t capacity) {
  TORCH_CHECK(capacity > 0, "collective trace capacity must be positive");
  TORCH_CHECK(!std::atomic_load(&session), "a collective trace is already active");
  auto owner = std::make_shared<Session>();
  owner->runtime = shadowspill_pytorch_bound_runtime();
  TORCH_CHECK(owner->runtime != nullptr, "collective tracing needs a bound runtime");
  owner->capacity = static_cast<size_t>(capacity);
  owner->records.reserve(owner->capacity);
  owner->copies.reserve(owner->capacity);
  std::atomic_store(&session, owner);
  try {
    owner->callback = at::addGlobalCallback(
        at::RecordFunctionCallback(&before, &after).needsInputs(true).needsOutputs(true));
  } catch (...) {
    std::atomic_store(&session, std::shared_ptr<Session>{});
    throw;
  }
}

c10::IValue optional_id(uint64_t value) {
  return value == SHADOWSPILL_RUNTIME_NO_ID ? c10::IValue{} : c10::IValue(static_cast<int64_t>(value));
}

c10::IValue instant(Session& owner, const std::shared_ptr<at::cuda::CUDAEvent>& event) {
  uint64_t ns = 0;
  if (!event || !event->isCreated() || shadowspill_trace_event_instant(
          owner.runtime, reinterpret_cast<ShadowSpillBackendEvent>(event->event()), &ns) != 0) {
    return {};
  }
  return static_cast<int64_t>(ns);
}

Dict end() {
  auto owner = std::atomic_exchange(&session, std::shared_ptr<Session>{});
  if (!owner) {
    return dictionary({});
  }
  at::removeCallback(owner->callback);
  // The caller has drained the runtime and GPU. No host polling timestamps,
  // synchronizations, or lane-stream brackets stand in for NCCL execution.
  c10::impl::GenericList records(c10::AnyType::get());
  uint64_t missing = 0;
  for (const auto& record : owner->records) {
    auto start = instant(*owner, record->start);
    auto stop = instant(*owner, record->stop);
    missing += start.isNone() || stop.isNone();
    records.push_back(dictionary({
        {"operation", record->operation}, {"group", record->group},
        {"group_rank", record->rank}, {"group_size", record->size},
        {"group_sequence", static_cast<int64_t>(record->sequence)},
        {"input_bytes", static_cast<int64_t>(record->input_bytes)},
        {"output_bytes", static_cast<int64_t>(record->output_bytes)},
        {"task_id", optional_id(record->context.task_id)},
        {"object_id", optional_id(record->context.object_id)},
        {"action_ordinal", optional_id(record->context.action_ordinal)},
        {"source", record->context.source == 0 ? "task" : record->context.source == 1 ? "fetch" : "evict"},
        {"issued_at_ns", optional_id(record->context.issued_at_ns)},
        {"started_at_ns", start}, {"finished_at_ns", stop}}));
  }
  c10::impl::GenericList copies(c10::AnyType::get());
  uint64_t copy_missing = 0;
  for (const auto& record : owner->copies) {
    auto start = instant(*owner, record->start);
    auto stop = instant(*owner, record->stop);
    copy_missing += start.isNone() || stop.isNone();
    copies.push_back(dictionary({
        {"bytes", static_cast<int64_t>(record->bytes)},
        {"task_id", optional_id(record->context.task_id)},
        {"object_id", optional_id(record->context.object_id)},
        {"action_ordinal", optional_id(record->context.action_ordinal)},
        {"source", record->context.source == 1 ? "fetch" : "evict"},
        {"issued_at_ns", optional_id(record->context.issued_at_ns)},
        {"started_at_ns", start}, {"finished_at_ns", stop}}));
  }
  return dictionary({{"capacity", static_cast<int64_t>(owner->capacity)},
                     {"dropped", static_cast<int64_t>(owner->dropped.load())},
                     {"errors", static_cast<int64_t>(owner->errors.load())},
                     {"missing_timings", static_cast<int64_t>(missing)},
                     {"records", records},
                     {"host_copies", dictionary({
                         {"capacity", static_cast<int64_t>(owner->capacity)},
                         {"dropped", static_cast<int64_t>(owner->copy_dropped.load())},
                         {"errors", static_cast<int64_t>(owner->copy_errors.load())},
                         {"missing_timings", static_cast<int64_t>(copy_missing)},
                         {"records", copies}})}});
}

} // namespace

int shadowspill_pytorch_copy_host_device(
    const ShadowSpillBackend* backend, ShadowSpillBackendStream stream,
    bool fetch, void* destination, const void* source, uint64_t bytes) {
  auto owner = std::atomic_load(&session);
  HostCopyRecord* record = nullptr;
  if (owner) {
    try {
      ShadowSpillTraceContext context{};
      if (shadowspill_trace_context(owner->runtime, &context) && context.source != 0) {
        std::lock_guard<std::mutex> lock(owner->mutex);
        if (owner->copies.size() == owner->capacity) {
          ++owner->copy_dropped;
        } else {
          auto item = std::make_unique<HostCopyRecord>();
          item->context = context;
          item->bytes = bytes;
          item->start = std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault);
          item->stop = std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault);
          owner->copies.push_back(std::move(item));
          record = owner->copies.back().get();
          record->start->record(c10::cuda::getCurrentCUDAStream());
        }
      }
    } catch (...) {
      ++owner->copy_errors;
    }
  }
  const int status = fetch
      ? backend->copy_host_to_device(backend->state, destination, source, bytes, stream)
      : backend->copy_device_to_host(backend->state, destination, source, bytes, stream);
  if (record) {
    try {
      record->stop->record(c10::cuda::getCurrentCUDAStream());
    } catch (...) {
      ++owner->copy_errors;
    }
    if (status != 0) {
      ++owner->copy_errors;
    }
  }
  return status;
}

TORCH_LIBRARY_FRAGMENT(shadowspill, m) {
  m.def("_collective_trace_begin(int capacity) -> ()", &begin);
  m.def("_collective_trace_end() -> Dict(str, Any)", &end);
}
