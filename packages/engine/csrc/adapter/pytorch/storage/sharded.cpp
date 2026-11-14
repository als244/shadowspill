/* Full execution tensors backed by owned spill byte shards. Collective
 * completion is joined to the lane stream before the runtime publishes ready.
 * Process groups are borrowed from explicit distributed bindings. */
#include "internal.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/library.h>

#include <atomic>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>

namespace {

struct ShardedTransfer {
  ShadowSpillObjectTransfer description{};
  std::atomic<uint64_t> references{1};
  c10::intrusive_ptr<c10d::ProcessGroup> group;
  c10::intrusive_ptr<c10d::ProcessGroup> replicas;
  std::optional<at::ScalarType> gradient_dtype;
  c10::DeviceIndex device;
  int64_t logical_bytes;
  int64_t execution_bytes;
  int64_t shard_bytes;
  int64_t offset;
};

void retain_transfer(void* context) {
  static_cast<ShardedTransfer*>(context)->references.fetch_add(1, std::memory_order_relaxed);
}

void release_transfer(void* context) {
  auto* transfer = static_cast<ShardedTransfer*>(context);
  if (transfer->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete transfer;
  }
}

int enqueue_transfer(
    void* context, const ShadowSpillBackend* backend,
    ShadowSpillBackendStream stream, uint8_t fetch,
    void* destination, const void* source,
    ShadowSpillTransferCopy copy, void* copy_context) {
  (void)backend;
  auto* transfer = static_cast<ShardedTransfer*>(context);
  try {
    const auto cuda_stream = c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(stream), transfer->device);
    const c10::cuda::CUDAStreamGuard stream_guard(cuda_stream);
    if (transfer->gradient_dtype.has_value()) {
      if (fetch) {
        // An already-reduced contribution has exactly one owner per element.
        // Reload it there, leaving every other rank's contribution zero.
        C10_CUDA_CHECK(cudaMemsetAsync(destination, 0, transfer->execution_bytes,
                                      cuda_stream.stream()));
        if (transfer->replicas && transfer->replicas->getRank() != 0) {
          return 0;
        }
        return shadowspill_pytorch_copy_spill(
            copy, copy_context, static_cast<char*>(destination) + transfer->offset,
            source, static_cast<uint64_t>(transfer->shard_bytes));
      }
      auto* mutable_source = const_cast<void*>(source);
      if (transfer->logical_bytes < transfer->execution_bytes) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            static_cast<char*>(mutable_source) + transfer->logical_bytes, 0,
            transfer->execution_bytes - transfer->logical_bytes, cuda_stream.stream()));
      }
      const auto dtype = *transfer->gradient_dtype;
      const int64_t width = c10::elementSize(dtype);
      const auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, transfer->device);
      auto full = at::from_blob(mutable_source, {transfer->execution_bytes / width}, options);
      auto owned = at::from_blob(static_cast<char*>(mutable_source) + transfer->offset,
                                {transfer->shard_bytes / width}, options);
      c10d::ReduceScatterOptions reduce_options;
      reduce_options.reduceOp = c10d::ReduceOp::SUM;
      reduce_options.asyncOp = true;
      transfer->group->reduce_scatter_single(owned, full, reduce_options)->blockCurrentStream();
      if (transfer->replicas && transfer->replicas->getSize() > 1) {
        std::vector<at::Tensor> values{owned};
        c10d::AllreduceOptions options;
        options.reduceOp = c10d::ReduceOp::SUM;
        options.asyncOp = true;
        transfer->replicas->allreduce(values, options)->blockCurrentStream();
      }
      return shadowspill_pytorch_copy_spill(
          copy, copy_context, destination, owned.data_ptr(),
          static_cast<uint64_t>(transfer->shard_bytes));
    }
    if (!fetch) {
      return shadowspill_pytorch_copy_spill(
          copy, copy_context, destination,
          static_cast<const char*>(source) + transfer->offset,
          static_cast<uint64_t>(transfer->shard_bytes));
    }
    auto* owned = static_cast<char*>(destination) + transfer->offset;
    const int status = shadowspill_pytorch_copy_spill(
        copy, copy_context, owned, source,
        static_cast<uint64_t>(transfer->shard_bytes));
    if (status != 0 || transfer->group->getSize() == 1) {
      return status;
    }
    const auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA, transfer->device);
    at::Tensor full = at::from_blob(destination, {transfer->execution_bytes}, options);
    at::Tensor shard = at::from_blob(owned, {transfer->shard_bytes}, options);
    c10d::AllgatherOptions gather_options;
    gather_options.asyncOp = true;
    auto work = transfer->group->all_gather_single(full, shard, gather_options);
    // Unlike wait(), this does not become a host wait when the caller enables
    // TORCH_NCCL_BLOCKING_WAIT. The worker must keep issuing independent work.
    work->blockCurrentStream();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ShadowSpill sharded transfer failed: %s\n", error.what());
    return -1;
  }
}

int64_t create_transfer(const std::string& group_name, int64_t device, int64_t bytes,
                       std::optional<at::ScalarType> gradient_dtype,
                       const std::optional<std::string>& replica_group,
                       int64_t logical_bytes) {
  TORCH_CHECK(device >= 0 && device <= std::numeric_limits<c10::DeviceIndex>::max(),
              "invalid sharded transfer device");
  TORCH_CHECK(bytes >= 0, "negative sharded transfer size");
  auto transfer = std::make_unique<ShardedTransfer>();
  transfer->group = c10d::resolve_process_group(group_name);
  transfer->gradient_dtype = gradient_dtype;
  if (replica_group.has_value()) {
    transfer->replicas = c10d::resolve_process_group(*replica_group);
  }
  TORCH_CHECK(transfer->group->getSize() > 0 && transfer->group->getRank() >= 0,
              "sharded transfer requires membership in its process group");
  const int64_t count = transfer->group->getSize();
  transfer->shard_bytes = bytes / count + (bytes % count != 0);
  TORCH_CHECK(transfer->shard_bytes <= std::numeric_limits<int64_t>::max() / count,
              "padded sharded transfer size overflows");
  transfer->execution_bytes = transfer->shard_bytes * count;
  transfer->logical_bytes = logical_bytes < 0 ? bytes : logical_bytes;
  TORCH_CHECK(transfer->logical_bytes <= transfer->execution_bytes,
              "logical gradient exceeds its padded allocation");
  TORCH_CHECK(!gradient_dtype || transfer->shard_bytes % c10::elementSize(*gradient_dtype) == 0,
              "gradient shards must contain whole elements");
  transfer->offset = transfer->shard_bytes * transfer->group->getRank();
  transfer->device = static_cast<c10::DeviceIndex>(device);
  transfer->description = ShadowSpillObjectTransfer{
      static_cast<uint64_t>(transfer->shard_bytes),
      static_cast<uint8_t>(gradient_dtype ? 2U : 1U), transfer.get(),
      retain_transfer, release_transfer, enqueue_transfer};
  auto* descriptor = &transfer->description;
  (void)transfer.release();
  return reinterpret_cast<int64_t>(descriptor);
}

void close_transfer(int64_t descriptor) {
  TORCH_CHECK(descriptor != 0, "invalid sharded transfer descriptor");
  auto* description = reinterpret_cast<ShadowSpillObjectTransfer*>(descriptor);
  release_transfer(description->context);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(shadowspill, m) {
  m.def("_create_sharded_transfer(str group_name, int device, int bytes, "
        "ScalarType? gradient_dtype=None, str? replica_group=None, int logical_bytes=-1) -> int",
        &create_transfer);
  m.def("_close_sharded_transfer(int descriptor) -> ()", &close_transfer);
}
