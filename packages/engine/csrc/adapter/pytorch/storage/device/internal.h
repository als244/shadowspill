#ifndef SHADOWSPILL_PYTORCH_DEVICE_STORAGE_INTERNAL_H
#define SHADOWSPILL_PYTORCH_DEVICE_STORAGE_INTERNAL_H

#include "../internal.h"
#include <ATen/ATen.h>
#include <torch/library.h>

namespace shadowspill_pytorch_storage {

void dematerialize_storages(at::TensorList tensors);
void rebind_replacement_views(
    at::TensorList replacement_tensors,
    at::IntArrayRef target_indices,
    at::TensorList adopted_tensors,
    at::IntArrayRef publication_ordinals,
    int64_t task_handle);

}  // namespace shadowspill_pytorch_storage

#endif
