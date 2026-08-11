from __future__ import annotations

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from shadowspill.pytorch import TensorSpec
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model


class _AliasedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        self.left = nn.Parameter(base)
        self.right = nn.Parameter(base.view(3, 8))
        self.tied = self.left
        self.register_buffer("window", base[1:5])


def test_fake_model_preserves_registration_and_storage_aliases() -> None:
    model = _AliasedModule()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    assert isinstance(replica.left, FakeTensor)
    assert replica.left.device.type == "cuda"
    assert replica.left is replica.tied
    storage = replica.left.untyped_storage()._cdata
    assert replica.right.untyped_storage()._cdata == storage
    assert replica.window.untyped_storage()._cdata == storage
    assert replica.window.storage_offset() == 4


def test_fake_inputs_retain_nested_static_values_and_exact_strides() -> None:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    values = [TensorSpec((3, 2), torch.bfloat16, stride=(1, 3)), {"length": 3}]
    converted = fake_cuda_inputs(values, mode)
    assert isinstance(converted[0], FakeTensor)
    assert tuple(converted[0].stride()) == (1, 3)
    assert converted[1] == {"length": 3}


def test_fake_inputs_preserve_representative_view_aliases() -> None:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    base = torch.arange(16, dtype=torch.float32)
    converted = fake_cuda_inputs([base[:8], base[2:10]], mode)
    assert (
        converted[0].untyped_storage()._cdata == converted[1].untyped_storage()._cdata
    )
    assert converted[0].storage_offset() == 0
    assert converted[1].storage_offset() == 2
