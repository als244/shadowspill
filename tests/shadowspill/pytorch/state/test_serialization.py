"""Quantized checkpoints contain physical tensors and use existing model types."""

from __future__ import annotations

import pytest
import torch

from shadowspill.pytorch.state.serialization import (
    decode_tensor_state,
    encode_tensor_state,
)
from shadowspill.training import Trainer, checkpoints
from shadowspill.training._model import initialize_model
from shadowspill.training.backends import PyTorch
from tests.shadowspill.pytorch.state.representations import ScaledWeight, model


def test_weights_only_checkpoint_preserves_nested_component_bytes(tmp_path):
    source = model()
    source.weight = torch.nn.Parameter(
        ScaledWeight(source.weight.detach(), torch.tensor(2.0))
    )
    source.tied = source.weight
    packed = encode_tensor_state(source.state_dict())
    path = tmp_path / "weights.pt"
    torch.save(packed, path)
    loaded = torch.load(path, weights_only=True, mmap=True)
    actual = decode_tensor_state(loaded, source.state_dict())
    assert isinstance(actual["weight"].payload, ScaledWeight)
    assert (
        actual["weight"].payload.payload.data_ptr()
        == actual["tied"].payload.payload.data_ptr()
    )
    torch.testing.assert_close(
        actual["weight"].dense(), source.weight.dense(), rtol=0, atol=0
    )
    assert (
        actual["weight"].payload.payload.data_ptr()
        != source.weight.payload.payload.data_ptr()
    )
    wrong = model()
    with pytest.raises(ValueError, match="representation differs"):
        decode_tensor_state(loaded, wrong.state_dict())


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_wrapper_checkpoint_publishes_or_preserves_compute(tmp_path, weights):
    source = model()
    master = torch.nn.Parameter(source.weight.dense().detach() + 0.004)
    optimizer = torch.optim.SGD([master], lr=0.01)
    path = tmp_path / "state.pt"
    checkpoints.save(path, source, optimizer, 3, {"weight": master}, weights=weights)
    state = torch.load(path, weights_only=True)
    target = model()
    restored_master = torch.nn.Parameter(torch.zeros_like(master))
    restored_optimizer = torch.optim.SGD([restored_master], lr=0.01)
    checkpoints.load(state, target, restored_optimizer, {"weight": restored_master})
    expected = (
        master.detach() if weights == "master" else source.weight.dense().detach()
    )
    torch.testing.assert_close(restored_master, expected, rtol=0, atol=0)
    reference = source.weight.detach().clone()
    if weights == "master":
        reference.copy_(master.detach())
    torch.testing.assert_close(target.weight.payload, reference.payload, rtol=0, atol=0)
    torch.testing.assert_close(target.weight.scale, reference.scale, rtol=0, atol=0)
    with torch.device("meta"):
        initialized = model()
    initialize_model(initialized, state=state["model"])
    torch.testing.assert_close(
        initialized.weight.payload, reference.payload, rtol=0, atol=0
    )


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_cpu_trainer_updates_logical_master_and_resumes(tmp_path, weights):
    sample = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24

    def trainer():
        return Trainer(
            model(),
            objective=lambda net, data: net(data).square().mean(),
            optimizer=torch.optim.SGD,
            optimizer_args={"lr": 0.01},
            master_dtype=torch.float32,
            backend=PyTorch(compile=False, device="cpu"),
        )

    with trainer() as original:
        original.prepare(sample)
        before = original.model.weight.payload.clone()
        for _ in range(3):
            original.step(sample)
        assert "weight" in original._execution.weights.masters
        assert not torch.equal(before, original.model.weight.payload)
        checkpoint = original.save(tmp_path / "checkpoint", weights=weights)
        expected = original.step(sample).loss
    with trainer() as resumed:
        resumed.prepare(sample, checkpoint=checkpoint)
        actual = resumed.step(sample).loss
        assert resumed.step_count == 4
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
