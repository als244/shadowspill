"""Operators with explicit CPU results, restored independently of the model."""

import torch


@torch.library.custom_op("shadowspill_fixture::host_control", mutates_args=())
def host_control(x: torch.Tensor) -> torch.Tensor:
    return (x.sum(1) > 0).to(device="cpu", dtype=torch.int64)


@host_control.register_fake
def fake_control(x):
    return torch.empty(x.shape[0], dtype=torch.int64, device="cpu")


@torch.library.custom_op("shadowspill_fixture::read_host_control", mutates_args=())
def read_host_control(x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    assert control.device.type == "cpu"
    return x * (control.to(x.device)[:, None] + 1)


@read_host_control.register_fake
def fake_read(x, control):
    assert control.device.type == "cpu"
    return torch.empty_like(x)
