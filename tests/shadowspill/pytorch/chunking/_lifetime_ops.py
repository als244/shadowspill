"""Shared dispatcher fixtures; register once under one module identity."""

import torch


@torch.library.custom_op("lifetime_copy_test::layout_observer", mutates_args=())
def layout_observer(x: torch.Tensor) -> torch.Tensor:
    return (x + x.stride(0)).clone(memory_format=torch.contiguous_format)


@layout_observer.register_fake
def _(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("lifetime_copy_test::logical_consumer", mutates_args=())
def logical_consumer(x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return (x * 2 + offset).clone(memory_format=torch.contiguous_format)


@logical_consumer.register_fake
def _(x, offset):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
