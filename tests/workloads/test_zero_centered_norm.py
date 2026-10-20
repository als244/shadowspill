"""The mlops implementations' zero-centered RMSNorm is the PyTorch
implementation's arithmetic, now through mlops's fused operation."""

from types import SimpleNamespace

import pytest
import torch

from workloads.models.shared import decoder, norms
from workloads.models.shared.qwen_moe import attention

EPSILON = 1e-6


def pair(width, dtype, device, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(5, 9, width, generator=generator).to(device=device, dtype=dtype)
    weight = 0.1 * torch.randn(width, generator=generator)
    grad = torch.randn(5, 9, width, generator=generator).to(device=device, dtype=dtype)
    return x, weight.to(device), grad


def run(module, x, grad):
    x = x.clone().requires_grad_(True)
    output = module(x)
    output.backward(grad)
    return output.detach(), x.grad, module.weight.grad


def norms_under_test(width, weight, device):
    reference = decoder.RMSNorm(width, epsilon=EPSILON, zero_centered=True)
    dense = norms.RMSNorm(width, epsilon=EPSILON, zero_centered=True)
    mixture = attention.RMSNorm(
        width, SimpleNamespace(norm_epsilon=EPSILON, zero_centered_norm=True)
    )
    for module in (reference, dense, mixture):
        with torch.no_grad():
            module.weight.copy_(weight)
        module.to(device)
    return reference, dense, mixture


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_on_the_host_the_mlops_norms_are_the_pytorch_norm_bit_for_bit(dtype):
    x, weight, grad = pair(64, dtype, "cpu")
    reference, dense, mixture = norms_under_test(64, weight, "cpu")
    expected = run(reference, x, grad)
    for module in (dense, mixture):
        for got, want in zip(run(module, x, grad), expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_the_device_the_fused_norms_match_the_pytorch_norm():
    x, weight, grad = pair(2048, torch.bfloat16, "cuda")
    reference, dense, mixture = norms_under_test(2048, weight, "cuda")
    output, grad_x, grad_weight = run(reference, x, grad)
    for module in (dense, mixture):
        fused = run(module, x, grad)
        # One rounding of the same FP32 value, and FP32 weight gradients.
        torch.testing.assert_close(
            fused[0].float(), output.float(), rtol=8e-3, atol=1e-6
        )
        torch.testing.assert_close(
            fused[1].float(), grad_x.float(), rtol=1e-2, atol=1e-3
        )
        torch.testing.assert_close(fused[2], grad_weight, rtol=1e-4, atol=1e-4)
