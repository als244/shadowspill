"""Summed token objectives with explicit validity and no hidden normalization."""

import torch


def cross_entropy(logprobs, valid, weights):
    """Weighted negative target logprobs, accumulated in FP32."""
    current = torch.where(valid, logprobs.float(), 0.0)
    weight = torch.where(valid, weights.float(), 0.0)
    return (-weight * current).sum()


def clipped_policy(
    logprobs,
    valid,
    weights,
    old_logprobs,
    advantages,
    clip_ratio,
    *,
    reference_logprobs=None,
    reference_beta=None,
):
    """PPO clipped surrogate, optionally with the prepared sampled k3 penalty.

    Positive advantages cap the ratio above; negative advantages floor it below.
    Applying those *same* surrogate bounds before exponentiation avoids overflow
    in an unused clipped branch. The unbounded branch is never clamped. Its
    overflow, or overflow in an enabled reference penalty, remains nonfinite.
    """
    current = torch.where(valid, logprobs.float(), 0.0)
    old = torch.where(valid, old_logprobs.float(), 0.0)
    advantage = torch.where(valid, advantages.float(), 0.0)
    weight = torch.where(valid, weights.float(), 0.0)
    difference = current - old
    clip = clip_ratio.float()
    upper, lower = torch.log1p(clip), torch.log1p(-clip)
    bounded = torch.where(
        advantage >= 0,
        torch.where(difference > upper, upper, difference),
        torch.where(difference < lower, lower, difference),
    )
    term = -torch.exp(bounded) * advantage
    if reference_logprobs is not None:
        if reference_beta is None:
            raise ValueError("reference logprobs require a reference coefficient")
        beta = reference_beta.float()
        reference = torch.where(valid, reference_logprobs.float(), 0.0)
        # Selecting a neutral difference first keeps expm1 and its derivative
        # finite for beta=0, even when the unused difference would overflow.
        difference = torch.where(beta != 0, reference - current, 0.0)
        term = term + beta * (torch.expm1(difference) - difference)
    elif reference_beta is not None:
        raise ValueError("reference coefficient requires prepared reference inputs")
    return (weight * term).sum()
