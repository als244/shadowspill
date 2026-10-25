"""Bounded custom operation, fake tensors and autograd for the reference recurrence."""

from __future__ import annotations

from typing import cast

import torch

from .reference import _delta_rule_backward_reference, _delta_rule_reference


@torch.library.custom_op("shadowspill_models::qwen35_delta_rule", mutates_args=())
def _delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference recurrence behind a bounded operator contract."""

    return _delta_rule_reference(query, key, value, beta, decay)


@_delta_rule.register_fake
def _delta_rule_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    del key, beta, decay
    return value.new_empty((*query.shape[:2], *value.shape[2:]))


@torch.library.custom_op(
    "shadowspill_models::qwen35_delta_rule_backward", mutates_args=()
)
def _delta_rule_backward(
    gradient: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _delta_rule_backward_reference(gradient, query, key, value, beta, decay)


@_delta_rule_backward.register_fake
def _delta_rule_backward_fake(
    gradient: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del gradient
    return (
        torch.empty(query.shape, dtype=query.dtype, device=query.device),
        torch.empty(key.shape, dtype=key.dtype, device=key.device),
        torch.empty(value.shape, dtype=value.dtype, device=value.device),
        torch.empty(beta.shape, dtype=beta.dtype, device=beta.device),
        torch.empty(decay.shape, dtype=decay.dtype, device=decay.device),
    )


def _delta_rule_setup(
    ctx: object,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    output: torch.Tensor,
) -> None:
    del output
    ctx.save_for_backward(*inputs)  # type: ignore[attr-defined]


def _delta_rule_autograd(
    ctx: object, gradient: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    saved = cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        ctx.saved_tensors,  # type: ignore[attr-defined]
    )
    return cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        _delta_rule_backward(gradient, *saved),
    )


_delta_rule.register_autograd(
    _delta_rule_autograd,
    setup_context=_delta_rule_setup,
)
