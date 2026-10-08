"""Package-owned chunked language-model head loss implementation."""

from __future__ import annotations

import torch

from ...dispatch import logical_costs as logical
from ...dispatch.context import weight_gradient_dtype
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...kernels.cross_entropy import cross_entropy_fwd_bwd
from ...kernels.head import default_head_chunk_size
from ...kernels.matmul import add_product_


def _common_support(hidden, head_weight, targets):
    if not all(
        isinstance(value, torch.Tensor) for value in (hidden, head_weight, targets)
    ):
        return SupportResult.no("head inputs must be tensors")
    if hidden.ndim < 1 or head_weight.ndim != 2:
        return SupportResult.no(
            "hidden must have rows and head_weight must be rank two"
        )
    if head_weight.shape[-1] != hidden.shape[-1]:
        return SupportResult.no("hidden and head widths are incompatible")
    if targets.numel() != hidden.numel() // hidden.shape[-1]:
        return SupportResult.no("targets must contain one label per hidden row")
    if hidden.device != head_weight.device or hidden.device != targets.device:
        return SupportResult.no("hidden, head_weight, and targets must share a device")
    return SupportResult.yes()


def _supports(
    hidden,
    head_weight,
    targets=None,
    *,
    surface,
    chunk_size=None,
    valid_rows=None,
    reduction="mean",
    _entrypoint="forward",
    **_kwargs,
):
    del surface, chunk_size, valid_rows, reduction
    if _entrypoint == "backward":
        if not all(isinstance(value, torch.Tensor) for value in (hidden, head_weight)):
            return SupportResult.no("explicit backward seeds must be tensors")
        return SupportResult.yes()
    return _common_support(hidden, head_weight, targets)


def _policy(hidden, head_weight, chunk_size, valid_rows, reduction):
    """The chunk of rows per logits block, and what the summed cross entropy
    is divided by: the rows, ``valid_rows`` of them, or one for a sum."""
    rows = hidden.numel() // hidden.shape[-1]
    chunk = (
        default_head_chunk_size(head_weight.shape[0])
        if chunk_size is None
        else int(chunk_size)
    )
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
    if reduction == "sum":
        if valid_rows is not None:
            raise ValueError("valid_rows names the mean's denominator; a sum has none")
        return chunk, 1
    normalizer = rows if valid_rows is None else int(valid_rows)
    if not 0 < normalizer <= rows:
        raise ValueError(f"valid_rows must be in [1, {rows}]; got {normalizer}")
    return chunk, normalizer


def forward(
    hidden,
    head_weight,
    targets,
    *,
    chunk_size=None,
    valid_rows=None,
    reduction="mean",
    weight_grad_dtype=None,
):
    """Return loss and seed-one hidden/head VJPs with bounded logits.

    The head's VJP is summed over the chunks at ``weight_grad_dtype``, each
    chunk's product added as the multiply writes it; ``None`` keeps it at the
    head's own dtype, adding each chunk's product rounded to it.
    """
    chunk, normalizer = _policy(hidden, head_weight, chunk_size, valid_rows, reduction)
    return _run(hidden, head_weight, targets, chunk, normalizer, weight_grad_dtype)


def _run(hidden, head_weight, targets, chunk, normalizer, weight_grad_dtype, need_hidden_grad=True, need_head_grad=True):
    """The chunked pass itself: the summed cross entropy over ``normalizer``
    and both seed-one gradients, ``chunk`` rows of logits at a time."""
    rows = hidden.numel() // hidden.shape[-1]
    with torch.no_grad():
        hidden_2d = hidden.reshape(rows, hidden.shape[-1])
        targets_1d = targets.reshape(rows)
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        grad_hidden = torch.empty_like(hidden_2d) if need_hidden_grad else None
        grad_head = torch.zeros_like(head_weight, dtype=weight_grad_dtype) if need_head_grad else None
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            hidden_chunk = hidden_2d[start:stop]
            logits = hidden_chunk @ head_weight.T
            partial, grad_logits = cross_entropy_fwd_bwd(
                logits,
                targets_1d[start:stop],
                total_rows=normalizer,
            )
            loss += partial
            if grad_head is not None:
                if weight_grad_dtype is None:
                    grad_head.add_((grad_logits.T @ hidden_chunk).to(grad_head.dtype))
                else:
                    add_product_(grad_head, grad_logits.T, hidden_chunk)
            if grad_hidden is not None:
                grad_hidden[start:stop].copy_(grad_logits @ head_weight)
    return loss, None if grad_hidden is None else grad_hidden.reshape_as(hidden), grad_head


def backward(grad_loss, grad_hidden_seed, grad_head_seed):
    """Scale immutable seed-one VJPs by an arbitrary scalar cotangent."""
    with torch.no_grad():
        return (
            None if grad_hidden_seed is None else grad_hidden_seed * grad_loss.to(grad_hidden_seed.dtype),
            None if grad_head_seed is None else grad_head_seed * grad_loss.to(grad_head_seed.dtype),
        )


@torch.library.custom_op(
    "mlops::head_loss_builtin_chunked_fwd",
    mutates_args=(),
    schema="(Tensor hidden, Tensor head_weight, Tensor targets, SymInt chunk_size, SymInt normalizer, ScalarType? weight_grad_dtype, bool need_hidden_grad=True, bool need_head_grad=True) -> (Tensor, Tensor?, Tensor?)",
)
def _forward_op(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    normalizer: int,
    weight_grad_dtype: torch.dtype | None,
    need_hidden_grad: bool = True,
    need_head_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    loss, grad_hidden, grad_head = _run(
        hidden,
        head_weight,
        targets,
        chunk_size,
        normalizer,
        weight_grad_dtype,
        need_hidden_grad,
        need_head_grad,
    )
    return loss, None if grad_hidden is None else grad_hidden.reshape(-1, hidden.shape[-1]), grad_head


@_forward_op.register_fake
def _forward_fake(
    hidden, head_weight, targets, chunk_size, normalizer, weight_grad_dtype, need_hidden_grad=True, need_head_grad=True
):
    del targets, chunk_size, normalizer
    rows = hidden.numel() // hidden.shape[-1]
    return (
        hidden.new_empty((), dtype=torch.float32),
        hidden.new_empty((rows, hidden.shape[-1])) if need_hidden_grad else None,
        torch.empty_like(head_weight, dtype=weight_grad_dtype) if need_head_grad else None,
    )


@flop_formula(_forward_op)
def _forward_flops(hidden, head_weight, targets, *_rest, out_val=None, **_kwargs):
    del out_val
    return logical.head_loss(
        hidden, head_weight, targets, entrypoint="forward"
    ).logical_flops


def _setup_context(ctx, inputs, output):
    hidden, _head_weight, _targets, *_policy_args = inputs
    _loss, grad_hidden, grad_head = output
    ctx.save_for_backward(grad_hidden, grad_head)
    ctx.hidden_shape = hidden.shape
    ctx.mark_non_differentiable(*(value for value in (grad_hidden, grad_head) if value is not None))


def _autograd_backward(ctx, grad_loss, _grad_hidden_output, _grad_head_output):
    grad_hidden, grad_head = ctx.saved_tensors
    grad_hidden, grad_head = backward(grad_loss, grad_hidden, grad_head)
    return (None if grad_hidden is None else grad_hidden.reshape(ctx.hidden_shape)), grad_head, None, None, None, None, None, None


_forward_op.register_autograd(
    _autograd_backward,
    setup_context=_setup_context,
)


def apply(
    hidden,
    head_weight,
    targets,
    *,
    chunk_size=None,
    valid_rows=None,
    reduction="mean",
):
    """Apply the autograd-enabled bounded-logits head loss."""
    chunk, normalizer = _policy(hidden, head_weight, chunk_size, valid_rows, reduction)
    loss, *_seeds = _forward_op(
        hidden,
        head_weight,
        targets,
        chunk,
        normalizer,
        weight_gradient_dtype(),
        torch.is_grad_enabled() and hidden.requires_grad,
        torch.is_grad_enabled() and head_weight.requires_grad,
    )
    return loss


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="head_loss",
        implementation_id="builtin.head_loss.chunked",
        provider="builtin",
        priority=100,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
    )
)


__all__ = ["IMPLEMENTATION", "apply", "backward", "forward"]
