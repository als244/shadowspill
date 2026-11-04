"""Importable opaque VJP provider; execution needs no capture/build modules."""

from collections import Counter

import torch

CALLS = Counter()


def interleave(b):
    return b.view(2, -1, b.shape[-1]).transpose(0, 1).reshape_as(b)


def _compute(x, a, b, tangent, scale, destinations, indices, canonical):
    CALLS["calls"] += 1
    hidden = x @ a.T
    hidden_grad = (tangent @ interleave(b)) * scale
    dx = hidden_grad @ a
    running = dict(zip(indices, destinations, strict=True))
    da = running.get(1)
    if da is None:
        CALLS["fresh_parameter_buffers"] += 1
        da = torch.empty_like(a)
    torch.addmm(da, hidden_grad.T, x, beta=int(1 in running), out=da)
    db = running.get(2)
    if db is None:
        CALLS["fresh_parameter_buffers"] += 1
        db = torch.empty_like(b)
    if 2 in running and canonical:
        width = b.shape[0] // 2
        for index in range(2):
            destination = db[index * width : (index + 1) * width]
            torch.addmm(
                destination,
                tangent[:, index::2].T,
                hidden,
                alpha=scale,
                out=destination,
            )
    else:
        torch.addmm(db, tangent.T, hidden, beta=int(2 in running), out=db)
    return (
        dx,
        x.new_empty(0) if 1 in running else da,
        x.new_empty(0) if 2 in running else db,
    )


@torch.library.custom_op("gradient_destination_test::fresh", mutates_args=())
def fresh(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tangent: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _compute(x, a, b, tangent, scale, [], [], False)


@fresh.register_fake
def _fresh_fake(x, a, b, tangent, scale):
    return torch.empty_like(x), torch.empty_like(a), torch.empty_like(b)


@torch.library.custom_op(
    "gradient_destination_test::accumulating",
    mutates_args=("destinations",),
    schema="(Tensor x, Tensor a, Tensor b, Tensor tangent, float scale, "
    "Tensor(a!)[] destinations, int[] output_indices, int contract_version, "
    "bool canonical_b=False) -> (Tensor, Tensor, Tensor)",
)
def accumulating(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tangent: torch.Tensor,
    scale: float,
    destinations: list[torch.Tensor],
    output_indices: list[int],
    contract_version: int,
    canonical_b: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert contract_version in (1, 2)
    return _compute(x, a, b, tangent, scale, destinations, output_indices, canonical_b)


@accumulating.register_fake
def _accumulating_fake(
    x,
    a,
    b,
    tangent,
    scale,
    destinations,
    output_indices,
    contract_version,
    canonical_b=False,
):
    return (
        torch.empty_like(x),
        x.new_empty(0) if 1 in output_indices else torch.empty_like(a),
        x.new_empty(0) if 2 in output_indices else torch.empty_like(b),
    )


def function(x, a, b, tangent):
    dx, da, db = fresh(x, a, b, tangent, 1.75)
    canonical = db.view(-1, 2, b.shape[-1]).transpose(0, 1).reshape_as(b) * 1.75
    return dx, da, canonical


@torch.library.custom_op("gradient_destination_test::projection", mutates_args=())
def projection(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((x @ a.T) @ interleave(b).T) * 1.75


@projection.register_fake
def _projection_fake(x, a, b):
    return x.new_empty((x.shape[0], b.shape[0]))


def _projection_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _projection_backward(ctx, tangent):
    return function(*ctx.saved_tensors, tangent)


projection.register_autograd(_projection_backward, setup_context=_projection_context)
