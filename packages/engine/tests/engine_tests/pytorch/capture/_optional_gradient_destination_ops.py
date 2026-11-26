"""Linear VJP with an optional bias gradient and a mutable weight destination."""

import torch


@torch.library.custom_op(
    "gradient_destination_optional_test::fresh",
    mutates_args=(),
    schema=(
        "(Tensor x, Tensor weight, Tensor tangent, bool bias=False) "
        "-> (Tensor, Tensor?, Tensor)"
    ),
)
def fresh(x, weight, tangent, bias=False):
    return tangent @ weight.T, tangent.sum(0) if bias else None, x.T @ tangent


@fresh.register_fake
def _fresh_fake(x, weight, tangent, bias=False):
    return (
        torch.empty_like(x),
        tangent.new_empty(tangent.shape[1]) if bias else None,
        torch.empty_like(weight),
    )


@torch.library.custom_op(
    "gradient_destination_optional_test::accumulating",
    mutates_args=("destinations",),
    schema=(
        "(Tensor x, Tensor weight, Tensor tangent, bool bias, "
        "Tensor(a!)[] destinations, int[] output_indices, int contract_version, "
        "bool drop_aux=False) -> (Tensor, Tensor, Tensor)"
    ),
)
def accumulating(
    x,
    weight,
    tangent,
    bias,
    destinations,
    output_indices,
    contract_version,
    drop_aux=False,
):
    assert output_indices == [2] and contract_version == 1
    torch.addmm(destinations[0], x.T, tangent, out=destinations[0])
    auxiliary = tangent.sum(0) if bias and not drop_aux else x.new_empty(0)
    return tangent @ weight.T, auxiliary, x.new_empty(0)


@accumulating.register_fake
def _accumulating_fake(
    x,
    weight,
    tangent,
    bias,
    destinations,
    output_indices,
    contract_version,
    drop_aux=False,
):
    assert output_indices == [2] and contract_version == 1
    auxiliary = (
        tangent.new_empty(tangent.shape[1]) if bias and not drop_aux else x.new_empty(0)
    )
    return torch.empty_like(x), auxiliary, x.new_empty(0)
