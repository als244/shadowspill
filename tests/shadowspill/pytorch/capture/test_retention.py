"""Which forward values the ``save`` graph pair retains, and which it regenerates.

The rule is arithmetic intensity: a forward operator at or under the
threshold is regenerated in the backward, one above it is retained, and a
custom operator no flop formula prices is retained as unknown. The custom
operators here are the test's own, so what each costs is known exactly, and
each test's graph is shaped so that the default partition would answer
differently: the value in question is free to regenerate from what the
backward keeps anyway, so only the rule decides whether it is.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.flop_counter import register_flop_formula

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.capture.retention import (
    MEMORY_BOUND_FLOPS_PER_BYTE,
    OperatorClass,
    RetentionPolicy,
)
from shadowspill.pytorch.graph_pairs import GraphPairVariant, saved_value_footprint

ROWS = 256
WIDTH = 256
BYTES = ROWS * WIDTH * 4


# A memory-bound operator with a formula: one pass over its bytes.
@torch.library.custom_op("shadowspill_retention_test::pass_over", mutates_args=())
def _pass_over(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return value * weight


@_pass_over.register_fake
def _(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


def _pass_over_setup(ctx: object, inputs: tuple, output: object) -> None:
    value, weight = inputs
    ctx.save_for_backward(value, weight)  # type: ignore[attr-defined]


def _pass_over_backward(ctx: object, grad: torch.Tensor) -> tuple:
    value, weight = ctx.saved_tensors  # type: ignore[attr-defined]
    return grad * weight, (grad * value).sum(0)


_pass_over.register_autograd(_pass_over_backward, setup_context=_pass_over_setup)
register_flop_formula(torch.ops.shadowspill_retention_test.pass_over, get_raw=True)(
    lambda value, weight, *_rest, out_val=None, **_kwargs: value.numel()
)


# A compute-bound operator with a formula: a matrix product.
@torch.library.custom_op("shadowspill_retention_test::product", mutates_args=())
def _product(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return value @ weight


@_product.register_fake
def _(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return value.new_empty((value.shape[0], weight.shape[1]))


def _product_setup(ctx: object, inputs: tuple, output: object) -> None:
    value, weight = inputs
    ctx.save_for_backward(value, weight)  # type: ignore[attr-defined]


def _product_backward(ctx: object, grad: torch.Tensor) -> tuple:
    value, weight = ctx.saved_tensors  # type: ignore[attr-defined]
    return grad @ weight.T, value.T @ grad


_product.register_autograd(_product_backward, setup_context=_product_setup)
register_flop_formula(torch.ops.shadowspill_retention_test.product, get_raw=True)(
    lambda value, weight, *_rest, out_val=None, **_kwargs: (
        2 * value.shape[0] * value.shape[1] * weight.shape[1]
    )
)


# The same pointwise square twice over: once priced, once not.
def _square_setup(ctx: object, inputs: tuple, output: object) -> None:
    (value,) = inputs
    ctx.save_for_backward(value)  # type: ignore[attr-defined]


def _square_backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
    (value,) = ctx.saved_tensors  # type: ignore[attr-defined]
    return grad * 2 * value


@torch.library.custom_op("shadowspill_retention_test::priced_square", mutates_args=())
def _priced_square(value: torch.Tensor) -> torch.Tensor:
    return value * value


@_priced_square.register_fake
def _(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


_priced_square.register_autograd(_square_backward, setup_context=_square_setup)
register_flop_formula(torch.ops.shadowspill_retention_test.priced_square, get_raw=True)(
    lambda value, *_rest, out_val=None, **_kwargs: value.numel()
)


@torch.library.custom_op("shadowspill_retention_test::unpriced_square", mutates_args=())
def _unpriced_square(value: torch.Tensor) -> torch.Tensor:
    return value * value


@_unpriced_square.register_fake
def _(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


_unpriced_square.register_autograd(_square_backward, setup_context=_square_setup)


# A formula that cannot price the call it is asked about.
@torch.library.custom_op("shadowspill_retention_test::mispriced", mutates_args=())
def _mispriced(value: torch.Tensor) -> torch.Tensor:
    return value * value


@_mispriced.register_fake
def _(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


_mispriced.register_autograd(_square_backward, setup_context=_square_setup)


def _refuse(*_args: object, **_kwargs: object) -> int:
    raise ValueError("no price for this call")


register_flop_formula(torch.ops.shadowspill_retention_test.mispriced, get_raw=True)(
    _refuse
)


def _inputs(*shapes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return tuple(torch.randn(*shape, requires_grad=True) for shape in shapes)


def _pair(
    forward: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    memory_budget: float = 1.0,
    retention: RetentionPolicy | None = None,
) -> AotGraphPair:
    graph = make_fx(forward)(*inputs)
    return capture_graph_pair(
        graph,
        inputs,
        original_output=forward(*inputs),
        memory_budget=memory_budget,
        retention=retention,
    )


def _internal_bytes(pair: AotGraphPair) -> int:
    return saved_value_footprint(pair).internal_minimum_bytes


def _normalized_product(
    value: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """A pass over the input feeding a product: the product's backward needs
    the pass's result, and the pass's own backward needs the input."""

    return _product(_pass_over(value, scale), weight)


def _product_then_pass(
    value: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """A product feeding a pass: the pass's backward needs the product's
    result, which is free to regenerate from what the product's own backward
    keeps."""

    return _pass_over(_product(value, weight), scale)


def test_a_memory_bound_operator_with_a_formula_is_regenerated() -> None:
    inputs = _inputs((ROWS, WIDTH), (WIDTH,), (WIDTH, WIDTH))
    pair = _pair(_normalized_product, inputs)

    assert pair.retention.memory_bound_flops_per_byte == MEMORY_BOUND_FLOPS_PER_BYTE
    assert pair.retention.regenerated_operators == (
        "shadowspill_retention_test.pass_over.default",
    )
    assert pair.retention.unknown_operators == ()
    assert _internal_bytes(pair) == 0


def test_a_compute_bound_operator_is_retained() -> None:
    inputs = _inputs((ROWS, WIDTH), (WIDTH, WIDTH), (WIDTH,))
    pair = _pair(_product_then_pass, inputs)

    assert pair.retention.regenerated_operators == ()
    assert _internal_bytes(pair) == BYTES


def test_the_threshold_is_the_request_s_to_set() -> None:
    inputs = _inputs((ROWS, WIDTH), (WIDTH, WIDTH), (WIDTH,))
    generous = RetentionPolicy(memory_bound_flops_per_byte=1e6)
    pair = _pair(_product_then_pass, inputs, retention=generous)

    assert pair.retention.memory_bound_flops_per_byte == 1e6
    assert pair.retention.regenerated_operators == (
        "shadowspill_retention_test.product.default",
    )
    assert _internal_bytes(pair) == 0


def test_an_unpriced_custom_operator_is_retained_as_unknown() -> None:
    def unpriced(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return _pass_over(_unpriced_square(value), scale)

    def priced(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return _pass_over(_priced_square(value), scale)

    inputs = _inputs((ROWS, WIDTH), (WIDTH,))
    retained = _pair(unpriced, inputs)
    regenerated = _pair(priced, inputs)

    assert retained.retention.unknown_operators == (
        "shadowspill_retention_test.unpriced_square.default",
    )
    assert retained.retention.regenerated_operators == ()
    assert _internal_bytes(retained) == BYTES
    assert regenerated.retention.unknown_operators == ()
    assert regenerated.retention.regenerated_operators == (
        "shadowspill_retention_test.priced_square.default",
    )
    assert _internal_bytes(regenerated) == 0


def test_a_pointwise_chain_into_a_matrix_product_is_regenerated() -> None:
    def forward(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (torch.sin(value) * 2.0 + 1.0) @ weight

    inputs = _inputs((ROWS, WIDTH), (WIDTH, WIDTH))
    pair = _pair(forward, inputs)

    assert {"aten.sin.default", "aten.mul.Tensor", "aten.add.Tensor"} <= set(
        pair.retention.regenerated_operators
    )
    assert "aten.mm.default" not in pair.retention.regenerated_operators
    assert _internal_bytes(pair) == 0


def test_a_matrix_product_s_result_is_retained() -> None:
    def forward(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.sin(value @ weight)

    inputs = _inputs((ROWS, WIDTH), (WIDTH, WIDTH))
    pair = _pair(forward, inputs)

    assert "aten.mm.default" not in pair.retention.regenerated_operators
    assert _internal_bytes(pair) == BYTES


def test_the_recompute_variant_retains_the_inputs_alone() -> None:
    inputs = _inputs((ROWS, WIDTH), (WIDTH, WIDTH), (WIDTH,))
    pair = _pair(_product_then_pass, inputs, memory_budget=0.0)

    assert pair.retention.regenerated_operators == (
        "shadowspill_retention_test.product.default",
    )
    assert _internal_bytes(pair) == 0


def test_the_policy_classes_a_graph_s_custom_operators() -> None:
    def forward(
        value: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return _unpriced_square(_product(_pass_over(value, scale), weight))

    inputs = _inputs((ROWS, WIDTH), (WIDTH,), (WIDTH, WIDTH))
    # Traced under one fake mode, as an exported stage is.
    graph = make_fx(forward, tracing_mode="fake")(*inputs)
    policy = RetentionPolicy()

    assert policy.operator_classes(graph) == {
        "shadowspill_retention_test.pass_over.default": OperatorClass.MEMORY_BOUND,
        "shadowspill_retention_test.product.default": OperatorClass.COMPUTE_BOUND,
        "shadowspill_retention_test.unpriced_square.default": OperatorClass.UNKNOWN,
    }
    generous = RetentionPolicy(memory_bound_flops_per_byte=1e6)
    assert (
        generous.operator_classes(graph)["shadowspill_retention_test.product.default"]
        == OperatorClass.MEMORY_BOUND
    )
    assert policy.digest(graph) != generous.digest(graph)
    assert policy.digest(graph) == RetentionPolicy().digest(graph)


def test_a_formula_that_cannot_price_its_call_is_an_error() -> None:
    def forward(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return _pass_over(_mispriced(value), scale)

    inputs = _inputs((ROWS, WIDTH), (WIDTH,))
    with pytest.raises(CaptureError, match="mispriced.*could not price"):
        _pair(forward, inputs)


def test_the_policy_and_the_budget_are_range_checked() -> None:
    with pytest.raises(ValueError):
        RetentionPolicy(memory_bound_flops_per_byte=-1.0)
    with pytest.raises(TypeError):
        RetentionPolicy(memory_bound_flops_per_byte="16")  # type: ignore[arg-type]
    inputs = _inputs((ROWS, WIDTH), (WIDTH,), (WIDTH, WIDTH))
    with pytest.raises(ValueError):
        _pair(_normalized_product, inputs, memory_budget=1.5)
    pair = _pair(_normalized_product, inputs)
    with pytest.raises(ValueError):
        GraphPairVariant("save", 2.0, pair)
    assert GraphPairVariant("save", 1, pair).memory_budget == 1.0
