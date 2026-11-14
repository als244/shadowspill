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
    RetentionRule,
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


@torch.library.custom_op("shadowspill_retention_test::sincos", mutates_args=())
def _sincos(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value.sin(), value.cos()


@_sincos.register_fake
def _(value):
    return torch.empty_like(value), torch.empty_like(value)


def _sincos_setup(ctx, inputs, output):
    ctx.save_for_backward(*output)


def _sincos_backward(ctx, dsin, dcos):
    sine, cosine = ctx.saved_tensors
    return dsin * cosine - dcos * sine


_sincos.register_autograd(_sincos_backward, setup_context=_sincos_setup)


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


@pytest.mark.parametrize("memory_budget", [0.0, 1.0])
@pytest.mark.parametrize("default", ["cost", "retain", "recompute"])
def test_explicit_retention_preserves_a_selected_result_and_correct_gradients(
    default, memory_budget
):
    def forward(value, weight):
        return torch.sin(value @ weight).square()

    inputs = _inputs((17, 8), (8, 8))
    policy = RetentionPolicy(
        default=default, rules=(RetentionRule("aten.sin.default", "retain"),)
    )
    pair = _pair(forward, inputs, retention=policy, memory_budget=memory_budget)
    assert "aten.sin.default" not in pair.retention.regenerated_operators
    if default == "recompute":
        assert "aten.mm.default" in pair.retention.regenerated_operators
    values = pair.forward.graph_module(*inputs)
    tangent = torch.randn_like(values[0])
    actual = pair.backward.graph_module(*values[-pair.saved_value_count :], tangent)
    expected = torch.autograd.grad(forward(*inputs), inputs, tangent)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)


def test_save_removals_and_recompute_additions_are_independent():
    from shadowspill.pytorch import GraphPairPolicy

    inputs = _inputs((ROWS, WIDTH), (WIDTH, WIDTH), (WIDTH,))
    target = "shadowspill_retention_test.product.default"
    policy = GraphPairPolicy(
        save=RetentionPolicy(rules=(RetentionRule(target, "recompute"),)),
        recompute=RetentionPolicy(
            default="recompute", rules=(RetentionRule(target, "retain"),)
        ),
    )
    replay = _pair(
        _product_then_pass,
        inputs,
        retention=policy.save,
    )
    assert target in replay.retention.regenerated_operators
    assert _internal_bytes(replay) == 0
    retained = _pair(
        _product_then_pass,
        inputs,
        memory_budget=0.0,
        retention=policy.recompute,
    )
    assert target not in retained.retention.regenerated_operators
    assert _internal_bytes(retained) == BYTES


def test_selected_only_save_and_inputs_only_recompute_have_independent_vjps():
    from shadowspill.pytorch import GraphPairPolicy

    def forward(value, first, second, scale):
        selected = _sincos(_product(value, first))[0]
        return _pass_over(_product(selected, second), scale)

    torch.manual_seed(15)
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in ((17, 8), (8, 8), (8, 8), (8,))
    )
    target = "shadowspill_retention_test.sincos.default"
    policy = GraphPairPolicy(
        save=RetentionPolicy(
            default="recompute", rules=(RetentionRule(target, "retain"),)
        ),
        recompute=RetentionPolicy(default="recompute"),
    )
    expected = forward(*inputs)
    tangent = torch.randn_like(expected)
    gradients = torch.autograd.grad(expected, inputs, tangent)
    saved_bytes = {}
    for variant, budget in (("save", 1.0), ("recompute", 0.0)):
        pair = _pair(
            forward, inputs, memory_budget=budget, retention=policy.variant(variant)
        )
        with torch.no_grad():
            values = pair.forward.graph_module(*inputs)
            actual = pair.backward.graph_module(
                *values[-pair.saved_value_count :], tangent
            )
        torch.testing.assert_close(values[0], expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual, gradients, rtol=1e-11, atol=1e-11)
        saved_bytes[variant] = _internal_bytes(pair)
        replayed = pair.retention.regenerated_operators
        assert (target in replayed) == (variant == "recompute")
        # The second product is expensive and needed by backward, but neither
        # policy retains it. SAVE keeps both required results of sincos alone.
        assert "shadowspill_retention_test.product.default" in replayed
    assert saved_bytes == {"save": 2 * expected.numel() * 8, "recompute": 0}


@pytest.mark.parametrize("memory_budget", [0.0, 1.0])
def test_selective_cut_keeps_auxiliary_results_without_unnecessary_replay(
    memory_budget,
):
    def forward(value, weight):
        return _sincos(value @ weight)[0]

    inputs = _inputs((17, 8), (8, 8))
    policy = RetentionPolicy(
        default="recompute",
        rules=(RetentionRule("shadowspill_retention_test.sincos.default", "retain"),),
    )
    pair = _pair(forward, inputs, retention=policy, memory_budget=memory_budget)
    replayed = pair.retention.regenerated_operators
    assert "shadowspill_retention_test.sincos.default" not in replayed
    # The selected operation's derivative uses only its two results.
    assert "aten.mm.default" not in replayed
    values = pair.forward.graph_module(*inputs)
    tangent = torch.randn_like(values[0])
    actual = pair.backward.graph_module(*values[-pair.saved_value_count :], tangent)
    expected = torch.autograd.grad(forward(*inputs), inputs, tangent)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
    assert len(values) == 5  # Public result, two inputs, and both sin/cos results.


@pytest.mark.cuda
@pytest.mark.parametrize("memory_budget", [0.0, 1.0])
@pytest.mark.parametrize(
    "batch, length, dtype", [(1, 17, torch.bfloat16), (2, 31, torch.float16)]
)
def test_selective_flash_attention_keeps_backward_state_and_replays_projection(
    batch, length, dtype, memory_budget
):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def forward(value, weight):
        q, k, v = (value @ weight).reshape(batch, length, 3, 4, 32).unbind(2)
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )

    torch.manual_seed(12)
    inputs = tuple(
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for shape in ((batch, length, 128), (128, 384))
    )
    target = "aten._scaled_dot_product_flash_attention.default"
    policy = RetentionPolicy(
        default="recompute", rules=(RetentionRule(target, "retain"),)
    )
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        pair = _pair(forward, inputs, retention=policy, memory_budget=memory_budget)
        values = pair.forward.graph_module(*inputs)
        tangent = torch.randn_like(values[0])
        actual = pair.backward.graph_module(*values[-pair.saved_value_count :], tangent)
        expected = torch.autograd.grad(forward(*inputs), inputs, tangent)
    assert target not in pair.retention.regenerated_operators
    assert "aten.mm.default" in pair.retention.regenerated_operators
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)


def test_rule_identity_covers_builtin_overrides_and_is_order_independent():
    graph = make_fx(lambda x: x.sin().cos())(*_inputs((8, 8)))
    rules = (
        RetentionRule("aten.sin.default", "retain"),
        RetentionRule("aten.cos.default", "recompute"),
    )
    policy = RetentionPolicy(rules=rules)
    assert policy.digest(graph) != RetentionPolicy().digest(graph)
    assert policy.digest(graph) == RetentionPolicy(rules=rules[::-1]).digest(graph)
    assert policy.digest(graph) != RetentionPolicy(
        rules=rules, default="recompute"
    ).digest(graph)


def test_conflicting_rules_and_model_annotations_are_refused():
    from torch.utils.checkpoint import CheckpointPolicy

    from shadowspill.pytorch.capture.retention import _mark_retained

    target = "aten.sin.default"
    with pytest.raises(ValueError, match="at most once"):
        RetentionPolicy(
            rules=(RetentionRule(target, "retain"), RetentionRule(target, "recompute"))
        )
    graph = make_fx(lambda x: x.sin())(*_inputs((8, 8)))
    node = next(n for n in graph.graph.nodes if str(n.target) == target)
    node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    with pytest.raises(CaptureError, match="model checkpoint annotation"):
        _mark_retained(graph, RetentionPolicy(rules=(RetentionRule(target, "retain"),)))


@pytest.mark.parametrize("memory_budget", [0.0, 1.0])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_selected_recompute_cut_preserves_random_values_and_rng_state(
    memory_budget, device
):
    def forward(value):
        return (value * torch.rand_like(value)).sin()

    inputs = (torch.randn(17, 8, device=device, requires_grad=True),)
    pair = _pair(
        forward,
        inputs,
        memory_budget=memory_budget,
        retention=RetentionPolicy(
            default="recompute", rules=(RetentionRule("aten.sin.default", "retain"),)
        ),
    )
    torch.manual_seed(21)
    values = pair.forward.graph_module(*inputs)
    rng_state = torch.get_rng_state if device == "cpu" else torch.cuda.get_rng_state
    state = rng_state().clone()
    tangent = torch.ones_like(values[0])
    actual = pair.backward.graph_module(*values[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(rng_state(), state)
    torch.manual_seed(21)
    expected = torch.autograd.grad(forward(*inputs), inputs, tangent)
    torch.testing.assert_close(actual, expected)
    assert "aten.rand_like.default" not in pair.retention.regenerated_operators


def test_explicit_replay_of_stateful_random_operations_is_refused():
    inputs = _inputs((17, 8))
    with pytest.raises(CaptureError, match="RNG/effect state"):
        _pair(
            lambda value: value * torch.rand_like(value),
            inputs,
            memory_budget=0.0,
            retention=RetentionPolicy(
                default="recompute",
                rules=(RetentionRule("aten.rand_like.default", "recompute"),),
            ),
        )


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
    with pytest.raises(CaptureError, match=r"mispriced.*could not price"):
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
