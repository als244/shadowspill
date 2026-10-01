"""The two gate arms use the same independent precision settings."""

from dataclasses import replace
from functools import partial

import pytest
import torch
from torch import nn

from qualification.numerical.reference_state import ReferenceState
from qualification.numerical.run import _decode_case, _parser
from qualification.precision import dtype_arguments, dtype_description


def test_dtype_flags_round_trip_and_all_affect_reference_identity() -> None:
    parser = _parser()
    options = parser.parse_args(
        [
            "run",
            "llama3",
            "out",
            "--model-dtype",
            "float16",
            "--master-dtype",
            "float32",
            "--grad-dtype",
            "float32",
            "--opt-state-dtype",
            "bfloat16",
        ]
    )
    request, _, _ = _decode_case(parser, options)
    assert request.dtypes.as_dict() == {
        "model_dtype": "float16",
        "master_dtype": "float32",
        "grad_dtype": "float32",
        "opt_state_dtype": "bfloat16",
    }
    replay = parser.parse_args(["run", "llama3", "out", *dtype_arguments(request)])
    assert _decode_case(parser, replay)[0].identity() == request.identity()
    for field, other in (
        ("model_dtype", "bfloat16"),
        ("master_dtype", "none"),
        ("grad_dtype", "float16"),
        ("opt_state_dtype", "float32"),
    ):
        assert replace(request, **{field: other}).identity() != request.identity()
    description = dtype_description(request)
    for text in (
        "model=float16",
        "masters=float32",
        "gradients=float32",
        "optimizer state=bfloat16",
    ):
        assert text in description


@pytest.mark.parametrize("grad_dtype", [torch.float16, torch.float32])
def test_reference_accumulates_at_the_selected_dtype(grad_dtype) -> None:
    model = nn.Linear(1, 1, bias=False, dtype=torch.float16)
    state = ReferenceState(
        model,
        partial(torch.optim.SGD, lr=1e-3),
        master_dtype=torch.float32,
        grad_dtype=grad_dtype,
    )
    state.begin_step()
    for value in (40_000, 40_000, -40_000):
        model.weight.grad = torch.full_like(model.weight, value)
        state.accumulate()
        assert model.weight.grad is None
    assert state.sums[0].dtype == grad_dtype
    if grad_dtype == torch.float16:
        assert torch.isinf(state.sums[0]).all()
    else:
        assert state.sums[0].item() == 40_000


def test_reference_updates_and_checkpoints_masters_before_casting_weights() -> None:
    model = nn.Linear(1, 1, bias=False, dtype=torch.float16)
    with torch.no_grad():
        model.weight.fill_(1)
    state = ReferenceState(
        model,
        partial(torch.optim.SGD, lr=1e-5),
        master_dtype=torch.float32,
        grad_dtype=torch.float32,
    )
    state.begin_step()
    model.weight.grad = torch.ones_like(model.weight)
    state.accumulate()
    state.step()
    saved = state.model_state()["weight"]
    assert saved.dtype == torch.float32
    assert saved.item() == pytest.approx(1 - 1e-5, abs=1e-7)
    assert model.weight.dtype == torch.float16
    assert model.weight.item() == 1
    assert torch.equal(model.weight, saved.to(torch.float16))


@pytest.mark.parametrize(
    "bf16,model,state",
    [
        (True, "bfloat16", "bfloat16"),
        (False, "float16", "float32"),
    ],
)
def test_device_defaults_resolve_before_constructing_a_case(bf16, model, state) -> None:
    from qualification.precision import resolve_dtype_defaults

    options = _parser().parse_args(["run", "llama3", "out"])
    resolve_dtype_defaults(options, hardware={"bf16": bf16})
    assert options.model_dtype == model
    assert options.grad_dtype == model
    assert options.master_dtype == "none"
    assert options.opt_state_dtype == state


def test_explicit_dtype_flags_override_device_defaults() -> None:
    from qualification.precision import resolve_dtype_defaults

    options = _parser().parse_args(
        [
            "run",
            "llama3",
            "out",
            "--model-dtype",
            "float32",
            "--master-dtype",
            "float32",
            "--grad-dtype",
            "float16",
            "--opt-state-dtype",
            "bfloat16",
        ]
    )
    resolve_dtype_defaults(options, hardware={"bf16": False})
    assert (
        options.model_dtype,
        options.master_dtype,
        options.grad_dtype,
        options.opt_state_dtype,
    ) == ("float32", "float32", "float16", "bfloat16")


@pytest.mark.parametrize("bf16", [True, False])
def test_hardware_defaults_preserve_or_reduce_model_and_budget(bf16) -> None:
    import json

    from qualification.device_defaults import (
        numerical_defaults,
        performance_defaults,
    )
    from qualification.performance.matrix import _parser as performance_parser

    options = _parser().parse_args(["run", "llama3", "out"])
    budgets = numerical_defaults(options, hardware={"bf16": bf16})
    assert budgets["llama3"] == (10 if bf16 else 3) << 30
    assert json.loads(options.model_config) == (
        {} if bf16 else {"n_layers": 4, "vocab_size": 8192}
    )
    perf = performance_parser().parse_args([])
    performance_defaults(perf, hardware={"bf16": bf16})
    assert perf.execution_budget_gib == (16 if bf16 else 10)
    assert options.external_headroom_mib == 512
    assert perf.external_headroom_mib == options.external_headroom_mib


def test_explicit_model_and_budget_overrides_win_on_a_smaller_gpu() -> None:
    import json

    from qualification.device_defaults import (
        numerical_defaults,
        performance_defaults,
    )
    from qualification.performance.matrix import _parser as performance_parser

    options = _parser().parse_args(
        [
            "run",
            "llama3",
            "out",
            "--model-config",
            '{"n_layers": 2, "vocab_size": 512}',
        ]
    )
    numerical_defaults(options, hardware={"bf16": False})
    assert json.loads(options.model_config) == {"n_layers": 2, "vocab_size": 512}
    perf = performance_parser().parse_args(["--execution-budget-gib", "7"])
    performance_defaults(perf, hardware={"bf16": False})
    assert perf.execution_budget_gib == 7


def test_original_bf16_reference_identity_is_preserved() -> None:
    from qualification.precision import resolve_dtype_defaults

    parser = _parser()
    options = parser.parse_args(["run", "llama3", "out"])
    resolve_dtype_defaults(options, hardware={"bf16": True})
    request, _, _ = _decode_case(parser, options)
    assert (
        request.identity()
        == "7b27b35072297e0b6eec7658192e79d1049735b23f4cb2927966f22144f8e473"
    )
