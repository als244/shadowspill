"""The precision contract every harness exposes, and the refusal of what a
harness cannot honor."""

import argparse

import pytest

from baselines.common import precision
from baselines.common.precision import (
    Capabilities,
    Precision,
    Unsupported,
    with_changes,
)


def parse(*argv, defaults=None):
    parser = argparse.ArgumentParser()
    precision.add_arguments(parser, defaults)
    return parser.parse_args(argv)


def test_the_default_is_bf16_compute_with_fp32_masters_gradients_and_state():
    default = Precision()
    assert default.resolved() == {
        "model_dtype": "bfloat16",
        "master_dtype": "float32",
        "grad_dtype": "float32",
        "opt_state_dtype": "float32",
        "parameter_rounding": "default",
        "opt_state_rounding": "default",
        "round_accumulation_once": False,
    }
    assert default.stepped_dtype == "float32"
    assert "model_dtype=bfloat16" in default.describe()


def test_defaults_resolve_against_what_the_step_computes():
    weights_only = Precision(
        master_dtype=None, grad_dtype=None, opt_state_dtype="parameter"
    )
    assert weights_only.stepped_dtype == "bfloat16"
    resolved = weights_only.resolved()
    assert resolved["master_dtype"] == "none"
    assert resolved["grad_dtype"] == "bfloat16"
    assert resolved["opt_state_dtype"] == "bfloat16"
    assert weights_only.as_dict()["master_dtype"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"model_dtype": "float8"},
        {"master_dtype": "parameter"},
        {"grad_dtype": "int8"},
        {"opt_state_dtype": "weights"},
        {"parameter_rounding": "truncate"},
        {"opt_state_rounding": "random"},
        {"round_accumulation_once": "yes"},
    ],
)
def test_values_outside_the_contract_are_refused(changes):
    with pytest.raises(ValueError):
        Precision(**changes)


def test_dict_and_flag_round_trips_reproduce_a_request():
    request = Precision(
        master_dtype=None,
        grad_dtype="bfloat16",
        opt_state_dtype="parameter",
        parameter_rounding="stochastic",
        round_accumulation_once=True,
    )
    assert Precision.from_dict(request.as_dict()) == request
    assert Precision.from_arguments(parse(*request.arguments())) == request
    assert Precision.from_arguments(parse()) == Precision()
    assert Precision.from_arguments(parse(*Precision().arguments())) == Precision()
    with pytest.raises(ValueError):
        Precision.from_dict({"model_dtype": "bfloat16", "weights": "float32"})
    with pytest.raises(ValueError):
        with_changes(request, weights="float32")
    assert with_changes(request, grad_dtype="float32").grad_dtype == "float32"


def test_flags_accept_torch_names_and_none():
    parsed = parse(
        "--master-dtype",
        "torch.float32",
        "--grad-dtype",
        "none",
        "--opt-state-dtype",
        "none",
    )
    assert parsed.master_dtype == "float32"
    assert parsed.grad_dtype is None
    assert parsed.opt_state_dtype is None
    with pytest.raises(SystemExit):
        parse("--master-dtype", "float8")
    assert parse(defaults=Precision(model_dtype="float16")).model_dtype == "float16"


def test_capabilities_refuse_what_a_harness_cannot_honor():
    Capabilities("any").check(
        Precision(parameter_rounding="stochastic", round_accumulation_once=True)
    )
    fsdp = Capabilities(
        "fsdp2",
        master_dtypes=("float32",),
        grad_dtypes=("float32",),
        opt_state_dtypes=("float32",),
        parameter_roundings=(None,),
        opt_state_roundings=(None,),
        round_accumulation_once=False,
    )
    fsdp.check(Precision())
    with pytest.raises(Unsupported, match="master_dtype=none"):
        fsdp.check(Precision(master_dtype=None))
    with pytest.raises(Unsupported, match="parameter_rounding=stochastic"):
        fsdp.check(Precision(parameter_rounding="stochastic"))
    with pytest.raises(Unsupported, match="accumulation"):
        fsdp.check(Precision(round_accumulation_once=True))
