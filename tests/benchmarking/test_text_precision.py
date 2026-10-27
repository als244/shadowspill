"""Catalog precision reaches model construction, build and optimizer capture."""

import functools
from types import SimpleNamespace

import pytest
import torch

from workloads.recipes.text.precision import Precision


def _arguments(**overrides):
    return SimpleNamespace(
        **{
            "model_dtype": "bfloat16",
            "master_dtype": "float32",
            "grad_dtype": "float32",
            "opt_state_dtype": "parameter",
            "parameter_rounding": None,
            "opt_state_rounding": "stochastic",
            "fuse_narrow_gradient_accumulation": False,
            **overrides,
        }
    )


def _adamw(parameters: object, **arguments: object) -> dict[str, object]:
    return {"parameters": parameters, **arguments}


def test_precision_reaches_planning_and_the_optimizer_as_the_harness_names_it() -> None:
    """The flags carry the training config's names into the same two places:
    build_step's keywords, and the optimizer's own arguments."""

    nothing = Precision.from_arguments(
        _arguments(
            master_dtype=None,
            grad_dtype=None,
            opt_state_dtype=None,
            parameter_rounding=None,
            opt_state_rounding=None,
            fuse_narrow_gradient_accumulation=False,
        )
    )
    assert nothing.plan_arguments() == {
        "master_dtype": None,
        "grad_dtype": None,
        "fuse_narrow_gradient_accumulation": False,
    }
    # Naming nothing hands planning the optimizer as it was, so a request
    # without the flags plans exactly as before.
    assert nothing.optimizer(_adamw) is _adamw

    precise = Precision.from_arguments(
        _arguments(fuse_narrow_gradient_accumulation=True)
    )
    assert precise.plan_arguments() == {
        "master_dtype": torch.float32,
        "grad_dtype": torch.float32,
        "fuse_narrow_gradient_accumulation": True,
    }
    built = precise.optimizer(_adamw)
    assert isinstance(built, functools.partial)
    assert built(["p"]) == {
        "parameters": ["p"],
        # A gradient dtype is also what the optimizer reads gradients at,
        # else it would round them to its default on the way in.
        "gradient_dtype": torch.float32,
        "opt_state_dtype": "parameter",
        "opt_state_rounding": "stochastic",
    }
    assert (
        Precision.from_arguments(
            _arguments(opt_state_dtype="float32")
        ).optimizer_arguments()["opt_state_dtype"]
        is torch.float32
    )
    labels = [label for label, _value, _meaning in precise.lines()]
    assert labels == [
        "model dtype",
        "master dtype",
        "grad dtype",
        "opt state dtype",
        "parameter rounding",
        "opt state rounding",
    ]


def test_text_recipe_normalizes_by_the_requested_step_total():
    from workloads.recipes.text.benchmark.factory import experiment

    setup = experiment(
        device="cuda:0",
        model="olmoe-7b-a1b",
        sequence_length=8192,
        sequences_per_step=32,
    )
    assert setup["units_per_step"] == 8192 * 32
    assert setup["metadata"]["tokens_per_step"] == 8192 * 32


def test_text_recipe_reports_head_loss_and_keeps_other_objectives():
    from shadowspill.training.observations import StepObservations
    from workloads.recipes.text.benchmark.factory import metrics

    observed = StepObservations((2.0, 3.0), ({"head_loss": 1.5}, None))
    assert metrics(observed)["loss"] == 4.5


@pytest.mark.parametrize("master", ["none", "float32"])
@pytest.mark.parametrize("grad", [None, "float16", "float32"])
def test_fp16_quickstart_constructs_the_requested_model_and_precision(master, grad):
    from workloads.recipes.text.benchmark.factory import experiment

    arguments = _arguments(
        model_dtype="float16",
        master_dtype=None if master == "none" else master,
        grad_dtype=grad,
        opt_state_dtype="float32",
    )
    setup = experiment(
        device="cuda:0",
        model="llama3-8b",
        precision={
            "model_dtype": "float16",
            "master_dtype": None if master == "none" else master,
            "grad_dtype": grad,
            "opt_state_dtype": "float32",
        },
    )
    model = setup["model_factory"]()
    assert {p.dtype for p in model.parameters()} == {torch.float16}
    assert all(p.is_meta for p in model.parameters())
    assert setup["metadata"]["dtypes"] == {
        "model_dtype": "float16",
        "master_dtype": master,
        "grad_dtype": grad or "float16",
        "opt_state_dtype": "float32",
    }
    precision = Precision.from_arguments(arguments)
    assert precision.master is (None if master == "none" else torch.float32)
    assert precision.gradients is (None if grad is None else getattr(torch, grad))
    assert precision.optimizer_arguments()["opt_state_dtype"] == torch.float32
    assert (
        "model dtype",
        "float16",
        "the model weights and activations",
    ) in precision.lines()


def test_quickstart_keeps_bf16_defaults_without_probing_the_gpu():
    from workloads.recipes.text.benchmark.factory import experiment

    setup = experiment(device="cuda:0", model="llama3-8b")
    assert setup["metadata"]["dtypes"] == {
        "model_dtype": "bfloat16",
        "master_dtype": "none",
        "grad_dtype": "bfloat16",
        "opt_state_dtype": "bfloat16",
    }
