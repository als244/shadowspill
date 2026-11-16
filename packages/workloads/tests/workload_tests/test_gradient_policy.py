"""One optimizer setting reaches ordinary and provider gradient storage."""

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import BatchConfig, Memory, Model, Objective, Optimizer, Weights


@pytest.mark.parametrize(
    "family", ["llama3", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "olmoe"]
)
@pytest.mark.parametrize("dtype", [None, "float32", "parameter"])
def test_optimizer_gradient_precision_reaches_model_build_and_provider(family, dtype):
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=Model(family, initialize=Weights.random(seed=0)),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer(
            "adamw", precision={} if dtype is None else {"gradient_dtype": dtype}
        ),
        batch=BatchConfig(
            limits={"token_capacity": 16, "max_sequence_length": 8, "max_sequences": 8}
        ),
        memory=Memory(1, 1),
    )
    selected = "bfloat16" if dtype is None else dtype
    assert setup.optimizer.gradient_dtype == selected
    assert not setup.config.model.precision  # No second caller setting.
    for template in capture_inventory(prepare_layouts(setup)).templates.values():
        assert template.model_build.grad_dtype == (
            None if selected == "parameter" else selected
        )
        provider_dtype = "bfloat16" if selected == "parameter" else selected
        assert template.model_build.config["grad_dtype"] == provider_dtype
        import torch
        from mlops.dispatch import weight_gradient_dtype

        before = weight_gradient_dtype()
        with setup.model.capture_context(template.model_build):
            assert weight_gradient_dtype() == getattr(torch, provider_dtype)
        assert weight_gradient_dtype() == before
