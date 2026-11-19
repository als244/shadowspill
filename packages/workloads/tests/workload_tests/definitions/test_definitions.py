"""A supplied model uses the same registry and CPU capture as external models."""

import subprocess
import sys
from hashlib import sha256

import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.parallelism import Parallelism
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.definitions import (
    InitializationContext,
    InitializationRuntime,
    Registry,
)
from shadowspill_client import Model, Weights
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_workloads.definitions import register


def test_supplied_regression_definition_uses_existing_model_and_capture():
    registry = Registry()
    register(registry)
    description = registry.resolve_model("shadowspill/regression@1")
    objective = registry.resolve_objective(
        "shadowspill/weighted_squared_error@1", model=description
    )
    model = description.create()
    inputs = {"value": torch.randn(3, 16)}
    loss_inputs = {
        "targets": torch.randn(3, 8),
        "valid": torch.tensor([True, True, False]),
        "weights": torch.tensor([1.0, 0.0, 0.0]),
    }
    captured = capture_training_objective(
        model, objective.computation(), [inputs, loss_inputs, {}]
    )
    actual, loss_sum, count = captured.exported.exported_program.module()(
        inputs, loss_inputs, {}
    )
    expected = (model(**inputs)[0] - loss_inputs["targets"][0]).square().sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(loss_sum, actual.detach())
    assert count.item() == 2
    assert description.definition.mutable_buffers == ("updates",)
    assert all(
        torch.isfinite(grad).all()
        for grad in torch.autograd.grad(actual, tuple(model.parameters()))
    )


def test_regression_description_and_initializer_preflight_are_framework_free():
    script = """
import sys
# MLOps is located for the AdamW definition, never imported.
for name in ('torch', 'triton'):
    sys.modules[name] = None
from shadowspill.training.definitions import load_registry, InitializationContext
from shadowspill.parallelism import Parallelism
from shadowspill_client import Model, Weights
registry = load_registry(['shadowspill-workloads'])
model = registry.resolve_model('shadowspill/regression@1', settings=Model(
    'shadowspill/regression@1', initialize=Weights.random(seed=0)))
initialization = model.describe_initialization(InitializationContext(
    model.settings, model.settings.initialize, Parallelism(), 0))
assert initialization.peak_bytes == 64 << 10
assert initialization.chunk_bytes == 4096
assert not any(name.split('.')[0] == 'mlops' for name in sys.modules)
assert 'shadowspill_workloads.models.regression.model' not in sys.modules
assert 'shadowspill_workloads.models.regression.initialization' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_regression_initialization_preserves_original_values_and_seeded_chunks(
    tmp_path,
):
    registry = Registry()
    register(registry)
    values = []
    for seed in (0, 17):
        model = registry.resolve_model(
            "shadowspill/regression@1",
            settings=Model(
                "shadowspill/regression@1", initialize=Weights.random(seed=seed)
            ),
        )
        description = model.describe_initialization(
            InitializationContext(
                model.settings, model.settings.initialize, Parallelism(), 0
            )
        )
        actual = {}
        with description.open(
            InitializationRuntime(
                tmp_path, lambda _: pytest.fail("unexpected artifact")
            )
        ) as fill:
            for name, tensor in model.create().state_dict().items():
                size = tensor.numel()
                full = torch.empty(size, dtype=tensor.dtype)
                fill(
                    TensorChunk(name, tuple(tensor.shape), tensor.dtype, 0, size), full
                )
                chunks = torch.empty_like(full)
                for start in range(0, size, 7):
                    fill(
                        TensorChunk(
                            name, tuple(tensor.shape), tensor.dtype, start, size
                        ),
                        chunks[start : start + 7],
                    )
                torch.testing.assert_close(chunks, full, rtol=0, atol=0)
                actual[name] = full
                if seed == 0 and name != "updates":
                    generator = torch.Generator().manual_seed(
                        int.from_bytes(sha256(name.encode()).digest()[:8], "little")
                    )
                    width = 16 if name.startswith("network.0.") else 32
                    original = torch.empty(size).uniform_(
                        -(width**-0.5), width**-0.5, generator=generator
                    )
                    torch.testing.assert_close(full, original, rtol=0, atol=0)
            with pytest.raises(ValueError, match="geometry"):
                fill(
                    TensorChunk(
                        "network.0.weight", (1 << 28,), torch.float32, 0, 1 << 28
                    ),
                    torch.empty(1),
                )
        values.append(actual)
    assert all(
        not torch.equal(values[0][name], values[1][name])
        for name in values[0]
        if name != "updates"
    )
    assert values[0]["updates"].item() == values[1]["updates"].item() == 0


def test_regression_prediction_requires_no_loss_fields_and_evaluation_is_read_only():
    registry = Registry()
    register(registry)
    description = registry.resolve_model("shadowspill/regression@1")
    objective = registry.resolve_objective(
        "shadowspill/weighted_squared_error@1", model=description
    )
    model = description.create()
    inputs = {"value": torch.randn(3, 16)}
    predictions = objective.computation(
        OperationKey("forward", "predictions", "evaluation")
    )
    model.train()
    first = predictions(model, inputs, {}, {})["prediction"]
    assert model.updates.item() == 1
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.eval()
    second = predictions(model, inputs, {}, {})["prediction"]
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert not objective.definition.outputs["predictions"].loss_inputs
