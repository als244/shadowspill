"""Installed token objectives use the generic packing and CPU capture path."""

import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import numpy as np
import pytest
import torch

from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.definitions import (
    InputAdapter,
    InputLayout,
    ModelDefinition,
    load_registry,
)
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.outputs import OutputLayout
from shadowspill.training.preparation.capture.binding import OperationBinding
from shadowspill.training.preparation.configuration.operations import OperationContract
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.packing import pack_batch
from shadowspill_client.inputs.shapes import BufferShape, PackingSchedule
from shadowspill_client.inputs.validation import combined_schema, validate_batch
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.payload import snapshot_batch
from workload_tests.objectives.test_token_objectives import oracle


def create(config):
    class TokenLookup(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.arange(49).reshape(7, 7).float()
            self.table = torch.nn.Parameter(torch.sin(values) / 4)

        def target_logprobs(self, tokens, targets):
            logits = torch.nn.functional.embedding(tokens, self.table)
            return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return TokenLookup()


def installed():
    registry = load_registry(["shadowspill-workloads"])
    registry.add_model(
        ModelDefinition(
            "external/token_lookup@1",
            dict,
            create,
            InputSchema(
                [
                    TensorField(
                        ("tokens",),
                        "int64",
                        ("position",),
                        "pack",
                        0,
                        minimum=0,
                        maximum=6,
                    )
                ],
                [
                    Capacity(
                        "token_capacity",
                        "tokens",
                        "sum",
                        ("tokens",),
                        0,
                        scheduled=True,
                    )
                ],
            ),
            capabilities=("target_logprobs",),
            input_adapter=lambda config, layout: InputAdapter(padding_safe=True),
        ),
        source=DefinitionSource("token-fixture", "1", "b" * 64),
    )
    return registry, registry.resolve_model("external/token_lookup@1")


def records(*, policy=False, reference=False):
    result = []
    for tokens, targets, valid, weights, old, advantages in (
        (
            [1, 2, 3],
            [2, np.nan, 4],
            [True, False, True],
            [1, np.nan, -0.25],
            [-2.0, np.nan, -2.2],
            [1, np.nan, -0.5],
        ),
        ([2, 5], [5, 6], [True, True], [0.5, 0], [-1.3, -2.1], [-0.7, 0]),
    ):
        loss = dict(target_tokens=targets, weights=weights)
        loss["response_mask" if policy else "loss_mask"] = valid
        if policy:
            loss.update(old_logprobs=old, advantages=advantages)
            if reference:
                loss["reference_logprobs"] = [-2.3 if v else np.nan for v in valid]
        result.append(Datum({"tokens": tokens}, loss))
    return result


@contextmanager
def bound(model, objective, key, datums):
    schema = combined_schema(model.inputs, objective.inputs)
    contract = OperationContract(key, objective)
    shape = BufferShape(4, {path: (8,) for path in contract.required_paths})
    schedule = PackingSchedule("tokens", [[shape]])
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields]
    )
    budget = BufferBudget(4 << 20, max_requests=8)
    with ExitStack() as stack:
        snapshot = stack.enter_context(snapshot_batch(datums, budget))
        batch = validate_batch(
            snapshot,
            schema,
            BatchConfig(limits={"token_capacity": 8}),
            required_paths=contract.required_paths,
        )
        assignment = pack_batch(batch, schema, batch.config, placement, schedule)
        physical = stack.enter_context(
            materialize_buffer(assignment, schema, 0, budget)
        )
        inputs = model.bind_inputs(InputLayout(schema, shape, placement, 0))
        output = OutputLayout(schema, contract.outputs, schedule, placement)
        binding = OperationBinding(model, contract, inputs, output, 0)
        adapted = stack.enter_context(
            inputs.adapt(route_buffer(physical, placement, 0), budget)
        )
        yield binding, adapted
    assert budget.used == (0, 0)


def independent(model, datums, *, policy, reference, clip, beta):
    table = model.table.detach().numpy().astype(np.float64)
    gradient = np.zeros_like(table)
    loss, count = 0.0, 0
    for datum in datums:
        fields = datum.loss_inputs
        for i, active in enumerate(fields["response_mask" if policy else "loss_mask"]):
            if not active:
                continue
            row, target = datum.inputs["tokens"][i], int(fields["target_tokens"][i])
            probabilities = np.exp(table[row] - np.max(table[row]))
            probabilities /= probabilities.sum()
            current = np.log(probabilities[target])
            weight = fields["weights"][i]
            if policy:
                value, derivative = oracle(
                    [current],
                    [True],
                    [weight],
                    [fields["old_logprobs"][i]],
                    [fields["advantages"][i]],
                    clip,
                    [fields["reference_logprobs"][i]] if reference else None,
                    beta,
                )
                derivative = derivative[0]
            else:
                value, derivative = -weight * current, -weight
            loss += value
            count += 1
            local = -probabilities
            local[target] += 1
            gradient[row] += derivative * local
    return loss, count, gradient


@pytest.mark.parametrize(
    "objective_name,reference",
    [
        ("cross_entropy", False),
        ("clipped_policy", False),
        ("clipped_policy", True),
    ],
)
def test_installed_objectives_pack_mask_capture_and_match_independent_gradients(
    objective_name, reference
):
    registry, description = installed()
    policy = objective_name == "clipped_policy"
    objective = registry.resolve_objective(
        f"shadowspill/{objective_name}@1",
        model=description,
        config={"reference_penalty": "k3"} if reference else {},
    )
    datums = records(policy=policy, reference=reference)
    key = OperationKey("forward_backward", "logprobs")
    model = description.create()
    with bound(description, objective, key, datums) as (binding, adapted):
        controls = {"clip_ratio": 0.17} if policy else {}
        if reference:
            controls["reference_beta"] = 0.13
        arguments = binding.arguments(adapted, controls)
        capture = capture_training_objective(model, binding.objective, list(arguments))
        loss, logprobs, loss_sum, count = capture.exported.exported_program.module()(
            *arguments
        )
        expected, expected_count, expected_gradient = independent(
            model,
            datums,
            policy=policy,
            reference=reference,
            clip=0.17,
            beta=0.13,
        )
        assert loss.item() == pytest.approx(expected, abs=2e-6)
        assert loss_sum.item() == loss.item() and count.item() == expected_count == 4
        assert logprobs.shape == (8,) and logprobs.dtype == torch.float32
        mask = arguments[1]["response_mask" if policy else "loss_mask"]
        assert torch.count_nonzero(logprobs[~mask]) == 0
        assert torch.isfinite(logprobs).all()
        np.testing.assert_allclose(
            torch.autograd.grad(loss, model.table)[0].numpy(),
            expected_gradient,
            atol=2e-6,
            rtol=2e-6,
        )


def test_policy_logprob_forward_needs_only_targets_and_does_not_use_unused_masks():
    registry, model = installed()
    objective = registry.resolve_objective(
        "shadowspill/clipped_policy@1", model=model, config={"reference_penalty": "k3"}
    )
    key = OperationKey("forward", "logprobs", "evaluation")
    datum = Datum({"tokens": [1, 2, 3]}, {"target_tokens": [3, 4, 5]})
    with bound(model, objective, key, [datum]) as (binding, adapted):
        created = model.create()
        args = binding.arguments(adapted)
        loss, values = binding.forward(created, *args)
        assert loss is None and not args[2]
        assert args[1].keys() == {"target_tokens"}
        torch.testing.assert_close(
            values["logprobs"][:3],
            created.target_logprobs(torch.tensor([1, 2, 3]), torch.tensor([3, 4, 5])),
        )
    for operation in ("step", "forward_backward"):
        with (
            pytest.raises(ValueError, match="missing required"),
            bound(model, objective, OperationKey(operation, "logprobs"), [datum]),
        ):
            pytest.fail("policy training accepted missing rollout inputs")
    with (
        pytest.raises(ValueError, match="above maximum"),
        bound(model, objective, key, [Datum({"tokens": [1]}, {"target_tokens": [7]})]),
    ):
        pytest.fail("out-of-vocabulary target reached execution")


def test_configuration_is_model_derived_and_reference_controls_are_structural():
    registry, model = installed()
    ordinary = registry.resolve_objective("shadowspill/clipped_policy@1", model=model)
    assert ordinary.config["vocab_size"] == 7
    assert ordinary.defaults == {"clip_ratio": 0.2}
    assert ("reference_logprobs",) not in {f.path for f in ordinary.inputs.fields}
    reference = registry.resolve_objective(
        ordinary.definition.id,
        model=model,
        config={"reference_penalty": "k3"},
        defaults={"reference_beta": 0.03},
    )
    contract = OperationContract(OperationKey("step", "metrics"), reference)
    assert contract.loss_params({"clip_ratio": 0.1}) == {
        "clip_ratio": 0.1,
        "reference_beta": 0.03,
    }
    assert contract.loss_params() == {"clip_ratio": 0.2, "reference_beta": 0.03}
    for config, defaults in (
        ({"reference_penalty": "typo"}, {}),
        ({"vocab_size": 8}, {}),
        ({}, {"reference_beta": 0.1}),
        ({}, {"clip_ratio": 1.0}),
        ({"reference_penalty": "k3"}, {"reference_beta": -1}),
    ):
        with pytest.raises(ValueError):
            registry.resolve_objective(
                ordinary.definition.id, model=model, config=config, defaults=defaults
            )
    with pytest.raises(ValueError, match="lacks required"):
        registry.resolve_objective(
            ordinary.definition.id,
            model=registry.resolve_model("shadowspill/regression@1"),
        )
    invalid = replace(
        model,
        inputs=replace(
            model.inputs, fields=[replace(model.inputs.fields[0], maximum=None)]
        ),
    )
    with pytest.raises(ValueError, match="bounded, aligned"):
        registry.resolve_objective(ordinary.definition.id, model=invalid)


def test_installed_registration_and_resolution_do_not_import_frameworks():
    script = """
import sys
for name in ('torch', 'triton', 'mlops', 'transformers'):
    sys.modules[name] = None
from shadowspill.training.definitions import load_registry, ModelDefinition
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.inputs import InputSchema, TensorField
registry = load_registry(['shadowspill-workloads'])
registry.add_model(ModelDefinition('external/token@1', dict, lambda _: None,
    InputSchema([TensorField(('tokens',), 'int64', ('position',), 'pack', 0,
                            minimum=0, maximum=6)]), capabilities=('target_logprobs',)),
    source=DefinitionSource('fixture', '1', 'a'*64))
model = registry.resolve_model('external/token@1')
for name in ('cross_entropy', 'clipped_policy'):
    objective = registry.resolve_objective('shadowspill/'+name+'@1', model=model)
    assert objective.config['vocab_size'] == 7
assert 'mlops.objectives' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
