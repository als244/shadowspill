"""Prepared public two-H100 EP training with independent numerical references."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from shadowspill import Parallelism
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.artifacts.bundle import TrainingBundle
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.packing import pack_batch
from shadowspill_client.inputs.validation import validate_batch
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.payload import snapshot_batch
from workload_tests.service.helpers._text_service_client import training_settings
from workload_tests.service.oracles._expert_oracle import (
    SPLIT,
    WHOLE,
    WeightGradients,
    initial_model,
    loss,
    options,
    records,
)


@pytest.fixture(scope="module", params=["bfloat16", "float32"])
def reference(request):
    return options(grad_dtype=request.param)


def test_oracle_has_distinct_shards_fp32_masters_and_geometry_specific_updates(
    reference,
):
    assert len(json.dumps(reference)) < 64 << 20
    assert abs(reference["whole_loss"] - sum(reference["loss_parts"])) > 1e-3
    initial = reference["oracles"]["whole"][0]["weights"]
    for name in ("gate_up_weight", "down_weight"):
        weights = torch.tensor(initial[f"blocks.0.moe.experts.{name}"])
        assert not torch.equal(*weights.chunk(2))
    for trajectory in reference["oracles"].values():
        for name, before in initial.items():
            final = torch.tensor(trajectory[-1]["weights"][name])
            assert torch.isfinite(final).all()
            assert not torch.equal(final, torch.tensor(before)), name
            assert not torch.equal(final, final.bfloat16().float()), name
    assert reference["oracles"]["whole"][1] != reference["oracles"]["split"][1]


def test_reference_padding_has_no_loss_or_gradient_participation():
    _, model = initial_model()
    values, derivatives = [], []
    for size in (4, 8):
        value = loss(model, records(), (0,), 0.07, size=size)
        values.append(value)
        derivatives.append(torch.autograd.grad(value, tuple(model.parameters())))
    torch.testing.assert_close(*values, rtol=3e-5, atol=3e-6)
    # Changing the matrix shape changes CPU reduction order before BF16 casts.
    # Measured worst gradient-norm difference here is 0.131%; bound it below a
    # BF16 rounding unit. The public update bound remains the existing 3%.
    for first, second in zip(*derivatives, strict=True):
        difference = first.float() - second.float()
        assert difference.norm() <= 0.005 * second.float().norm() + 1e-7
    empty = loss(model, records(), (), 0.07)
    assert empty.item() == 0
    assert all(
        not gradient.any()
        for gradient in torch.autograd.grad(empty, tuple(model.parameters()))
    )


def test_fp32_weight_products_match_autograd_across_owner_contributions():
    _, model = initial_model()
    model = model.float()
    reference = deepcopy(model)
    observer = WeightGradients(model)
    for indices, coefficient in (((0, 3), 0.07), ((1, 2), 0.03), ((), 0.11)):
        loss(model, records(), indices, coefficient).backward()
        loss(reference, records(), indices, coefficient).backward()
    expected = dict(reference.named_parameters())
    assert observer.values.keys() == expected.keys()
    for name, parameter in expected.items():
        torch.testing.assert_close(
            observer.values[name], parameter.grad, rtol=3e-5, atol=3e-6, msg=name
        )
    observer.close()
    assert all(block.moe.gradient_observer is None for block in model.blocks)


def test_expert_weight_product_is_retained_before_bf16_leaf_rounding():
    _, model = initial_model()
    observer = WeightGradients(model)
    weight = model.blocks[0].moe.experts.down_weight
    inputs = torch.zeros(4, weight.shape[-1], dtype=torch.bfloat16)
    inputs[:, 0] = torch.tensor([1, 1 / 256, 1 / 256, 1 / 256])
    output = torch.nn.functional.linear(inputs.float(), weight[0].float())
    observer._observe("blocks.0.moe.experts.down_weight", 0, inputs, output)
    output.sum().backward()
    product = observer.values["blocks.0.moe.experts.down_weight"]
    assert product.dtype == torch.float32
    assert product[0, 0, 0].item() == 1 + 3 / 256
    assert weight.grad[0, 0, 0].item() != product[0, 0, 0].item()
    assert not product[1:].any()
    observer.close()


def test_explicit_oracle_groups_match_every_prepared_count_and_empty_owner(reference):
    requested = training_settings(reference)
    requested["parallelism"] = Parallelism.from_dict(requested["parallelism"])
    setup = resolve_training(
        load_registry(["shadowspill", "shadowspill-workloads"]), **requested
    )
    layouts = prepare_layouts(setup)
    key = OperationKey("forward_backward", "metrics")
    budget = BufferBudget(8 << 20, 8)
    for selected, groups in (
        (range(4), WHOLE),
        ((0,), SPLIT[0]),
        ((1, 2, 3), SPLIT[1]),
    ):
        selected = tuple(selected)
        expected = tuple(part for part in groups if part)
        for candidate in layouts.families[key]:
            binding = candidate.bindings[0, 0]
            with snapshot_batch(
                [Datum(**reference["records"][i]) for i in selected], budget
            ) as snapshot:
                batch = validate_batch(
                    snapshot,
                    setup.context.schema,
                    setup.context.batches[0],
                    required_paths=binding.contract.required_paths,
                )
                assignment = pack_batch(
                    batch,
                    setup.context.schema,
                    batch.config,
                    setup.context.parallelism.inputs,
                    candidate.candidate.schedule,
                )
                actual = tuple(
                    tuple(selected[index] for index in part)
                    for part in assignment.records
                    if part
                )
                assert actual == expected
                assert (
                    candidate.candidate.schedule.capacities(setup.context.schema)[
                        "token_capacity"
                    ]
                    <= 16
                )
                if len(selected) == 1:
                    assert (
                        len(assignment.records) >= 2
                        and sum(bool(part) for part in assignment.records) == 1
                    )
    assert budget.used == (0, 0)


@pytest.mark.cuda
@pytest.mark.slow
def test_public_two_h100_ep_step_split_empty_owner_and_checkpoint_resume(
    tmp_path, reference
):
    if torch.cuda.device_count() < 2 or any(
        torch.cuda.get_device_capability(rank)[0] != 9 for rank in range(2)
    ):
        pytest.skip("requires two SM90 GPUs and the installed MoonEP provider")
    (tmp_path / "setup.json").write_text(json.dumps(reference))
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "helpers/_catalog_service_server.py"),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=2100,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    for name in (
        "all_client_claims_released",
        "all_server_claims_released",
        "independent_geometry_oracles",
        "evaluation_isolation",
        "separate_fp32_masters",
        "exact_resume",
    ):
        assert receipt[name], name
    source = (
        tmp_path
        / "service/artifacts"
        / hashlib.sha256(b"owner").hexdigest()
        / "data"
        / hashlib.sha256(receipt["plans"]["id"].encode()).hexdigest()
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt["plans"]["sha256"]
    bundle = TrainingBundle.load(source)
    assert set(bundle.plans) == {0, 1}
    calls = set()
    grad_dtype = reference["gradient_dtype"]
    for plans in bundle.plans.values():
        for plan in plans.values():
            if grad_dtype == "float32":
                assert len(plan.program.bindings["component_layouts"]) == 3
            for task in plan.program.bindings["allocation_coverage"]["tasks"].values():
                calls.update(
                    name for name in task["contracts"] if "mlops_ep_quack" in name
                )
    prefix = "bf16_" if grad_dtype == "float32" else ""
    assert {
        f"torch.ops.mlops_ep_quack_v1.{prefix}forward.default",
        f"torch.ops.mlops_ep_quack_v1.{prefix}backward.default",
    } <= calls
    (tmp_path / "provider.json").write_text(
        json.dumps(
            {
                "ranks": 2,
                "calls": sorted(calls),
                "gradient_dtype": grad_dtype,
                "parameter_elements": sum(
                    np.size(value)
                    for value in reference["oracles"]["whole"][0]["weights"].values()
                ),
            },
            indent=2,
        )
        + "\n"
    )
