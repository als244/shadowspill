"""Prepared public two-H100 EP training with independent numerical references."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from shadowspill import Parallelism
from shadowspill.training.definitions import load_registry
from shadowspill.training.inputs.packing import pack_batch
from shadowspill.training.inputs.validation import validate_batch
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.payload import snapshot_batch
from workload_tests._expert_oracle import (
    SPLIT,
    WHOLE,
    initial_model,
    loss,
    options,
    records,
)
from workload_tests._text_service_client import training_settings


@pytest.fixture(scope="module")
def reference():
    return options()


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
            str(Path(__file__).with_name("_catalog_service_server.py")),
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
    for plans in bundle.plans.values():
        for plan in plans.values():
            for task in plan.program.bindings["allocation_coverage"]["tasks"].values():
                calls.update(
                    name for name in task["contracts"] if "mlops_ep_quack" in name
                )
    assert {
        "torch.ops.mlops_ep_quack_v1.forward.default",
        "torch.ops.mlops_ep_quack_v1.backward.default",
    } <= calls
    (tmp_path / "provider.json").write_text(
        json.dumps(
            {
                "ranks": 2,
                "calls": sorted(calls),
                "gradient_dtype": "bfloat16",
                "parameter_elements": sum(
                    np.size(value)
                    for value in reference["oracles"]["whole"][0]["weights"].values()
                ),
            },
            indent=2,
        )
        + "\n"
    )
