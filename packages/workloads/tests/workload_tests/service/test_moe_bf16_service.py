"""Public BF16 MoE provider preparation, independent updates and saved-Plan reuse."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.artifacts.bundle import TrainingBundle
from shadowspill_workloads.initialization.decoder import Initialize
from workload_tests.definitions.test_moe_definitions import settings
from workload_tests.objectives.test_moe_objective import independent
from workload_tests.service.oracles._moe_oracle import WeightGradients
from workload_tests.service.test_moe_service import options


def bf16_options(lora, *, lora_rank=2, grad_dtype="bfloat16"):
    assert grad_dtype in ("bfloat16", "float32")
    value = options(lora=lora)
    # Keep tiny, signed BF16 derivatives well conditioned through AdamW. With
    # eps=1e-3, a <0.6% gradient-norm difference at one nearly cancelled LoRA
    # coordinate becomes >3% update error. This epsilon still checks gradient
    # scaling and moments against independent equations without that amplification.
    value["optim"]["eps"] = float(torch.tensor(0.01))
    requested = settings("qwen3moe", dtype="bfloat16", lora=lora)
    if lora:
        requested = replace(
            requested,
            lora=replace(requested.lora, rank=lora_rank, alpha=float(lora_rank)),
        )
    requested = replace(
        requested,
        precision={
            **requested.precision,
            "grad_dtype": grad_dtype,
            "lora_dtype": "bfloat16",
        },
    )
    registry = load_registry(["shadowspill-workloads"])
    initial = registry.resolve_model(requested.definition, settings=requested).create()
    fill = Initialize("qwen3moe", initial.config, seed=17)
    with torch.no_grad():
        for name, tensor in (*initial.named_parameters(), *initial.named_buffers()):
            fill(
                TensorChunk(name, tuple(tensor.shape), tensor.dtype, 0, tensor.numel()),
                tensor.reshape(-1),
            )
    records = value["records"]

    def loss(model, rows, observer=None):
        tokens = [token for row in rows for token in row["inputs"]["tokens"]]
        padding = 8 - len(tokens)
        inputs = {
            "tokens": torch.tensor([tokens + [0] * padding]),
            "sequence_lengths": torch.tensor(
                [len(row["inputs"]["tokens"]) for row in rows]
                + [padding]
                + [0] * (4 - len(rows)),
                dtype=torch.int32,
            ),
        }
        fields = {
            name: torch.tensor(
                [[v for row in rows for v in row["loss_inputs"][name]] + [0] * padding]
            )
            for name in rows[0]["loss_inputs"]
        }
        fields["loss_mask"] = fields["loss_mask"].bool()
        fields["router_mask"] = torch.arange(8)[None, :] < len(tokens)
        return independent(
            model,
            inputs,
            fields,
            value["controls"]["balance.weight"],
            include_padding=False,
            router_observer=observer.router if observer is not None else None,
        )[0]

    def snapshot(model, masters):
        with torch.no_grad():
            return {
                "weights": {
                    name: masters.get(name, tensor).detach().float().tolist()
                    for name, tensor in model.named_parameters()
                },
                "logprobs": [
                    model(torch.tensor([row["inputs"]["tokens"]]))
                    .float()[0]
                    .log_softmax(-1)
                    .gather(
                        -1, torch.tensor(row["loss_inputs"]["target_tokens"])[..., None]
                    )
                    .squeeze(-1)
                    .tolist()
                    for row in records
                ],
            }

    oracles = {}
    for split in (False, True):
        model = deepcopy(initial)
        masters = {
            name: torch.nn.Parameter(tensor.detach().float().clone())
            for name, tensor in model.named_parameters()
            if tensor.requires_grad
        }
        config = value["optim"]
        optimizer = torch.optim.AdamW(
            list(masters.values()),
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
            eps=config["eps"],
            weight_decay=config["weight_decay"],
            foreach=False,
        )
        trajectory = [snapshot(model, masters)]
        observer = WeightGradients(model) if grad_dtype == "float32" else None
        for index, scale in enumerate((0.25, 1.0)):
            for rows in (
                (records[:1], records[1:]) if split and index == 0 else (records,)
            ):
                loss(model, rows, observer).backward()
            if observer is not None:
                assert observer.values.keys() == masters.keys()
            for name, parameter in model.named_parameters():
                if name in masters:
                    gradient = (
                        parameter.grad.float()
                        if observer is None
                        else observer.values[name]
                    )
                    masters[name].grad = gradient * scale
            if observer is not None:
                observer.values.clear()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in masters:
                        parameter.copy_(masters[name])
            model.zero_grad(set_to_none=True)
            trajectory.append(snapshot(model, masters))
        if observer is not None:
            observer.close()
        oracles["split" if split else "whole"] = trajectory
    return {
        **value,
        "model": requested.to_dict(),
        "oracles": oracles,
        "whole_loss": float(loss(initial, records).detach()),
        "loss_parts": [
            float(loss(initial, rows).detach()) for rows in (records[:1], records[1:])
        ],
        "gradient_dtype": grad_dtype,
        "separate_fp32_masters": True,
        "device_gib": 6,
        "weight_rtol": 0.04,
        "weight_atol": 2e-5,
        # Eager GPU/CPU and captured GPU derivatives differ at BF16 cancellation
        # boundaries. Compare updates (never full weights) with a per-parameter
        # norm bound as well as an element bound; large parameters cannot hide
        # omitted or incorrectly scaled optimizer updates.
        "weight_norm_rtol": 0.03,
        "weight_max_rtol": 0.20,
        "logprob_rtol": 0.003,
        "logprob_atol": 0.008,
        "loss_rtol": 0.003,
        "loss_atol": 0.008,
        "model_description": (
            f"Qwen3 BF16 compute/{grad_dtype} gradients, FP32 AdamW masters/moments"
        )
        + (f", LoRA rank {lora_rank}" if lora else ", full training"),
        "oracle_description": (
            "independent CPU BF16 activations, explicit router formula and FP32 "
            "master AdamW; separate whole/split trajectories; "
            + (
                "FP32 weight reductions observed before BF16 leaf rounding"
                if grad_dtype == "float32"
                else "BF16 autograd gradients"
            )
        ),
    }


@pytest.mark.parametrize("lora,rank", [(False, 2), (True, 2), (True, 16)])
@pytest.mark.parametrize("grad_dtype", ["bfloat16", "float32"])
def test_bf16_oracle_keeps_master_precision_and_independent_trajectories(
    lora, rank, grad_dtype
):
    value = bf16_options(lora, lora_rank=rank, grad_dtype=grad_dtype)
    assert json.loads(json.dumps(value)) == value
    initial = value["oracles"]["whole"][0]["weights"]
    for trajectory in value["oracles"].values():
        changed = set()
        precise = False
        for name, after in trajectory[2]["weights"].items():
            actual, before = torch.tensor(after), torch.tensor(initial[name])
            assert torch.isfinite(actual).all()
            if not torch.equal(actual, before):
                changed.add(name)
                precise |= not torch.equal(actual, actual.bfloat16().float())
        assert changed and changed <= set(value["trainable"]) and precise
    assert value["oracles"]["whole"][1] != value["oracles"]["split"][1]


def qualify(tmp_path, options, *, lora):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    (tmp_path / "setup.json").write_text(json.dumps(options))
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "helpers/_catalog_service_server.py"),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=1200,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert receipt["independent_geometry_oracles"] and receipt["exact_resume"]
    assert receipt["evaluation_isolation"] and receipt["separate_fp32_masters"]
    source = (
        tmp_path
        / "service/artifacts"
        / hashlib.sha256(b"owner").hexdigest()
        / "data"
        / hashlib.sha256(receipt["plans"]["id"].encode()).hexdigest()
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt["plans"]["sha256"]
    bundle = TrainingBundle.load(source)
    calls, saved = set(), []
    for plan in bundle.plans[0].values():
        bindings = plan.program.bindings
        for task in bindings["allocation_coverage"]["tasks"].values():
            calls.update(name for name in task["contracts"] if "sonicmoe" in name)
        saved.extend(
            manifest
            for task in bindings.get("operator_preparation", {})
            .get("tasks", {})
            .values()
            for manifest in task
            if manifest.get("provider") == "sonic"
            and not manifest["manifest"].get("empty")
        )
    expected_sonic = torch.cuda.get_device_capability()[0] in (9, 10)
    if expected_sonic:
        assert calls and saved, "supported device did not exercise saved Sonic kernels"
        assert any("lora_sonicmoe" in name for name in calls) == lora
    else:
        assert not calls and not saved
    (tmp_path / "provider.json").write_text(
        json.dumps(
            {
                "sonic": expected_sonic,
                "calls": sorted(calls),
                "saved_tasks": len(saved),
            },
            indent=2,
        )
        + "\n"
    )
    return bundle


@pytest.mark.cuda
@pytest.mark.parametrize("lora", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_public_bf16_moe_provider_step_split_and_saved_plan_resume(tmp_path, lora):
    qualify(tmp_path, bf16_options(lora), lora=lora)
