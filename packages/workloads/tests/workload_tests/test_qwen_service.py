"""Public MoE training compares each operation geometry with its own oracle."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.training.definitions import load_registry
from shadowspill_client import Objective
from shadowspill_workloads.recipes.text.initialization import Initialize
from workload_tests.test_moe_objective import independent
from workload_tests.test_qwen_definition import settings


def options(family="qwen3moe", *, lora=False):
    requested = settings(family, lora=lora)
    registry = load_registry(["shadowspill-workloads"])
    description = registry.resolve_model(requested.definition, settings=requested)
    initial = description.create()
    fill = Initialize(family, initial.config, seed=17)
    with torch.no_grad():
        for name, tensor in (*initial.named_parameters(), *initial.named_buffers()):
            fill(
                TensorChunk(name, tuple(tensor.shape), tensor.dtype, 0, tensor.numel()),
                tensor.reshape(-1),
            )
    records = [
        {
            "inputs": {"tokens": [1, 2, 3]},
            "loss_inputs": {
                "target_tokens": [2, 0, 4],
                "loss_mask": [True, False, True],
                "weights": [1.0, 1.0, -0.5],
            },
        },
        {
            "inputs": {"tokens": [4, 5]},
            "loss_inputs": {
                "target_tokens": [5, 6],
                "loss_mask": [True, True],
                "weights": [0.0, 1.2],
            },
        },
    ]
    coefficient = 0.07
    optim = {
        "lr": 0.002,
        "beta1": 0.8,
        "beta2": 0.95,
        "eps": 0.001,
        "weight_decay": 0.01,
    }
    optim = {name: float(torch.tensor(value)) for name, value in optim.items()}

    def loss(model, rows):
        lengths = [len(row["inputs"]["tokens"]) for row in rows]
        tokens = [token for row in rows for token in row["inputs"]["tokens"]]
        padding = 8 - len(tokens)
        inputs = {
            "tokens": torch.tensor([tokens + [0] * padding]),
            "sequence_lengths": torch.tensor(
                [*lengths, padding, *([0] * (4 - len(lengths)))], dtype=torch.int32
            ),
        }
        fields = {
            name: torch.tensor(
                [
                    [value for row in rows for value in row["loss_inputs"][name]]
                    + [False if name == "loss_mask" else 0] * padding
                ]
            )
            for name in rows[0]["loss_inputs"]
        }
        fields["router_mask"] = torch.arange(8)[None, :] < len(tokens)
        return independent(model, inputs, fields, coefficient, include_padding=False)[0]

    def snapshot(model):
        with torch.no_grad():
            logprobs = [
                model(torch.tensor([row["inputs"]["tokens"]]))
                .float()[0]
                .log_softmax(-1)
                .gather(
                    -1, torch.tensor(row["loss_inputs"]["target_tokens"])[..., None]
                )
                .squeeze(-1)
                .tolist()
                for row in records
            ]
        return {
            "weights": {
                name: tensor.detach().float().tolist()
                for name, tensor in model.named_parameters()
            },
            "logprobs": logprobs,
        }

    parts = [float(loss(initial, rows).detach()) for rows in (records[:1], records[1:])]
    whole_loss = float(loss(initial, records).detach())
    oracles = {}
    for split in (False, True):
        model = deepcopy(initial)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=optim["lr"],
            betas=(optim["beta1"], optim["beta2"]),
            eps=optim["eps"],
            weight_decay=optim["weight_decay"],
            foreach=False,
        )
        trajectory = [snapshot(model)]
        for index, scale in enumerate((0.25, 1.0)):
            rows = (records[:1], records[1:]) if split and index == 0 else (records,)
            for part in rows:
                (loss(model, part) * scale).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trajectory.append(snapshot(model))
        oracles["split" if split else "whole"] = trajectory
    return {
        "model": requested.to_dict(),
        "objective": Objective.sum(
            {
                "tokens": Objective("shadowspill/cross_entropy@1"),
                "balance": Objective("shadowspill/moe_balance@1"),
            },
            weights={"balance": 0.001},
        ).to_dict(),
        "optim": optim,
        "controls": {"balance.weight": coefficient},
        "records": records,
        "mask": "loss_mask",
        "trainable": [
            name for name, tensor in initial.named_parameters() if tensor.requires_grad
        ],
        "oracles": oracles,
        "whole_loss": whole_loss,
        "loss_parts": parts,
        "whole_split_equivalence": False,
        "output_name": "tokens.logprobs",
        "microbatch_sizes": [8],
        "microbatch_counts": [1],
        # The IEEE FLA kernels retain 4.7 GiB in the Chicago measurement. Honor
        # that admission result instead of lowering the recorded overhead.
        "device_gib": 6 if family == "qwen35moe" else 2,
        "weight_rtol": 0.003,
        "weight_atol": 2e-6,
        "logprob_rtol": 1e-5,
        "logprob_atol": 2e-5,
        "loss_rtol": 1e-5,
        "loss_atol": 2e-5,
        "separate_fp32_masters": False,
        "exact_resume": True,
        "client_script": "_text_service_client.py",
        "model_description": (
            f"public {family}, FP32 compute/gradients/masters, "
            "composed CE and padding-excluded balancing"
        )
        + (", LoRA" if lora else ", full training"),
        "oracle_description": (
            "independent full vocabulary projection and explicit router formula, "
            "CPU autograd/AdamW, separate fixed-geometry whole/split trajectories"
        ),
    }


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe"])
def test_qwen_service_oracles_preserve_frozen_weights_and_distinguish_geometries(
    family, lora
):
    value = options(family, lora=lora)
    assert json.loads(json.dumps(value)) == value
    assert abs(value["whole_loss"] - sum(value["loss_parts"])) > 1e-5
    trained = set(value["trainable"])
    for trajectory in value["oracles"].values():
        initial, _, final = [row["weights"] for row in trajectory]
        changed = set()
        for name in initial:
            before, after = torch.tensor(initial[name]), torch.tensor(final[name])
            assert torch.isfinite(after).all()
            if not torch.equal(before, after):
                changed.add(name)
        assert changed and changed <= trained
        if lora:
            assert all("lora_" in name for name in changed)


@pytest.mark.cuda
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe"])
def test_public_qwen_step_split_and_checkpoint_resume_match_independent_oracles(
    tmp_path, monkeypatch, family, lora
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Match the independent CPU oracle's FP32 products. FLA's default TF32
    # products change small recurrent gradients even in ordinary eager execution.
    # This scopes arithmetic for qualification; production provider defaults and
    # the separate BF16 capture checks remain unchanged.
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    (tmp_path / "setup.json").write_text(json.dumps(options(family, lora=lora)))
    server = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_catalog_service_server.py")),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=900,
    )
    (tmp_path / "server.stdout").write_text(server.stdout)
    (tmp_path / "server.stderr").write_text(server.stderr)
    assert server.returncode == 0, server.stdout + server.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["all_client_claims_released"] and receipt["all_server_claims_released"]
    )
    assert receipt["independent_geometry_oracles"] and receipt["evaluation_isolation"]
    assert receipt["adamw_resume"] and receipt["same_dtype_master_reuse"]
    assert receipt["exact_resume"]
