"""Actual Qwen full/LoRA CLIs, independent updates and cold checkpoint resume."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from shadowspill.training.definitions import load_registry
from shadowspill_client import LoRA
from shadowspill_workloads.initialization.text import Initialize
from workload_tests.test_moe_definitions import settings
from workload_tests.test_moe_objective import independent
from workload_tests.test_pretrained_initialization import checkpoint, fill


def prepare(root, pieces, family="qwen35moe"):
    lora = family == "qwen35moe"
    requested = settings(family, lora=lora)
    architecture = {
        **requested.config["architecture"],
        "max_seq_len": 4,
        "router_aux_loss_coef": 0.07,
    }
    requested = replace(
        requested,
        lora=LoRA(2) if lora else None,
        config={"architecture": architecture, **({"lora_seed": 45} if lora else {})},
    )
    registry = load_registry(["shadowspill-workloads"])
    base = registry.resolve_model(
        requested.definition,
        settings=replace(requested, lora=None, config={"architecture": architecture}),
    ).create()
    source = root / "base"
    source.mkdir()
    generator = Initialize(family, base.config, seed=17)
    for name, value in (*base.named_parameters(), *base.named_buffers()):
        fill(generator, name, value, 257)
    checkpoint(source, family, base)
    model = registry.resolve_model(requested.definition, settings=requested).create()
    generator = Initialize(
        family, model.config, seed=17, lora_seed=45 if lora else None
    )
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        fill(generator, name, value, 257)
    trainable = {
        name: value for name, value in model.named_parameters() if value.requires_grad
    }
    initial = {name: value.detach().tolist() for name, value in trainable.items()}
    optimizer = torch.optim.AdamW(
        list(trainable.values()),
        lr=0.001,
        betas=(float(torch.tensor(0.9)), float(torch.tensor(0.999))),
        eps=float(torch.tensor(1e-8)),
        weight_decay=float(torch.tensor(0.01)),
        foreach=False,
    )
    tokens = np.arange(97, dtype=np.int64) % 16

    def loss(start, count, *, validation=False):
        values = (tokens + 1) % 16 if validation else tokens
        inputs = {
            "tokens": torch.tensor(
                [values[start : start + count].tolist() + [0] * (8 - count)]
            ),
            "sequence_lengths": torch.tensor(
                [4] * (count // 4) + ([8 - count] if count < 8 else []) + [0, 0],
                dtype=torch.int32,
            ),
        }
        valid = torch.arange(8)[None, :] < count
        fields = {
            "target_tokens": torch.tensor(
                [values[start + 1 : start + count + 1].tolist() + [0] * (8 - count)]
            ),
            "weights": torch.ones(1, 8),
            "loss_mask": valid,
            "router_mask": valid,
        }
        return independent(
            model, inputs, fields, float(torch.tensor(0.07)), include_padding=False
        )[0]

    losses, evaluations = [], []
    # The existing one-step warmup/cosine schedule for three updates.
    rates = [0.001, 0.001, 0.0001]
    for index, rate in enumerate(rates):
        optimizer.param_groups[0]["lr"] = float(torch.tensor(rate))
        total = 0.0
        for part in range(pieces):
            count = 8 // pieces
            value = loss(index * 8 + part * count, count) / 8
            total += float(value.detach())
            value.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(total)
        with torch.no_grad():
            evaluations.append(float(loss(0, 8, validation=True) / 8))
    return {
        "client_script": "_qwen_example_client.py",
        "checkpoint": str(source),
        "family": family,
        "lora": lora,
        "pieces": pieces,
        "initial": initial,
        "final": {name: value.detach().tolist() for name, value in trainable.items()},
        "losses": losses,
        "evaluations": evaluations,
        "rates": rates,
        "fresh_saved_plans": True,
        "model_description": (
            f"Actual {family} {'LoRA' if lora else 'full'} application, "
            f"FP32, {pieces} backward pieces"
        ),
        "oracle_description": (
            "Independent full-head CE/router formula and CPU AdamW; local weights, "
            "exact native resume"
        ),
    }


@pytest.mark.parametrize("pieces", [1, 2])
@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe"])
def test_qwen_application_reference_has_finite_nonzero_updates(
    tmp_path, pieces, family
):
    options = prepare(tmp_path, pieces, family)
    assert all(np.isfinite(options["losses"]))
    assert all(np.isfinite(options["evaluations"]))
    assert any(
        not np.array_equal(value, options["initial"][name])
        for name, value in options["final"].items()
    )
    assert json.loads(json.dumps(options)) == options


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("pieces", [1, 2])
@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe"])
def test_qwen_application_step_split_and_checkpoint_resume(
    tmp_path, monkeypatch, pieces, family
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    (tmp_path / "setup.json").write_text(json.dumps(prepare(tmp_path, pieces, family)))
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_catalog_service_server.py")),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=900,
    )
    (tmp_path / "server.stdout").write_text(child.stdout)
    (tmp_path / "server.stderr").write_text(child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["actual_example"] and receipt["independent_loss_and_weights"]
    assert (
        receipt["exact_checkpoint_continuation"]
        and receipt["all_server_claims_released"]
    )
