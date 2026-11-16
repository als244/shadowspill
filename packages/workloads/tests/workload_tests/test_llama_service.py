"""Public BF16 text training with FP32 masters and an independent eager oracle."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill_client import LoRA, Objective
from shadowspill_workloads.initialization.text import Initialize
from shadowspill_workloads.models.llama3.config import Config
from shadowspill_workloads.models.llama3.implementations.pytorch import (
    Model as Reference,
)
from workload_tests._llama_oracle import WeightGradients
from workload_tests.test_llama_definition import settings


def options(policy=False):
    requested = replace(
        settings(lora=LoRA(2, targets=["blocks.*.attn.wq"]) if policy else None),
        precision={"compute_dtype": "bfloat16", "lora_dtype": "bfloat16"},
    )
    architecture = Config(**requested.config["architecture"])
    model = Reference(architecture).bfloat16()
    if policy:
        from shadowspill_workloads.lora import configure_lora

        model = configure_lora(model, rank=2, alpha=2, targets=requested.lora.targets)
    fill = Initialize("llama3", architecture, seed=17)
    with torch.no_grad():
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            fill(
                TensorChunk(name, tuple(value.shape), value.dtype, 0, value.numel()),
                value.reshape(-1),
            )
    mask = "response_mask" if policy else "loss_mask"
    records = []
    for tokens, targets, active, weights, advantages, old in (
        (
            [1, 2, 3],
            [2, 0, 4],
            [True, False, True],
            [0.5, 1, 1],
            [1, 0, -0.7],
            [-3.3, -3, -2.8],
        ),
        ([4, 5], [5, 6], [True, True], [-0.25, 0.75], [0.8, -0.5], [-2.6, -3.3]),
        ([6], [7], [True], [0], [0.5], [-3.4]),
    ):
        fields = {"target_tokens": targets, mask: active, "weights": weights}
        if policy:
            fields.update(
                old_logprobs=old,
                advantages=advantages,
                reference_logprobs=[-3.0] * len(tokens),
            )
        records.append({"inputs": {"tokens": tokens}, "loss_inputs": fields})
    controls = {"clip_ratio": 0.17, "reference_beta": 0.03} if policy else {}

    def loss(rows):
        # One packed projection accumulates each parameter VJP before its BF16
        # cast. Separate model calls would round every sequence's weight VJP
        # first, losing accuracy under signed-weight cancellation. Attention is
        # still independent per sequence in the pure PyTorch implementation.
        lengths = tuple(len(row["inputs"]["tokens"]) for row in rows)
        tokens = [token for row in rows for token in row["inputs"]["tokens"]]
        fields = {
            name: [value for row in rows for value in row["loss_inputs"][name]]
            for name in rows[0]["loss_inputs"]
        }
        logits = model(torch.tensor([tokens]), lengths).float()[0]
        current = (
            logits.log_softmax(-1)
            .gather(-1, torch.tensor(fields["target_tokens"]).unsqueeze(-1))
            .squeeze(-1)
        )
        if policy:
            ratio = (current - torch.tensor(fields["old_logprobs"])).exp()
            advantages = torch.tensor(fields["advantages"])
            term = -torch.minimum(
                ratio * advantages,
                ratio.clamp(1 - controls["clip_ratio"], 1 + controls["clip_ratio"])
                * advantages,
            )
            delta = torch.tensor(fields["reference_logprobs"]) - current
            term = term + controls["reference_beta"] * (delta.exp() - 1 - delta)
        else:
            term = -current
        return (
            term * torch.tensor(fields[mask]) * torch.tensor(fields["weights"])
        ).sum()

    trained = {
        name: value for name, value in model.named_parameters() if value.requires_grad
    }
    masters = {
        name: torch.nn.Parameter(value.detach().float().clone())
        for name, value in trained.items()
    }
    optim = {
        "lr": 0.002,
        "beta1": 0.8,
        "beta2": 0.95,
        "eps": 0.001,
        "weight_decay": 0.01,
    }
    # Runtime controls are FP32 tensors in the prepared updater.
    optim = {name: float(torch.tensor(value)) for name, value in optim.items()}
    optimizer = torch.optim.AdamW(
        masters.values(),
        lr=optim["lr"],
        betas=(optim["beta1"], optim["beta2"]),
        eps=optim["eps"],
        weight_decay=optim["weight_decay"],
        foreach=False,
    )

    def snapshot():
        weights = {
            name: masters.get(name, value).detach().float().tolist()
            for name, value in model.named_parameters()
        }
        with torch.no_grad():
            predictions = [
                model(torch.tensor([row["inputs"]["tokens"]]))
                .float()[0]
                .log_softmax(-1)
                .gather(
                    -1, torch.tensor(row["loss_inputs"]["target_tokens"]).unsqueeze(-1)
                )
                .squeeze(-1)
                .tolist()
                for row in records
            ]
        return {"weights": weights, "logprobs": predictions}

    parts = [float(loss(rows).detach()) for rows in (records[:1], records[1:])]
    oracle = [snapshot()]
    gradients = WeightGradients(model)
    for scale in (0.25, 1):
        loss(records).backward()
        assert gradients.values.keys() == trained.keys()
        for name, value in trained.items():
            masters[name].grad = gradients.values[name] * scale
            value.grad = None
        gradients.values.clear()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in trained.items():
                value.copy_(masters[name])
        oracle.append(snapshot())
    gradients.close()
    return {
        "model": requested.to_dict(),
        "objective": Objective(
            "shadowspill/clipped_policy@1", config={"reference_penalty": "k3"}
        ).to_dict()
        if policy
        else Objective("shadowspill/cross_entropy@1").to_dict(),
        "optim": optim,
        "controls": controls,
        "records": records,
        "mask": mask,
        "trainable": list(trained),
        "oracle": oracle,
        "loss_parts": parts,
        "client_script": "_text_service_client.py",
        "model_description": "supplied packed Llama, BF16 compute and FP32 masters"
        + (", LoRA policy" if policy else ", full CE"),
        "oracle_description": (
            "independent PyTorch Llama packed autograd, explicit FP32 weight VJPs, "
            "per-sequence logits and FP32 AdamW"
        ),
    }


@pytest.mark.parametrize("policy", [False, True])
def test_llama_eager_oracle_changes_only_trainable_parameters_and_stays_finite(policy):
    value = options(policy)
    initial, first, final = [row["weights"] for row in value["oracle"]]
    changed = set()
    for name in initial:
        before, after = torch.tensor(initial[name]), torch.tensor(final[name])
        assert torch.isfinite(after).all()
        if not torch.equal(before, after):
            changed.add(name)
    assert changed and changed <= set(value["trainable"])
    if policy:
        assert all("lora_" in name for name in changed)
    assert json.loads(json.dumps(value)) == value
    assert first != initial and final != first


@pytest.mark.cuda
@pytest.mark.parametrize("policy", [False, True])
def test_supplied_llama_public_step_split_masters_and_resume(tmp_path, policy):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    (tmp_path / "setup.json").write_text(json.dumps(options(policy)))
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
    assert receipt["whole_split_equivalence"] and receipt["evaluation_isolation"]
    assert receipt["adamw_resume"] and receipt["separate_fp32_masters"]
