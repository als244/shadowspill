"""Real text updates keep evaluation, schedules and exact saved-Plan resume."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    build_forward,
    build_step,
    measure_program,
    plan_program,
)
from tests.workloads.test_fixed_updates import corpus
from workloads.models.llama3.config import Config
from workloads.models.llama3.implementations.pytorch import Model
from workloads.recipes.text.data.calls import Inputs
from workloads.recipes.text.objectives import normalized_next_token_prediction as loss


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


@pytest.mark.cuda
@pytest.mark.parametrize("partial_checkpoint", [False, True])
def test_fixed_updates_evaluate_log_and_resume_with_identical_next_update(
    tmp_path, partial_checkpoint
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    corpus(tmp_path)
    options = {
        "data": str(tmp_path),
        "sequence_length": 8,
        "sequences_per_step": 4,
        "sequences_per_example": 1,
        "partial_checkpoint": partial_checkpoint,
    }
    (tmp_path / "config.json").write_text(json.dumps(options))
    torch.manual_seed(231)
    model = Model(Config(1, 16, 2, 1, 32, 32, max_seq_len=8))
    if partial_checkpoint:
        model.requires_grad_(False)
        model.lm_head.requires_grad_(True)
    torch.save(
        dict((*model.named_parameters(), *model.named_buffers())),
        tmp_path / "initial.pt",
    )
    examples = Inputs(options).example()
    common = dict(
        example_args=examples["args"][0], example_kwargs=examples["kwargs"][0]
    )
    training = build_step(
        model,
        objective=loss,
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=torch.tensor(0.01), foreach=False
        ),
        num_examples=4,
        **common,
    )
    evaluation = build_forward(model.eval(), forward_fn=loss, **common)
    for name, program in (("training", training), ("evaluation", evaluation)):
        program.save(tmp_path / f"{name}.program.ssp")
        profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
        profiles.save(tmp_path / f"{name}.profiles.ssp")
        plan = plan_program(
            apply_profiles(program, profiles),
            memory_budgets=MemoryBudgets(4 << 30, 64 << 20),
            communication_info=CommunicationInfo(
                fetch_bytes_per_second=1e10, evict_bytes_per_second=1e10
            ),
        )
        plan.save(tmp_path / f"{name}.ssp")

    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    for output in (whole, resumed):
        command = [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            str(Path(__file__).with_name("_fixed_training.py")),
            str(tmp_path),
            str(output),
        ]
        if output == resumed:
            command.append(str(whole / "checkpoints/step_00000004.ssp"))
        worker = subprocess.run(
            command, env=os.environ, capture_output=True, text=True, timeout=180
        )
        assert worker.returncode == 0, worker.stdout + worker.stderr
    expected = checkpoint(whole / "checkpoints/step_00000005.ssp")
    actual = checkpoint(resumed / "checkpoints/step_00000005.ssp")
    if partial_checkpoint:
        assert "embed.weight" in expected["omitted_parameters"]
        assert "lm_head.weight" not in expected["omitted_parameters"]
    assert {x["name"]: x["sha256"] for x in actual["tensors"]} == {
        x["name"]: x["sha256"] for x in expected["tensors"]
    }
    assert sorted(p.name for p in (whole / "checkpoints").iterdir()) == [
        "step_00000004.ssp",
        "step_00000005.ssp",
    ]
    for filename, keys in (
        ("metrics.jsonl", ("train/loss", "hyperparameters/optimizer_group.0.lr")),
        ("evaluation.jsonl", ("eval/loss",)),
    ):
        before, after = records(whole / filename), records(resumed / filename)
        assert [row["step"] for row in before] == (
            [1, 2, 3, 4, 5] if filename == "metrics.jsonl" else [2, 4, 5]
        )
        assert [row["step"] for row in after] == [5]
        assert all(after[0][key] == before[-1][key] for key in keys)
