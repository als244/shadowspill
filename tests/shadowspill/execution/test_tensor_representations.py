"""Physical payloads/scales and logical gradients survive saved phase boundaries."""

import os
import subprocess
import sys
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
from shadowspill.ir import TaskAlternativeChoice
from tests.shadowspill.pytorch.state.representations import model


def objective(net, sample):
    return net(sample).square().mean()


def save_plan(program, path):
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    plan = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(1e10, 1e10),
    )
    plan.save(path)
    return plan


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_quantized_training_evaluation_and_cold_checkpoint(
    tmp_path, monkeypatch, variant
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    def choices(program, _shares):
        return (
            tuple(
                TaskAlternativeChoice(g.group_id, variant)
                for g in program.task_alternative_groups
            ),
        )

    monkeypatch.setattr(
        "shadowspill.planner.search.algorithms.pressurefit.resolutions", choices
    )
    sample = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
    with torch.device("meta"):
        template = model()
        template.register_buffer("empty", torch.empty(0, 3))
    with pytest.raises(ValueError, match="require master_dtype"):
        build_step(
            template,
            optimizer=torch.optim.SGD(template.parameters(), lr=0.01, foreach=False),
            objective=objective,
            example_args=(sample,),
            partition="whole",
        )
    training = build_step(
        template,
        optimizer=torch.optim.SGD(template.parameters(), lr=0.01, foreach=False),
        objective=objective,
        example_args=(sample,),
        partition="whole",
        master_dtype=torch.float32,
        grad_dtype=torch.float32,
    )
    bindings = {b["name"]: b for b in training.bindings["optimizer_bindings"]}
    assert bindings["weight"]["dtype"] == "torch.float32"
    assert bindings["compute.weight.payload"]["dtype"] == "torch.int8"
    assert bindings["compute.weight.scale"]["dtype"] == "torch.float32"
    assert bindings["weight"]["initialization"]["kind"] == "external"
    selected = save_plan(training, tmp_path / "training.ssp")
    assert selected.selections and all(
        c.option_id == variant for c in selected.selections
    )
    evaluation = build_forward(template, example_args=(sample,), partition="whole")
    save_plan(evaluation, tmp_path / "evaluation.ssp")

    reference = model()
    master = torch.nn.Parameter(reference.weight.dense().detach())
    optimizer = torch.optim.SGD([master], lr=0.01, foreach=False)
    initial = {
        "weight.payload": reference.weight.payload.clone(),
        "weight.scale": reference.weight.scale.clone(),
        "master.weight": master.detach().clone(),
    }
    losses, outputs, states = [], [], []
    for _ in range(4):
        loss = objective(reference, sample)
        master.grad = torch.autograd.grad(loss, reference.weight)[0]
        optimizer.step()
        with torch.no_grad():
            reference.weight.copy_(master)
        losses.append(loss.item())
        outputs.append(reference(sample).detach().clone())
        states.append(
            {
                "weight.payload": reference.weight.payload.clone(),
                "weight.scale": reference.weight.scale.clone(),
                "master.weight": master.detach().clone(),
            }
        )
    torch.save(
        dict(
            initial=initial,
            sample=sample,
            losses=losses,
            outputs=outputs,
            states=states,
        ),
        tmp_path / "oracle.pt",
    )
    for mode in ("train", "resume"):
        worker = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("_representation_execute.py")),
                str(tmp_path),
                mode,
            ],
            env={
                **os.environ,
                "TRITON_CACHE_DIR": str(tmp_path / f"{mode}-triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / f"{mode}-inductor"),
            },
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert worker.returncode == 0, worker.stdout + worker.stderr
