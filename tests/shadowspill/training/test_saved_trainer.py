"""Numerical training/resume through the public phases and independent workers."""

import copy
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
    build_step,
    measure_program,
    plan_program,
)
from tests.shadowspill.build.test_training import Network, objective
from tests.shadowspill.execution._oracle import optimizer_values


@pytest.mark.cuda
def test_trainer_updates_and_cold_resume_match_independent_eager_loop(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(91)
    model = Network(tied=True)
    initial = copy.deepcopy(model.state_dict())
    examples = [
        [((torch.randn(4, 8),), {"labels": torch.randn(4, 8)}) for _ in range(2)]
        for _ in range(6)
    ]
    rates = [0.001 / (1 + index) for index in range(6)]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=torch.tensor(rates[0]), foreach=False
    )
    losses = []
    for rate, inputs in zip(rates, examples, strict=True):
        optimizer.param_groups[0]["lr"].fill_(rate)
        optimizer.zero_grad(set_to_none=True)
        outputs = [objective(model, *args, **kwargs)[0] for args, kwargs in inputs]
        losses.append([loss.item() for loss in outputs])
        for loss in outputs:
            loss.backward()
        optimizer.step()
    final = {
        **dict(model.named_parameters()),
        **optimizer_values(model, optimizer),
        "optimizer_group.0.lr": optimizer.param_groups[0]["lr"].clone(),
    }
    torch.save(
        dict(
            initial=initial,
            examples=examples,
            rates=rates,
            losses=losses,
            final={name: value.detach().clone() for name, value in final.items()},
        ),
        tmp_path / "oracle.pt",
    )
    with torch.device("meta"):
        template = Network(tied=True)
    program = build_step(
        template,
        optimizer=torch.optim.AdamW(
            template.parameters(), lr=torch.tensor(rates[0]), foreach=False
        ),
        objective=objective,
        example_args=examples[0][0][0],
        example_kwargs=examples[0][0][1],
        num_examples=2,
    )
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    plan = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(
            fetch_bytes_per_second=1e10, evict_bytes_per_second=1e10
        ),
    )
    plan.save(tmp_path / "plan.ssp")
    for mode in ("whole", "part", "resume"):
        worker = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("_train.py")),
                str(tmp_path),
                mode,
            ],
            env={
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "empty-inductor"),
                "TRITON_CACHE_DIR": str(tmp_path / "empty-triton"),
            },
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert worker.returncode == 0, worker.stdout + worker.stderr
        assert "Qualified Trainer " + mode in worker.stdout
