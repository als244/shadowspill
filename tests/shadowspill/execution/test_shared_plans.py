"""Separate saved graphs share current model storage, including mixed precision."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_map

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    build_forward,
    build_step,
    measure_program,
    plan_program,
)
from shadowspill.planning.shared import shared_memory


class Network(nn.Module):
    def __init__(self, *, reverse=False):
        super().__init__()
        parameters = {
            "weight": nn.Parameter(torch.empty(8, 8, dtype=torch.bfloat16)),
            "bias": nn.Parameter(torch.empty(8, dtype=torch.bfloat16)),
        }
        for name in reversed(parameters) if reverse else parameters:
            self.register_parameter(name, parameters[name])
        self.tied_weight = self.weight
        self.register_buffer("scale", torch.empty(8, dtype=torch.bfloat16))
        self.register_buffer("evaluation_only", torch.empty(11))

    def forward(self, x):
        return {
            "prediction": (x @ self.weight + self.bias) * self.scale,
            "view": x[:, ::2],
            "weight": self.tied_weight,
            "extra": self.evaluation_only + 1,
            "constant": torch.tensor([1.0, 2.0], device=x.device),
        }


def objective(model, x):
    output = model(x)["prediction"].float()
    return output.sum(), {"mean": output.mean()}


@pytest.mark.cuda
def test_shared_evaluation_and_checkpoint_in_cold_workers(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("meta"):
        training, evaluation = Network(), Network(reverse=True).eval()
    examples = [(torch.full((4, 8), v, dtype=torch.bfloat16),) for v in (0.125, 0.25)]
    x = torch.arange(512).reshape(64, 8).to(torch.bfloat16) / 512
    programs = {
        "training": build_step(
            training,
            optimizer=torch.optim.AdamW(training.parameters(), lr=0.001, foreach=False),
            objective=objective,
            example_args=examples[0],
            num_examples=2,
            partition="whole",
            master_dtype=torch.float32,
            grad_dtype=torch.float32,
        ),
        "evaluation": build_forward(evaluation, example_args=(x,), partition="whole"),
    }
    plans = {}
    for name, program in programs.items():
        profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
        plans[name] = plan_program(
            apply_profiles(program, profiles),
            memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
            communication_info=CommunicationInfo(1e10, 1e10),
        )
        plans[name].save(tmp_path / f"{name}.ssp")
    shared = shared_memory(plans["training"], {"evaluation": plans["evaluation"]})
    assert any(a != b for a, b in shared.aliases["evaluation"].items())
    assert shared.fixed_slice_bytes == max(
        p.memory_layout.fixed_slice_bytes for p in plans.values()
    )
    # Both fit separately; their retained spill storage must also fit together.
    assert shared.spill_bytes >= max(
        p.timeline.spill_peak_bytes for p in plans.values()
    )
    for field, minimum in (
        (
            "execution_bytes",
            max(
                p.memory_layout.required_bytes + p.profiled_program.fixed_device_bytes
                for p in plans.values()
            ),
        ),
        ("spill_bytes", max(p.timeline.spill_peak_bytes for p in plans.values())),
    ):
        assert minimum < getattr(shared, field)
        budget = replace(plans["training"].memory_budgets, **{field: minimum})
        reduced = {name: replace(p, memory_budgets=budget) for name, p in plans.items()}
        with pytest.raises(ValueError, match="shared Plans require"):
            shared_memory(reduced["training"], {"evaluation": reduced["evaluation"]})

    reference = Network()
    with torch.no_grad():
        reference.weight.copy_(torch.arange(64).reshape(8, 8) / 64)
        reference.bias.fill_(0.125)
        reference.scale.fill_(0.5)
        reference.evaluation_only.fill_(0.75)
    initial = {name: value.clone() for name, value in reference.state_dict().items()}
    masters = {
        name: nn.Parameter(p.float().detach().clone())
        for name, p in reference.named_parameters()
    }
    optimizer = torch.optim.AdamW(masters.values(), lr=0.001, foreach=False)
    expected, losses, final = [], [], {}
    for _ in range(3):
        expected.append(tree_map(lambda t: t.detach().clone(), reference(x)))
        reference.zero_grad(set_to_none=True)
        outputs = [objective(reference, *args)[0] for args in examples]
        losses.append([v.item() for v in outputs])
        for loss in outputs:
            loss.backward()
        for name, p in reference.named_parameters():
            masters[name].grad = p.grad.float()
        optimizer.step()
        with torch.no_grad():
            for name, p in reference.named_parameters():
                p.copy_(masters[name])
    expected.append(tree_map(lambda t: t.detach().clone(), reference(x)))
    for name, parameter in reference.named_parameters():
        final[name] = parameter.detach().clone()
        final[f"master.{name}"] = masters[name].detach().clone()
        final.update(
            {
                f"optimizer.{name}.{key}": value.clone()
                for key, value in optimizer.state[masters[name]].items()
            }
        )
    torch.save(
        dict(
            initial=initial,
            examples=examples,
            evaluation=x,
            expected=expected,
            losses=losses,
            final=final,
        ),
        tmp_path / "oracle.pt",
    )
    for mode in ("direct", "trainer"):
        worker = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("_evaluate.py")),
                str(tmp_path),
                mode,
            ],
            env={
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / f"{mode}-inductor"),
                "TRITON_CACHE_DIR": str(tmp_path / f"{mode}-triton"),
            },
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert worker.returncode == 0, worker.stdout + worker.stderr
        assert "Qualified shared evaluation" in worker.stdout
