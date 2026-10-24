import copy
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils._pytree import tree_map

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    measure_program,
    plan_program,
)
from shadowspill.artifacts._values import plain
from shadowspill.build import build_forward, build_forward_backward, build_step
from tests.shadowspill.build.test_forward import Network as ForwardNetwork
from tests.shadowspill.build.test_training import Network, objective


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,variant,tied",
    [
        ("forward", None, False),
        ("forward_backward", "save", False),
        ("forward_backward", "recompute", True),
        ("step", "save", True),
        ("step", "recompute", False),
    ],
)
def test_saved_plan_executes_without_build_or_measurement(
    tmp_path, kind, variant, tied
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(17)
    model = ForwardNetwork() if kind == "forward" else Network(tied=tied)
    initial = copy.deepcopy(model.state_dict())
    examples = (
        [((torch.randn(4, 8),), {"options": {"scale": 2}})]
        if kind == "forward"
        else [((torch.randn(4, 8),), {"labels": torch.randn(4, 8)}) for _ in range(2)]
    )
    with torch.device("meta"):
        template = ForwardNetwork() if kind == "forward" else Network(tied=tied)
    common = dict(example_args=examples[0][0], example_kwargs=examples[0][1])
    if kind == "forward":
        program = build_forward(template, **common)
    else:
        common.update(objective=objective, num_examples=2)
        if kind == "step":
            program = build_step(
                template,
                optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
                **common,
            )
        else:
            program = build_forward_backward(template, **common)
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    if variant:
        bindings = plain(program.bindings)
        bindings["training_templates"] = [
            [
                [
                    f
                    for f in stage
                    if f["variant"]
                    == ("save" if index == len(example) - 1 else variant)
                ]
                for index, stage in enumerate(example)
            ]
            for example in bindings["training_templates"]
        ]
        program = replace(program, bindings=bindings)
    plan = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(
            fetch_bytes_per_second=1e10, evict_bytes_per_second=1e10
        ),
    )
    plan.save(tmp_path / "plan.ssp")
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    expected = []
    for _ in range(3):
        state = {}
        if kind == "forward":
            outputs = model(*examples[0][0], **examples[0][1])
        else:
            optimizer.zero_grad(set_to_none=True)
            outputs = [objective(model, *args, **kwargs) for args, kwargs in examples]
            for loss, _ in outputs:
                loss.backward()
            if kind == "step":
                optimizer.step()
                for name, parameter in model.named_parameters():
                    for entry, value in optimizer.state[parameter].items():
                        state[f"optimizer.{name}.{entry}"] = value.detach().clone()
            else:
                state.update(
                    {
                        f"gradient.{name}": p.grad.clone()
                        for name, p in model.named_parameters()
                    }
                )
        state.update(
            {name: value.detach().clone() for name, value in model.named_parameters()}
        )
        expected.append(
            (
                tree_map(
                    lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else v,
                    outputs,
                ),
                state,
            )
        )
    torch.save(
        dict(initial=initial, examples=examples, expected=expected),
        tmp_path / "oracle.pt",
    )
    worker = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("_execute.py")),
            str(tmp_path),
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
    assert "Qualified three saved-plan invocations" in worker.stdout
