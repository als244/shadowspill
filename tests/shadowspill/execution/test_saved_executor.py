import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    measure_program,
    plan_program,
)
from shadowspill.artifacts._values import plain
from shadowspill.build import build_forward, build_forward_backward, build_step
from shadowspill.ir import ResourceKind
from tests.shadowspill.build.test_forward import Network as ForwardNetwork
from tests.shadowspill.build.test_training import Network, ResidualNetwork, objective
from tests.shadowspill.execution._oracle import oracle


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,variant,tied",
    [
        ("forward", None, False),
        ("forward", "streams", False),
        ("forward_backward", "save", False),
        ("forward_backward", "recompute", True),
        ("step", "save", True),
        ("step", "recompute", False),
        ("step_tensor_lr", "save", False),
        ("step_resume", "recompute", True),
        ("backward_resume", "save", False),
        ("step_residual", "save", True),
        ("backward_residual", "recompute", True),
    ],
)
def test_saved_plan_executes_without_build_or_measurement(
    tmp_path, kind, variant, tied
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(17)
    network = ResidualNetwork if kind.endswith("residual") else Network
    model = ForwardNetwork() if kind == "forward" else network(tied=tied)
    examples = (
        [((torch.randn(4, 8),), {"options": {"scale": 2}})]
        if kind == "forward"
        else [((torch.randn(4, 8),), {"labels": torch.randn(4, 8)}) for _ in range(2)]
    )
    with torch.device("meta"):
        template = ForwardNetwork() if kind == "forward" else network(tied=tied)
    common = dict(example_args=examples[0][0], example_kwargs=examples[0][1])
    if kind == "forward":
        program = build_forward(template, **common)
    else:
        common.update(objective=objective, num_examples=2)
        if kind.startswith("step"):
            program = build_step(
                template,
                optimizer=torch.optim.AdamW(
                    template.parameters(),
                    foreach=False,
                    lr=torch.tensor(0.001) if kind == "step_tensor_lr" else 0.001,
                ),
                **common,
            )
        else:
            program = build_forward_backward(template, **common)
    if variant == "streams":
        program = replace(
            program,
            tasks=tuple(
                replace(
                    task,
                    resource=replace(
                        task.resource,
                        kind=(
                            ResourceKind.COMMUNICATION
                            if index % 3 == 1
                            else ResourceKind.COMPUTE
                        ),
                        lane=int(index % 3 == 2),
                    ),
                )
                for index, task in enumerate(program.tasks)
            ),
        )
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    if variant == "streams":
        assert len(profiles.metadata["resource_streams"]) > 1
        assert profiles.metadata["stream_workspace_bytes"] > 0
    elif variant:
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
    oracle_kind = "forward_backward" if kind == "backward_residual" else kind
    expected = oracle(model, examples, oracle_kind)
    expected["check_task_streams"] = variant == "streams"
    torch.save(expected, tmp_path / "oracle.pt")
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
